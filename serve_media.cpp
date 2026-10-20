// serve_media.cpp -- the media engines' in-process home.  See
// serve_media.h for the role: PhysModel/Residency own the whisper,
// diffusion and transcribe handles and their load/unload/TTL lifecycle,
// and the main thread's pump() is the only place engine calls run.

#include "serve_media.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// stb_image (decode uploads for /v1/images/edits) and stb_image_write
// (encode generated images to PNG).  Only the declarations are pulled in
// here: the implementations already live in the linked binary via
// llamafile's stb.a (see BUILD.mk).
#include "stb_image.h"
#include "stb_image_write.h"

// Audio decoding for /v1/audio/transcriptions uses miniaudio, the
// single-header decoder bundled with whisper.cpp's examples (the same one
// its read_audio_data() uses); Vorbis support comes through stb_vorbis,
// included exactly like whisper's own examples do it.  Both
// implementations compile into this one translation unit.
#define STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"
// We only decode; keep the rest of miniaudio (device I/O, playback, ...)
// out of the binary, exactly like whisper's own examples do.
#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

namespace serve {

static std::string phys_key(const ModelEntry &e) {
    return std::string(engine_name(e.engine)) + "\x01" + e.file;
}

static void load_whisper(PhysModel &p, std::string &err) {
    p.wctx = whisper_init_from_file(p.file.c_str());
    if (!p.wctx)
        err = "whisper: failed to load '" + p.file + "' (out of memory?)";
}

static void load_diffusion(PhysModel &p, std::string &err) {
    sd_ctx_params_t sp;
    sd_ctx_params_init(&sp);
    sp.model_path = p.file.c_str();
    // One APE constraint: the engine's loader uploads tensors to the
    // GPU from worker threads, and that path crashes (NULL buffer
    // context) in this APE -- the native CLI has the same crash, so the
    // CLI's own --params-backend cpu workaround is the fix here.
    // Weights sit in RAM; the denoise compute still runs on the
    // auto-selected GPU backend.
    sp.params_backend = "cpu";
    // The init default is a decode-only VAE, which trips an assert the
    // first time /v1/images/edits needs to encode an input image; keep
    // the encoder weights around so both endpoints work.
    sp.vae_decode_only = false;
    p.sctx = new_sd_ctx(&sp);
    if (!p.sctx)
        err = "diffusion: failed to load '" + p.file + "' (out of memory?)";
}

static void load_transcribe(PhysModel &p, std::string &err) {
    static std::once_flag backends_once;
    std::call_once(backends_once, [] {
        transcribe_init_backends_default();
    });
    p.tmodel = nullptr;
    transcribe_status st = transcribe_model_load_file(p.file.c_str(), nullptr,
                                                      &p.tmodel);
    if (st != TRANSCRIBE_OK) {
        err = std::string("transcribe: ") + transcribe_status_string(st);
        return;
    }
    // One session per loaded model: it borrows the model and is only ever
    // used from the main thread (see Residency), so the engine's "one
    // thread per session" rule holds by construction.
    transcribe_status ss = transcribe_session_init(p.tmodel, nullptr,
                                                   &p.tsess);
    if (ss != TRANSCRIBE_OK) {
        err = std::string("transcribe: session: ") +
              transcribe_status_string(ss);
        transcribe_model_free(p.tmodel);
        p.tmodel = nullptr;
    }
}

static void load_engine(PhysModel &p, std::string &err) {
    switch (p.engine) {
        case EngineKind::Whisper:    load_whisper(p, err);    break;
        case EngineKind::Diffusion:  load_diffusion(p, err);  break;
        case EngineKind::Transcribe: load_transcribe(p, err); break;
        default: err = "llama models never load in-process"; break;
    }
}

static void unload_engine(PhysModel &p) {
    if (p.wctx) { whisper_free(p.wctx); p.wctx = nullptr; }
    if (p.sctx) { free_sd_ctx(p.sctx); p.sctx = nullptr; }
    if (p.tsess) { transcribe_session_free(p.tsess); p.tsess = nullptr; }
    if (p.tmodel) { transcribe_model_free(p.tmodel); p.tmodel = nullptr; }
}

// ── Audio transcription (whisper + transcribe) ───────────────────────────
//
// One OpenAI transcription request.  The uploaded audio is decoded to
// 16 kHz mono float32 PCM -- the input format both speech engines accept
// -- with miniaudio, then handed to the engine's own full-pass API
// (whisper_full / transcribe_run); the segments come straight back out of
// the engine and are reshaped into the requested response format.  The
// engine work runs on the main thread through an EngineJob; the HTTP
// worker only decodes the upload, waits, and serializes the response.

// Decode an uploaded audio file to 16 kHz mono float32 PCM (WAV, MP3,
// FLAC and Vorbis containers; resampled and mixed down as needed).
// Same pattern as whisper.cpp's read_audio_data: ask for the length, then
// read it all in one shot (reading past the end of a miniaudio decoder
// reports an error, so the exact-length read is the safe one).
bool decode_audio(const std::string &data, std::vector<float> &pcm) {
    if (data.size() < 12)
        return false;
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, 16000);
    ma_decoder dec;
    if (ma_decoder_init_memory(data.data(), data.size(), &cfg, &dec) !=
        MA_SUCCESS)
        return false;
    ma_uint64 frame_count = 0;
    if (ma_decoder_get_length_in_pcm_frames(&dec, &frame_count) != MA_SUCCESS ||
        frame_count == 0) {
        ma_decoder_uninit(&dec);
        return false;
    }
    pcm.resize(frame_count);
    ma_uint64 frames_read = 0;
    bool ok = ma_decoder_read_pcm_frames(&dec, pcm.data(), frame_count,
                                         &frames_read) == MA_SUCCESS;
    pcm.resize(frames_read);
    ma_decoder_uninit(&dec);
    return ok && frames_read > 0;
}

// Run one transcription on a loaded model (main thread only).  pcm is 16
// kHz mono float32, as decode_audio produces.
TranscribeResult run_transcribe(PhysModel &p,
                                       const std::vector<float> &pcm,
                                       const std::string &language,
                                       const std::string &prompt,
                                       float temperature) {
    TranscribeResult out;
    if (pcm.empty()) {
        out.err = "no audio samples to transcribe";
        return out;
    }
    out.duration_s = (double)pcm.size() / 16000.0;

    if (p.engine == EngineKind::Whisper) {
        whisper_full_params wp =
            whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        if (!language.empty())
            wp.language = language.c_str();
        if (!prompt.empty())
            wp.initial_prompt = prompt.c_str();
        if (temperature >= 0.0f)
            wp.temperature = temperature;
        if (whisper_full(p.wctx, wp, pcm.data(), (int)pcm.size()) != 0) {
            out.err = "whisper: transcription failed";
            return out;
        }
        int lid = whisper_full_lang_id(p.wctx);
        if (lid >= 0)
            out.language = whisper_lang_str(lid);
        int n = whisper_full_n_segments(p.wctx);
        for (int i = 0; i < n; i++) {
            const char *t = whisper_full_get_segment_text(p.wctx, i);
            std::string s = t ? t : "";
            // whisper pads segment text with a leading space
            size_t b = s.find_first_not_of(' ');
            if (b == std::string::npos)
                continue;
            s.erase(0, b);
            out.segments.push_back(
                {whisper_full_get_segment_t0(p.wctx, i) / 1000.0,
                 whisper_full_get_segment_t1(p.wctx, i) / 1000.0, s});
            if (!out.text.empty())
                out.text += ' ';
            out.text += s;
        }
    } else {  // EngineKind::Transcribe
        transcribe_run_params rp;
        transcribe_run_params_init(&rp);
        rp.task = TRANSCRIBE_TASK_TRANSCRIBE;
        rp.timestamps = TRANSCRIBE_TIMESTAMPS_AUTO;
        if (!language.empty())
            rp.language = language.c_str();
        transcribe_status st =
            transcribe_run(p.tsess, pcm.data(), (int)pcm.size(), &rp);
        if (st != TRANSCRIBE_OK) {
            out.err = std::string("transcribe: ") +
                      transcribe_status_string(st);
            return out;
        }
        const char *ft = transcribe_full_text(p.tsess);
        if (ft)
            out.text = ft;
        const char *lang = transcribe_detected_language(p.tsess);
        if (lang)
            out.language = lang;
        int n = transcribe_n_segments(p.tsess);
        for (int i = 0; i < n; i++) {
            transcribe_segment seg;
            transcribe_segment_init(&seg);
            if (transcribe_get_segment(p.tsess, i, &seg) != TRANSCRIBE_OK ||
                !seg.text)
                continue;
            out.segments.push_back({seg.t0_ms / 1000.0, seg.t1_ms / 1000.0,
                                    seg.text});
        }
    }
    out.ok = true;
    return out;
}

static std::string fmt_srt_time(double s) {
    long ms = (long)(s * 1000.0 + 0.5);
    char b[32];
    snprintf(b, sizeof b, "%02ld:%02ld:%02ld,%03ld", ms / 3600000,
             (ms / 60000) % 60, (ms / 1000) % 60, ms % 1000);
    return b;
}

static std::string fmt_vtt_time(double s) {
    long ms = (long)(s * 1000.0 + 0.5);
    char b[32];
    snprintf(b, sizeof b, "%02ld:%02ld:%02ld.%03ld", ms / 3600000,
             (ms / 60000) % 60, (ms / 1000) % 60, ms % 1000);
    return b;
}

std::string build_srt(const std::vector<TranscribeResult::Segment> &segs) {
    std::string out;
    int n = 1;
    for (const auto &s : segs) {
        out += std::to_string(n++) + "\n";
        out += fmt_srt_time(s.t0) + " --> " + fmt_srt_time(s.t1) + "\n";
        out += s.text + "\n\n";
    }
    return out;
}

std::string build_vtt(const std::vector<TranscribeResult::Segment> &segs) {
    std::string out = "WEBVTT\n\n";
    for (const auto &s : segs) {
        out += fmt_vtt_time(s.t0) + " --> " + fmt_vtt_time(s.t1) + "\n";
        out += s.text + "\n\n";
    }
    return out;
}

// ── Image generation (diffusion) ─────────────────────────────────────────
//
// One OpenAI image request, built on stable-diffusion.cpp's own
// generate_image(): an sd_img_gen_params_t carries the prompt, size,
// batch and (for /v1/images/edits) the decoded input image, and the raw
// pixels come straight back out of the engine.  As with the other engines,
// the generation runs on the main thread through an EngineJob; the HTTP
// worker only decodes/encodes image bytes, waits, and serializes.
//
// The engine context is one-shot: the pinned engine crashes on a second
// generate_image() against the same context (reproduced outside this
// server with a minimal harness against the engine alone), so each
// request loads a fresh context and frees it again afterwards -- a few
// seconds per request, in exchange for the rest of the design staying
// identical to the other engines.

// Run one generation on a loaded model (main thread only).
// edit is the decoded input image for image edits (nullptr = text-to-image).
ImageResult run_image(PhysModel &p, const std::string &prompt,
                             const std::string &negative, int w, int h,
                             int n, int steps, float cfg, long seed,
                             const std::vector<uint8_t> *edit,
                             int edit_w, int edit_h) {
    ImageResult out;
    // The pinned engine cannot run a second generation on the same
    // context -- the second compute graph crashes, and a minimal
    // harness against the engine alone reproduces it -- so every
    // request gets a fresh context: load it if there is none, free it
    // again once the pixels are out.  The pump serializes engine work,
    // so at most one request touches p.sctx at a time.
    std::string lerr;
    {
        std::lock_guard<std::mutex> lk(p.mu);
        if (!p.sctx)
            load_diffusion(p, lerr);
    }
    if (!p.sctx) {
        out.err = lerr.empty() ? "model not loaded" : lerr;
        return out;
    }
    if (!sd_ctx_supports_image_generation(p.sctx)) {
        out.err = "this model does not support image generation";
        std::lock_guard<std::mutex> lk(p.mu);
        free_sd_ctx(p.sctx);  // never ran a generation, but don't hold it
        p.sctx = nullptr;
        return out;
    }
    sd_img_gen_params_t sp;
    sd_img_gen_params_init(&sp);
    sp.prompt = prompt.c_str();
    if (!negative.empty())
        sp.negative_prompt = negative.c_str();
    sp.width = w;
    sp.height = h;
    sp.batch_count = n;
    if (steps > 0)
        sp.sample_params.sample_steps = steps;
    if (cfg > 0.0f)
        sp.sample_params.guidance.txt_cfg = cfg;
    if (seed >= 0)
        sp.seed = seed;
    if (edit) {
        sp.init_image.width = (uint32_t)edit_w;
        sp.init_image.height = (uint32_t)edit_h;
        sp.init_image.channel = 3;
        // the engine only reads the input image's pixels
        sp.init_image.data = const_cast<uint8_t *>(edit->data());
    }
    sd_image_t *imgs = generate_image(p.sctx, &sp);
    if (!imgs) {
        out.err = "generation failed (out of memory?)";
        std::lock_guard<std::mutex> lk(p.mu);
        free_sd_ctx(p.sctx);  // a failed run may leave it half-set-up
        p.sctx = nullptr;
        return out;
    }
    for (int i = 0; i < n; i++) {
        ImageResult::Img img;
        img.w = (int)imgs[i].width;
        img.h = (int)imgs[i].height;
        img.ch = (int)imgs[i].channel;
        img.px.resize((size_t)img.w * img.h * img.ch);
        if (imgs[i].data)
            std::memcpy(img.px.data(), imgs[i].data, img.px.size());
        free(imgs[i].data);
        out.images.push_back(std::move(img));
    }
    free(imgs);  // the engine's array itself
    {
        // The context is one-shot (see the note at the top): drop it
        // now that the pixels are safely copied out.
        std::lock_guard<std::mutex> lk(p.mu);
        free_sd_ctx(p.sctx);
        p.sctx = nullptr;
    }
    out.ok = true;
    return out;
}

// Encode one image to PNG with stb_image_write (linked in via stb.a).
// The header only declares the _to_func variant, so collect the bytes
// through a callback -- the same trick the engine's own media_io uses.
static void png_mem_write(void *ctx, void *data, int size) {
    auto *s = static_cast<std::string *>(ctx);
    s->append((const char *)data, (size_t)size);
}

bool encode_png(const ImageResult::Img &img, std::string &out) {
    out.clear();
    return stbi_write_png_to_func(png_mem_write, &out, img.w, img.h, img.ch,
                                  img.px.data(), 0) != 0;
}

// Decode an uploaded image (PNG or JPEG) to 3-channel RGB with stb_image.
bool decode_image(const std::string &data, int &w, int &h,
                         std::vector<uint8_t> &px) {
    int pw = 0, ph = 0, pch = 0;
    unsigned char *p = stbi_load_from_memory(
        (const unsigned char *)data.data(), (int)data.size(), &pw, &ph, &pch, 3);
    if (!p)
        return false;
    w = pw;
    h = ph;
    px.assign(p, p + (size_t)pw * ph * 3);
    free(p);
    return true;
}

// One unit of engine work submitted to the main thread's pump.  The pump
// owns the run (in this APE only the main thread has a working host-glibc
// malloc state, which the host-compiled engine libraries allocate
// through); the submitting HTTP worker waits for `finished` before
// reading the result its run closure captured.
// ── Residency ────────────────────────────────────────────────────

void Residency::push_task(LifeTask t) {
    (t.priority ? hi : lo).push_back(std::move(t));
}

std::shared_ptr<PhysModel> Residency::get_or_create(const ModelEntry &e) {
    std::lock_guard<std::mutex> lk(mu);
    std::string k = phys_key(e);
    auto it = phys.find(k);
    if (it != phys.end()) {
        auto p = it->second;
        p->entries.push_back(&e);
        int et = e.effective_ttl();
        p->pinned = p->pinned || e.pinned;
        if (et == 0) p->ttl = 0;
        else if (p->ttl != 0) p->ttl = std::max(p->ttl, et);
        return p;
    }
    auto p = std::make_shared<PhysModel>();
    p->key = k;
    p->engine = e.engine;
    p->file = e.file;
    p->entries.push_back(&e);
    p->pinned = e.pinned;
    p->ttl = e.effective_ttl();
    phys[k] = p;
    return p;
}

std::shared_ptr<PhysModel> Residency::find(const ModelEntry &e) const {
    std::lock_guard<std::mutex> lk(mu);
    auto it = phys.find(phys_key(e));
    return it == phys.end() ? nullptr : it->second;
}

bool Residency::evict_one(const PhysModel &self) {
    std::shared_ptr<PhysModel> victim;
    {
        std::lock_guard<std::mutex> lk(mu);
        long best = -1;
        for (auto &kv : phys) {
            const PhysModel &v = *kv.second;
            if (&v == &self || v.state.load() != 2) continue;
            if (v.pinned || v.in_use.load() > 0) continue;
            if (best < 0 || v.last_used.load() < best) {
                best = v.last_used.load();
                victim = kv.second;
            }
        }
    }
    if (!victim)
        return false;
    {
        std::lock_guard<std::mutex> lk(victim->mu);
        unload_engine(*victim);
        victim->state = 0;
        victim->cv.notify_all();
    }
    serve_log("evicted %s (%s) to make room", victim->file.c_str(),
              engine_name(victim->engine));
    return true;
}

void Residency::sweeper_loop() {
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (!running) break;
        long now = (long)time(nullptr);
        std::vector<std::shared_ptr<PhysModel>> snap;
        {
            std::lock_guard<std::mutex> lk(mu);
            for (auto &kv : phys)
                snap.push_back(kv.second);
        }
        for (auto &p : snap) {
            if (p->state.load() != 2 || p->pinned || p->ttl == 0) continue;
            if (p->in_use.load() > 0) continue;    // paused while queued/in-flight
            if (now - p->last_used.load() <= p->ttl) continue;
            // the actual unload happens on the main thread
            std::lock_guard<std::mutex> qk(queue_mu);
            push_task(LifeTask{p, false, nullptr});
            queue_cv.notify_one();
        }
    }
}

void Residency::handle_load(std::shared_ptr<PhysModel> p) {
    int evictions = 0;
    for (;;) {
        std::string lerr;
        bool loaded = false;
        {
            std::unique_lock<std::mutex> lk(p->mu);
            if (p->state.load() == 2)
                return;  // a duplicate task found it already loaded
            p->state = 1;
            load_engine(*p, lerr);
            if (lerr.empty()) {
                p->state = 2;
                p->last_used = (long)time(nullptr);
                loaded = true;
            }
            p->cv.notify_all();
        }
        if (loaded) {
            serve_log("loaded %s (%s)", p->file.c_str(),
                      engine_name(p->engine));
            return;
        }
        // Load failed: evict an LRU non-pinned model and retry, bounded.
        bool evicted = evictions < 8 && evict_one(*p);
        {
            std::lock_guard<std::mutex> lk(p->mu);
            p->state = evicted ? 1 : 0;  // keep "loading" while retrying
            if (!evicted)
                p->last_err = lerr;
            p->cv.notify_all();
        }
        if (!evicted)
            return;
        evictions++;
    }
}

void Residency::handle_unload(std::shared_ptr<PhysModel> p,
                       std::shared_ptr<std::promise<std::string>> result) {
    std::string res;
    {
        std::lock_guard<std::mutex> lk(p->mu);
        if (p->state.load() != 2)
            res = "model not loaded";
        else if (p->in_use.load() > 0)
            res = "model is in use";
        else {
            unload_engine(*p);
            p->state = 0;
        }
        p->cv.notify_all();
    }
    if (res.empty())
        serve_log("unloaded %s (%s)", p->file.c_str(),
                  engine_name(p->engine));
    if (result)
        result->set_value(res);
}

void Residency::handle_engine(std::shared_ptr<EngineJob> job) {
    job->run();
    {
        std::lock_guard<std::mutex> lk(job->mu);
        job->finished = true;
        job->cv.notify_all();
    }
    release(*job->p);  // the model can now be TTL-evicted
}

std::shared_ptr<PhysModel> Residency::acquire(const ModelEntry &e, std::string &err) {
    auto p = get_or_create(e);
    p->in_use.fetch_add(1);
    {
        std::unique_lock<std::mutex> lk(p->mu);
        if (p->state.load() == 0 && !pump_running.load()) {
            err = "server is shutting down";
            p->in_use.fetch_sub(1);
            return nullptr;
        }
        if (p->state.load() == 0) {
            int expected = 0;
            if (p->state.compare_exchange_strong(expected, 1)) {
                p->last_err.clear();
                std::lock_guard<std::mutex> qk(queue_mu);
                push_task(LifeTask{p, true, nullptr, nullptr, 1});
                queue_cv.notify_one();
            }
        }
        p->cv.wait(lk, [&] { return p->state.load() != 1; });
        if (p->state.load() == 2)
            return p;
        err = p->last_err;
    }
    return nullptr;
}

void Residency::release(PhysModel &p) {
    p.last_used = (long)time(nullptr);
    p.in_use.fetch_sub(1);
}

bool Residency::submit_engine(std::shared_ptr<EngineJob> job) {
    if (!pump_running.load())
        return false;
    std::lock_guard<std::mutex> qk(queue_mu);
    LifeTask t{job->p, false, nullptr, job};
    if (job->p->engine != EngineKind::Diffusion)
        t.priority = 1;  // interactive work jumps queued image jobs
    push_task(std::move(t));
    queue_cv.notify_one();
    return true;
}

nlohmann::json Residency::queue_json() const {
    std::lock_guard<std::mutex> qk(queue_mu);
    nlohmann::json q;
    q["pending"] = (int)(hi.size() + lo.size());
    q["active"] = task_running.load();
    return q;
}

bool Residency::unload(const ModelEntry &e, std::string &err) {
    auto p = find(e);
    if (!p) {
        err = "model not loaded";
        return false;
    }
    auto result = std::make_shared<std::promise<std::string>>();
    {
        std::lock_guard<std::mutex> qk(queue_mu);
        push_task(LifeTask{p, false, result});
        queue_cv.notify_one();
    }
    err = result->get_future().get();
    return err.empty();
}

nlohmann::json Residency::resident_json() const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &e : cfg.models) {
        if (e.engine == EngineKind::Llama)
            continue;
        auto p = find(e);
        nlohmann::json r;
        r["id"] = e.id;
        r["engine"] = engine_name(e.engine);
        if (p && p->state.load() == 2) {
            r["state"] = "loaded";
            long lu = p->last_used.load();
            r["last_used"] = lu ? nlohmann::json(lu) : nlohmann::json(nullptr);
            r["in_use"] = p->in_use.load();
            r["ttl_s"] = p->ttl;
            r["pinned"] = p->pinned;
        } else if (p && p->state.load() == 1) {
            r["state"] = "loading";
            r["last_used"] = nullptr;
            r["in_use"] = p->in_use.load();
            r["ttl_s"] = p->ttl;
            r["pinned"] = p->pinned;
        } else {
            r["state"] = "unloaded";
            r["last_used"] = nullptr;
            r["in_use"] = 0;
            r["ttl_s"] = p ? p->ttl : e.effective_ttl();
            r["pinned"] = p ? p->pinned : e.pinned;
        }
        arr.push_back(r);
    }
    return arr;
}

void Residency::start() {
    running = true;
    sweeper = std::thread([this] { sweeper_loop(); });
}

void Residency::pump() {
    pump_running = true;
    for (;;) {
        LifeTask t{nullptr, false, nullptr};
        {
            std::unique_lock<std::mutex> qk(queue_mu);
            // a timeout (not a pure wait) so the pump notices
            // stop_pump() even if the signal handler cannot notify
            queue_cv.wait_for(qk, std::chrono::milliseconds(500), [&] {
                return !hi.empty() || !lo.empty() || !pump_running.load();
            });
            if (hi.empty() && lo.empty()) {
                if (!pump_running.load())
                    break;
                continue;
            }
            auto &q = hi.empty() ? lo : hi;
            t = q.front();
            q.pop_front();
        }
        task_running = true;
        if (t.engine)
            handle_engine(t.engine);
        else if (t.p) {
            if (t.is_load)
                handle_load(t.p);
            else
                handle_unload(t.p, t.result);
        }
        task_running = false;
    }
}

void Residency::stop_pump() {
    pump_running = false;
}

void Residency::stop() {
    running = false;
    if (sweeper.joinable())
        sweeper.join();
    std::vector<std::shared_ptr<PhysModel>> snap;
    {
        std::lock_guard<std::mutex> lk(mu);
        for (auto &kv : phys)
            snap.push_back(kv.second);
    }
    for (auto &p : snap) {
        for (int i = 0; i < 100 && p->in_use.load() > 0; i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::lock_guard<std::mutex> lk(p->mu);
        if (p->state.load() == 2) {
            unload_engine(*p);
            p->state = 0;
        }
    }
}

} // namespace serve
