// serve_media.cpp -- the media engines' in-process home.  See
// serve_media.h for the role: PhysModel/Residency own the whisper,
// diffusion and transcribe handles and their load/unload/TTL lifecycle,
// and the main thread's pump() is the only place engine calls run.

#include "serve_media.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <csignal>
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

// Base64 for the data: image urls (the only user of this lives in the
// image routes below).
static std::string b64_encode(const std::string &in) {
    static const char *tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(in.size() / 3 * 4 + 4);
    for (size_t i = 0; i < in.size(); i += 3) {
        uint32_t v = (uint8_t)in[i] << 16;
        int left = (int)in.size() - (int)i - 1;
        if (left > 0) v |= (uint8_t)in[i + 1] << 8;
        if (left > 1) v |= (uint8_t)in[i + 2];
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += left > 0 ? tbl[(v >> 6) & 63] : '=';
        out += left > 1 ? tbl[v & 63] : '=';
    }
    return out;
}

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

// ── The media child ──────────────────────────────────────────────────────
//
// A thin HTTP front over the in-process residency: the serve layer spawns
// this (chimerafile media --config FILE --host H --port P) and relays the
// /v1/audio/* and /v1/images/* requests to it.  The child's main thread
// is the pump; its loopback port carries no auth (the serve process is
// the only client, and it gates the key on the public port).

struct MediaServer {
    Config cfg;
    Residency residency;
    httplib::Server http;
    time_t start_at = 0;
    std::thread http_thread;

    explicit MediaServer(Config c) : cfg(std::move(c)), residency(cfg) {}

    // OpenAI image generation + edits share one core: validate the model
    // (diffusion engine only), run the generation on the main thread
    // through an EngineJob, and serialize the PNGs (b64_json or data: url).
    // `edit` is the decoded input image for the img2img path (nullptr =
    // text-to-image).  Runs on the HTTP worker thread; only the engine
    // call is submitted to the pump.
    void handle_image_request(const std::string &model_id,
                              const std::string &prompt,
                              const std::string &negative,
                              const std::string &fmt,
                              const std::string &size, int n, int steps,
                              double cfg_scale, long seed,
                              const std::vector<uint8_t> *edit,
                              int edit_w, int edit_h,
                              httplib::Response &res) {
        auto send_err = [](httplib::Response &res, int status,
                           const char *type, const char *code,
                           const std::string &message) {
            nlohmann::json out;
            out["error"]["message"] = message;
            out["error"]["type"] = type;
            out["error"]["code"] = code;
            res.status = status;
            res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
        };
        const ModelEntry *e = cfg.find(model_id);
        if (!e) {
            send_err(res, 404, "invalid_request_error", "model_not_found",
                     "unknown model id");
            return;
        }
        if (e->engine != EngineKind::Diffusion) {
            send_err(res, 400, "invalid_request_error", "unsupported_model",
                     std::string("model '") + e->id +
                     "' runs on the " + engine_name(e->engine) +
                     " engine, not diffusion");
            return;
        }
        if (fmt != "b64_json" && fmt != "url") {
            send_err(res, 400, "invalid_request_error", "invalid_request",
                     "response_format must be b64_json or url");
            return;
        }
        int w = 512, h = 512;
        if (!size.empty()) {
            const size_t x = size.find('x');
            if (x == std::string::npos ||
                !(w = atoi(size.substr(0, x).c_str())) ||
                !(h = atoi(size.substr(x + 1).c_str()))) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "size must be WxH (for example 512x512)");
                return;
            }
            if (w <= 0 || h <= 0 || w > 2048 || h > 2048) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "size must be between 1x1 and 2048x2048");
                return;
            }
        } else if (edit) {
            // edits without an explicit size keep the input's
            w = edit_w;
            h = edit_h;
        }
        if (n < 1 || n > 4) {
            send_err(res, 400, "invalid_request_error", "invalid_request",
                     "'n' must be between 1 and 4");
            return;
        }

        std::string err;
        auto p = residency.acquire(*e, err);
        if (!p) {
            send_err(res, 503, "server_error", "model_load_failed", err);
            return;
        }

        auto job = std::make_shared<EngineJob>();
        job->p = p;
        ImageResult ir;
        job->run = [p, prompt, negative, w, h, n, steps,
                    cfg_f = cfg_scale > 0 ? (float)cfg_scale : -1.0f, seed,
                    edit, edit_w, edit_h, &ir] {
            ir = run_image(*p, prompt, negative, w, h, n, steps, cfg_f, seed,
                           edit, edit_w, edit_h);
        };
        if (!residency.submit_engine(job)) {
            residency.release(*p);
            send_err(res, 503, "server_error", "server_busy",
                     "server is shutting down");
            return;
        }
        {
            std::unique_lock<std::mutex> lk(job->mu);
            job->cv.wait(lk, [&] { return job->finished; });
        }
        if (!ir.ok) {
            send_err(res, 500, "server_error", "generation_failed", ir.err);
            return;
        }

        nlohmann::json data = nlohmann::json::array();
        for (const auto &img : ir.images) {
            std::string png;
            if (!encode_png(img, png)) {
                send_err(res, 500, "server_error", "generation_failed",
                         "failed to encode the generated image");
                return;
            }
            nlohmann::json d;
            if (fmt == "url")
                d["url"] = "data:image/png;base64," + b64_encode(png);
            else
                d["b64_json"] = b64_encode(png);
            d["revised_prompt"] = prompt;
            data.push_back(std::move(d));
        }
        nlohmann::json out;
        out["created"] = (long)time(nullptr);
        out["model"] = e->id;
        out["data"] = data;
        res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
    }

    void register_routes() {
        // SO_REUSEADDR, not httplib's default SO_REUSEPORT: the kernel
        // refuses a second bind on this port (see Server in serve.cpp).
        http.set_socket_options([](socket_t sock) {
            int yes = 1;
            setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
        });
        http.set_logger([](const httplib::Request &req,
                           const httplib::Response &res) {
            serve_log("resp %s %s %d\n", req.method.c_str(), req.path.c_str(),
                      res.status);
        });
        // No auth gate: loopback-only, and the only client is the serve
        // process, which checks the key on the public port.

        http.Get("/health", [](const httplib::Request &, httplib::Response &res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });

        // OpenAI: audio transcriptions (whisper + transcribe engines).
        // Multipart form: model, file, and optionally language, prompt,
        // response_format (json | verbose_json | text | srt | vtt), and
        // temperature (applied to whisper; transcribe has no such knob).
        http.Post("/v1/audio/transcriptions",
                  [this](const httplib::Request &req, httplib::Response &res,
                         const httplib::ContentReader &content_reader) {
            auto send_err = [](httplib::Response &res, int status,
                               const char *type, const char *code,
                               const std::string &message) {
                nlohmann::json out;
                out["error"]["message"] = message;
                out["error"]["type"] = type;
                out["error"]["code"] = code;
                res.status = status;
                res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
            };

            if (!req.is_multipart_form_data()) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "Content-Type must be multipart/form-data");
                return;
            }
            // Collect the form: the header callback fires at the start of
            // each part (a file part is recognised by its filename); the
            // receiver then gets that part's bytes.
            std::map<std::string, std::string> fields;
            std::string audio;
            bool got_file = false;
            std::string part_name;
            std::string part_filename;
            bool parsed = content_reader(
                [&](const httplib::FormData &fd) -> bool {
                    // The FormData reference is only valid for the
                    // duration of the header callback.
                    part_name     = fd.name;
                    part_filename = fd.filename;
                    return true;
                },
                [&](const char *d, size_t n) -> bool {
                    if (part_filename.empty())
                        fields[part_name] += std::string(d, n);
                    else {
                        audio += std::string(d, n);
                        got_file = true;
                    }
                    return true;
                });
            if (!parsed) {
                send_err(res, 400, "invalid_request_error", "invalid_request",
                         "could not parse the multipart form");
                return;
            }
            auto field = [&fields](const char *name) -> std::string {
                auto it = fields.find(name);
                return it == fields.end() ? std::string() : it->second;
            };
            const std::string model = field("model");
            if (model.empty()) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request", "'model' is required");
                return;
            }
            const ModelEntry *e = cfg.find(model);
            if (!e) {
                send_err(res, 404, "invalid_request_error", "model_not_found",
                         "unknown model id");
                return;
            }
            if (e->engine != EngineKind::Whisper &&
                e->engine != EngineKind::Transcribe) {
                send_err(res, 400, "invalid_request_error",
                         "unsupported_model",
                         std::string("model '") + e->id +
                         "' runs on the " + engine_name(e->engine) +
                         " engine, not a speech engine");
                return;
            }
            if (!got_file) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "'file' (the audio) is required");
                return;
            }
            std::string language = field("language");
            std::string prompt = field("prompt");
            std::string fmt = field("response_format");
            if (fmt.empty())
                fmt = "json";
            if (fmt != "json" && fmt != "verbose_json" && fmt != "text" &&
                fmt != "srt" && fmt != "vtt") {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "response_format must be json, verbose_json, "
                         "text, srt or vtt");
                return;
            }
            float temperature = -1.0f;  // < 0 = not set in the request
            const std::string t = field("temperature");
            if (!t.empty())
                temperature = (float)strtod(t.c_str(), nullptr);

            std::vector<float> pcm;
            if (!decode_audio(audio, pcm)) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "could not decode the audio file (WAV, MP3, FLAC "
                         "or OGG)");
                return;
            }

            std::string err;
            auto p = residency.acquire(*e, err);
            if (!p) {
                send_err(res, 503, "server_error", "model_load_failed", err);
                return;
            }

            auto job = std::make_shared<EngineJob>();
            job->p = p;
            TranscribeResult tr;
            job->run = [p, pcm = std::move(pcm), language, prompt,
                        temperature, &tr] {
                tr = run_transcribe(*p, pcm, language, prompt, temperature);
            };
            if (!residency.submit_engine(job)) {
                residency.release(*p);
                send_err(res, 503, "server_error", "server_busy",
                         "server is shutting down");
                return;
            }
            {
                std::unique_lock<std::mutex> lk(job->mu);
                job->cv.wait(lk, [&] { return job->finished; });
            }
            if (!tr.ok) {
                send_err(res, 500, "server_error", "transcription_failed",
                         tr.err);
                return;
            }

            if (fmt == "text") {
                res.set_content(tr.text, "text/plain");
                return;
            }
            if (fmt == "srt") {
                res.set_content(build_srt(tr.segments),
                                "application/x-subrip");
                return;
            }
            if (fmt == "vtt") {
                res.set_content(build_vtt(tr.segments), "text/vtt");
                return;
            }
            nlohmann::json out;
            out["text"] = tr.text;
            if (fmt == "verbose_json") {
                out["language"] =
                    tr.language.empty() ? std::string("en") : tr.language;
                out["duration"] = tr.duration_s;
                nlohmann::json segs = nlohmann::json::array();
                for (size_t i = 0; i < tr.segments.size(); i++) {
                    segs.push_back(
                        {{"id", (int)i},
                         {"start", tr.segments[i].t0},
                         {"end", tr.segments[i].t1},
                         {"text", tr.segments[i].text}});
                }
                out["segments"] = segs;
            }
            res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
        });

        // OpenAI: image generation (diffusion engine models).
        http.Post("/v1/images/generations",
                  [this](const httplib::Request &req, httplib::Response &res) {
            auto send_err = [](httplib::Response &res, int status,
                               const char *type, const char *code,
                               const std::string &message) {
                nlohmann::json out;
                out["error"]["message"] = message;
                out["error"]["type"] = type;
                out["error"]["code"] = code;
                res.status = status;
                res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
            };
            nlohmann::json body;
            try {
                body = nlohmann::json::parse(req.body);
            } catch (...) {
                send_err(res, 400, "invalid_request_error", "invalid_json",
                         "request body is not valid JSON");
                return;
            }
            if (!body.contains("model") || !body["model"].is_string()) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request", "'model' is required");
                return;
            }
            if (!body.contains("prompt") || !body["prompt"].is_string() ||
                body["prompt"].get<std::string>().empty()) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request", "'prompt' is required");
                return;
            }
            // negative_prompt, steps, cfg_scale and seed are chimerafile
            // extensions (OpenAI's API has none of them); the rest is
            // OpenAI-shaped.
            handle_image_request(
                body["model"].get<std::string>(),
                body["prompt"].get<std::string>(),
                body.value("negative_prompt", std::string()),
                body.value("response_format", std::string("b64_json")),
                body.value("size", std::string()),
                body.value("n", 1), body.value("steps", 0),
                body.value("cfg_scale", -1.0),
                body.contains("seed") && body["seed"].is_number()
                    ? body["seed"].get<long>()
                    : -1,
                nullptr, 0, 0, res);
        });

        // OpenAI: image edits (diffusion engine models, img2img path).
        http.Post("/v1/images/edits",
                  [this](const httplib::Request &req, httplib::Response &res,
                         const httplib::ContentReader &content_reader) {
            auto send_err = [](httplib::Response &res, int status,
                               const char *type, const char *code,
                               const std::string &message) {
                nlohmann::json out;
                out["error"]["message"] = message;
                out["error"]["type"] = type;
                out["error"]["code"] = code;
                res.status = status;
                res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
            };
            if (!req.is_multipart_form_data()) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "Content-Type must be multipart/form-data");
                return;
            }
            std::map<std::string, std::string> fields;
            std::string file;
            bool got_file = false;
            std::string part_name;
            std::string part_filename;
            bool parsed = content_reader(
                [&](const httplib::FormData &fd) -> bool {
                    // The FormData reference is only valid for the
                    // duration of the header callback.
                    part_name     = fd.name;
                    part_filename = fd.filename;
                    return true;
                },
                [&](const char *d, size_t n) -> bool {
                    if (part_filename.empty())
                        fields[part_name] += std::string(d, n);
                    else {
                        file += std::string(d, n);
                        got_file = true;
                    }
                    return true;
                });
            if (!parsed) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "could not parse the multipart form");
                return;
            }
            auto field = [&fields](const char *name) -> std::string {
                auto it = fields.find(name);
                return it == fields.end() ? std::string() : it->second;
            };
            if (field("model").empty()) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request", "'model' is required");
                return;
            }
            if (field("prompt").empty()) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request", "'prompt' is required");
                return;
            }
            if (!got_file) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "'image' (the input image) is required");
                return;
            }
            int iw = 0, ih = 0;
            std::vector<uint8_t> ipp;
            if (!decode_image(file, iw, ih, ipp)) {
                send_err(res, 400, "invalid_request_error",
                         "invalid_request",
                         "could not decode the input image (PNG or JPEG)");
                return;
            }
            handle_image_request(
                field("model"), field("prompt"), field("negative_prompt"),
                field("response_format").empty() ? "b64_json"
                                                 : field("response_format"),
                field("size"),
                (int)atol(field("n").empty() ? "1" : field("n").c_str()),
                (int)atol(field("steps").empty()
                              ? "0"
                              : field("steps").c_str()),
                field("cfg_scale").empty()
                    ? -1.0
                    : atof(field("cfg_scale").c_str()),
                field("seed").empty() ? -1 : atoll(field("seed").c_str()),
                &ipp, iw, ih, res);
        });

        // /v1/status for the serve layer to merge: the child's queue and
        // its resident rows (media models only -- llama rows come from the
        // router child).
        http.Get("/v1/status", [this](const httplib::Request &,
                                      httplib::Response &res) {
            nlohmann::json out;
            out["listen"] = cfg.host + ":" + std::to_string(cfg.port);
            out["key"] = false;  // the child gates no key; serve does
            out["uptime_s"] = (long)(time(nullptr) - start_at);
            out["queue"] = residency.queue_json();
            out["resident"] = residency.resident_json();
            res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
        });

        // Chimerafile extensions: preload / explicit-unload controls,
        // media models only (the serve layer dispatches llama models to
        // the router child).
        http.Post(R"(^/v1/residency/([A-Za-z0-9_-]+)$)",
                  [this](const httplib::Request &req, httplib::Response &res) {
            const ModelEntry *e = cfg.find(req.matches[1].str());
            if (!e) {
                res.status = 404;
                res.set_content(
                    "{\"error\":{\"message\":\"unknown model id\","
                    "\"type\":\"invalid_request_error\",\"code\":null}}",
                    "application/json");
                return;
            }
            if (e->engine == EngineKind::Llama) {
                res.status = 400;
                res.set_content(
                    "{\"error\":{\"message\":\"llama models are loaded by "
                    "the router child\","
                    "\"type\":\"invalid_request_error\",\"code\":null}}",
                    "application/json");
                return;
            }
            std::string err;
            auto p = residency.acquire(*e, err);
            if (!p) {
                res.status = 503;
                res.set_content(
                    nlohmann::json({{"error", {{"message", err},
                                               {"type", "server_error"},
                                               {"code", "model_load_failed"}}}})
                        .dump(-1, ' ', false, nlohmann::json::error_handler_t::replace),
                    "application/json");
                return;
            }
            residency.release(*p);  // keep it warm; the TTL clock starts now
            nlohmann::json out;
            out["id"] = e->id;
            out["state"] = "loaded";
            res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
        });

        http.Post(R"(^/v1/residency/([A-Za-z0-9_-]+)/unload$)",
                  [this](const httplib::Request &req, httplib::Response &res) {
            const ModelEntry *e = cfg.find(req.matches[1].str());
            nlohmann::json out;
            out["id"] = e ? e->id : req.matches[1].str();
            if (!e) {
                res.status = 404;
                out["state"] = "unknown";
                res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
                return;
            }
            if (e->engine == EngineKind::Llama) {
                res.status = 400;
                out["state"] = "unknown";
                out["error"] = "llama models are unloaded by the router child";
                res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
                return;
            }
            std::string err;
            bool ok = residency.unload(*e, err);
            if (!ok) {
                res.status = 409;
                out["state"] = "busy";
                out["error"] = err;
                res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
                return;
            }
            out["state"] = "unloaded";
            res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
        });

        // In this httplib fork the error handler runs for every response
        // with status >= 400, including the JSON errors our own handlers
        // send; only rewrite the body of truly unrouted requests, which
        // arrive with an empty body.
        http.set_error_handler([](const httplib::Request &,
                                  httplib::Response &res) {
            if ((res.status == 404 || res.status == 405) && res.body.empty()) {
                res.set_content(
                    "{\"error\":{\"message\":\"no such route\","
                    "\"type\":\"invalid_request_error\",\"code\":null}}",
                    "application/json");
            }
        });
    }

    // Binds the port and starts the accept loop on a background thread;
    // the calling (main) thread then owns the model lifecycle via
    // residency.pump().
    bool start(std::string &err) {
        register_routes();
        start_at = time(nullptr);
        residency.start();
        serve_log("media: listening on http://%s:%d\n", cfg.host.c_str(),
                  cfg.port);
        if (!http.bind_to_port(cfg.host.c_str(), cfg.port)) {
            err = "cannot bind " + cfg.host + ":" + std::to_string(cfg.port) +
                  " (is another media child running?)";
            residency.stop();
            return false;
        }
        http_thread = std::thread([this] { http.listen_after_bind(); });
        return true;
    }

    // Main thread, after the pump exits: stop the server and release every
    // resident model.
    void shutdown() {
        http.stop();
        if (http_thread.joinable())
            http_thread.join();
        residency.stop();
    }
};

} // namespace serve

// ── Entry point ──────────────────────────────────────────────────────────

static void print_media_usage(FILE *f) {
    fprintf(f, "usage: chimerafile media --config FILE [--host H] [--port P]\n\n");
    fprintf(f,
            "  FILE    declarative server config (JSON); the same file the\n"
            "          serve layer uses -- the child hosts the media engines\n"
            "          (whisper / diffusion / transcribe) in its own process\n"
            "  --host  override the config's listen host (the serve layer\n"
            "          always passes its own loopback port)\n"
            "  --port  override the config's listen port\n\n");
    fprintf(f,
            "The child listens on loopback with no key; the serve layer is\n"
            "its only client.  Run it directly only for debugging.\n");
}

static serve::MediaServer *g_media = nullptr;
static void on_media_signal(int) {
    if (g_media) {
        g_media->residency.stop_pump();
        g_media->http.stop();
    }
}

int media_main(int argc, char **argv) {
    const char *config = nullptr;
    const char *host = nullptr;
    int port = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --config requires a path\n");
                return 1;
            }
            config = argv[++i];
        } else if (strcmp(argv[i], "--host") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --host requires a value\n");
                return 1;
            }
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --port requires a value\n");
                return 1;
            }
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_media_usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    if (!config) {
        fprintf(stderr, "error: media requires --config FILE\n\n");
        print_media_usage(stderr);
        return 1;
    }

    serve::Config cfg;
    std::string err;
    if (!serve::parse_config(config, cfg, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    // The serve layer spawns the child with its own loopback port; the
    // flags override the config's listen entry.
    if (host)
        cfg.host = host;
    if (port > 0)
        cfg.port = port;

    serve::MediaServer server(cfg);
    g_media = &server;
    std::signal(SIGINT, on_media_signal);
    std::signal(SIGTERM, on_media_signal);

    if (!server.start(err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    // The main thread is the pump (engine calls only run there, in this
    // APE); pump() returns when a signal asks for shutdown.
    server.residency.pump();
    server.shutdown();
    return 0;
}
