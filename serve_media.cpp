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
#include <cmath>
#include <dirent.h>
#include <iomanip>
#include <sstream>
#include <unordered_map>

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

// b64_encode's inverse, for the sdcpp surface's init_image payloads (a bare base64 body or a data: url, the two forms upstream accepts).
static bool b64_decode(const std::string &in, std::string &out) {
    std::string s = in;
    size_t comma = s.find(',');
    if (s.rfind("data:", 0) == 0 && comma != std::string::npos)
        s.erase(0, comma + 1);
    out.clear();
    out.reserve(s.size() * 3 / 4 + 4);
    int acc = 0, bits = 0;
    for (char c : s) {
        if (c == '=' || c == '\n' || c == '\r')
            continue;
        int v;
        if (c >= 'A' && c <= 'Z')       v = c - 'A';
        else if (c >= 'a' && c <= 'z')  v = c - 'a' + 26;
        else if (c >= '0' && c <= '9')  v = c - '0' + 52;
        else if (c == '+')              v = 62;
        else if (c == '/')              v = 63;
        else                            return false;
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += (char)((acc >> bits) & 0xFF);
        }
    }
    return !out.empty();
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

// The one-shot context dance behind every generation (see the note above): load the
// context if there is none, generate, copy the pixels out, and free the context again.
// The pump serializes engine work, so at most one request touches p.sctx at a time. Main thread only.
static void image_generate(PhysModel &p, sd_img_gen_params_t &sp,
                           std::vector<ImageResult::Img> &out,
                           std::string &err) {
    std::string lerr;
    {
        std::lock_guard<std::mutex> lk(p.mu);
        if (!p.sctx)
            load_diffusion(p, lerr);
    }
    if (!p.sctx) {
        err = lerr.empty() ? "model not loaded" : lerr;
        return;
    }
    if (!sd_ctx_supports_image_generation(p.sctx)) {
        err = "this model does not support image generation";
        std::lock_guard<std::mutex> lk(p.mu);
        free_sd_ctx(p.sctx);  // never ran a generation, but don't hold it
        p.sctx = nullptr;
        return;
    }
    sd_image_t *imgs = generate_image(p.sctx, &sp);
    if (!imgs) {
        err = "generation failed (out of memory?)";
        std::lock_guard<std::mutex> lk(p.mu);
        free_sd_ctx(p.sctx);  // a failed run may leave it half-set-up
        p.sctx = nullptr;
        return;
    }
    for (int i = 0; i < sp.batch_count; i++) {
        ImageResult::Img img;
        img.w = (int)imgs[i].width;
        img.h = (int)imgs[i].height;
        img.ch = (int)imgs[i].channel;
        img.px.resize((size_t)img.w * img.h * img.ch);
        if (imgs[i].data)
            std::memcpy(img.px.data(), imgs[i].data, img.px.size());
        free(imgs[i].data);
        out.push_back(std::move(img));
    }
    free(imgs);  // the engine's array itself
    {
        // One-shot context: drop it now that the pixels are copied out.
        std::lock_guard<std::mutex> lk(p.mu);
        free_sd_ctx(p.sctx);
        p.sctx = nullptr;
    }
}

// Run one generation on a loaded model (main thread only).
// edit is the decoded input image for image edits (nullptr = text-to-image).
ImageResult run_image(PhysModel &p, const std::string &prompt,
                      const std::string &negative, int w, int h,
                      int n, int steps, float cfg, long seed,
                      const std::vector<uint8_t> *edit,
                      int edit_w, int edit_h) {
    ImageResult out;
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
    std::vector<ImageResult::Img> imgs;
    std::string err;
    image_generate(p, sp, imgs, err);
    if (!err.empty()) {
        out.err = err;
        return out;
    }
    out.images = std::move(imgs);
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


// ── sdcpp surface (mirrors stable-diffusion.cpp's own server) ────────────
//
// The native /sdcpp/v1/* API of the sd.cpp server (examples/server), mirrored onto this
// child's one-shot generation path so clients that point at an sd.cpp server work unchanged.
// Request field names, the ignore-unknown-fields leniency and response shapes follow the
// pinned examples/server + examples/common; sampler/scheduler/upscaler names resolve through
// the engine's own str_to_* / *_name APIs. Only img_gen is exposed: the child hosts no video
// models, and execution goes through the same residency queue as every other engine job --
// the manager below only keeps the status/result JSON clients poll.

enum class SdcppStatus { Queued, Generating, Completed, Failed, Cancelled };

static const char *sdcpp_status_name(SdcppStatus s) {
    switch (s) {
        case SdcppStatus::Queued:     return "queued";
        case SdcppStatus::Generating: return "generating";
        case SdcppStatus::Completed:  return "completed";
        case SdcppStatus::Failed:     return "failed";
        case SdcppStatus::Cancelled:  return "cancelled";
    }
    return "?";
}

// One /sdcpp/v1/img_gen job: parsed and resolved at submission, then shared between the
// HTTP workers (which serialize it) and the pump (which runs it), so the engine params
// can point straight into its members for the duration of the run.
struct SdcppJob {
    std::string id;
    SdcppStatus status = SdcppStatus::Queued;
    long created_at = 0, started_at = 0, completed_at = 0;
    uint64_t seq = 0;  // submission order; queue_position counts ahead of it

    // Request, as parsed.
    std::string prompt, negative_prompt;
    int clip_skip = -1;
    int width = 0, height = 0;  // <= 0 = resolve from the init image, else 512
    int batch_count = 1;
    long seed = -1;             // < 0 = random at run time
    float strength = 0.75f;
    sd_sample_params_t sp{};    // engine defaults, request fields override
    std::vector<float> custom_sigmas;
    std::vector<uint8_t> init_px;  int init_w = 0, init_h = 0;
    bool hires_enabled = false;
    sd_hires_params_t hires{};
    std::string hires_model_path;  // resolved upscaler file ("" = builtin)
    std::string output_format = "png";

    // Result, set by the run on the pump thread.
    std::vector<std::string> result_b64;
    std::string error_code, error_message;
};

// Status/result bookkeeping for the jobs, mirroring upstream's AsyncJobManager: finished
// jobs linger 600 s, then stay 410-able for another 600 s. No execution lives here --
// that is the residency queue.
struct SdcppJobs {
    std::mutex mu;
    std::unordered_map<std::string, std::shared_ptr<SdcppJob>> jobs;
    std::unordered_map<std::string, long> expired;  // id -> when it forgets
    uint64_t next_id = 0;

    static const size_t MAX_PENDING = 64;  // upstream's default
    static const long FINISHED_TTL = 600;
    static const long EXPIRED_TTL = 600;
};

// Call with mgr.mu held.
static void sdcpp_purge(SdcppJobs &mgr, long now) {
    for (auto it = mgr.expired.begin(); it != mgr.expired.end();)
        it = (it->second <= now) ? mgr.expired.erase(it) : std::next(it);
    for (auto it = mgr.jobs.begin(); it != mgr.jobs.end();) {
        const auto &job = it->second;
        if (job->completed_at == 0) {
            ++it;
            continue;
        }
        if (now - job->completed_at >= SdcppJobs::FINISHED_TTL) {
            mgr.expired[job->id] = now + SdcppJobs::EXPIRED_TTL;
            it = mgr.jobs.erase(it);
        } else {
            ++it;
        }
    }
}

// Call with mgr.mu held.
static size_t sdcpp_pending(const SdcppJobs &mgr) {
    size_t n = 0;
    for (const auto &kv : mgr.jobs)
        if (kv.second->status == SdcppStatus::Queued ||
            kv.second->status == SdcppStatus::Generating)
            n++;
    return n;
}

// The job as clients see it; mirrors upstream's make_async_job_json.
// Call with mgr.mu held.
static nlohmann::json sdcpp_job_json(const SdcppJobs &mgr,
                                     const SdcppJob &job) {
    nlohmann::json r;
    r["id"] = job.id;
    r["kind"] = "img_gen";
    r["status"] = sdcpp_status_name(job.status);
    r["created"] = job.created_at;
    r["started"] = job.started_at == 0 ? nlohmann::json(nullptr)
                                      : nlohmann::json(job.started_at);
    r["completed"] = job.completed_at == 0 ? nlohmann::json(nullptr)
                                           : nlohmann::json(job.completed_at);
    r["queue_position"] = 0;
    if (job.status == SdcppStatus::Queued) {
        size_t position = 1;
        for (const auto &kv : mgr.jobs)
            if (kv.second->status == SdcppStatus::Queued &&
                kv.second->seq < job.seq)
                position++;
        r["queue_position"] = position;
    }
    if (job.status == SdcppStatus::Completed) {
        nlohmann::json images = nlohmann::json::array();
        for (size_t i = 0; i < job.result_b64.size(); i++)
            images.push_back({{"index", (int)i},
                              {"b64_json", job.result_b64[i]}});
        r["result"] = {{"output_format", job.output_format},
                       {"images", images}};
        r["error"] = nullptr;
    } else if (job.status == SdcppStatus::Failed ||
               job.status == SdcppStatus::Cancelled) {
        r["result"] = nullptr;
        r["error"] = {{"code", job.error_code},
                      {"message", job.error_message}};
    } else {
        r["result"] = nullptr;
        r["error"] = nullptr;
    }
    return r;
}

// The request's inline init image: a base64 payload or a data: url, decoded to 3-channel
// pixels (stb, the same decoder the OpenAI edits path uses).
static bool decode_sdcpp_image(const std::string &encoded,
                               std::vector<uint8_t> &px, int &w, int &h) {
    std::string bytes;
    if (!b64_decode(encoded, bytes))
        return false;
    int pw = 0, ph = 0, pch = 0;
    unsigned char *p = stbi_load_from_memory(
        (const unsigned char *)bytes.data(), (int)bytes.size(), &pw, &ph,
        &pch, 3);
    if (!p)
        return false;
    w = pw;
    h = ph;
    px.assign(p, p + (size_t)pw * ph * 3);
    free(p);
    return true;
}

// Parse and validate one /sdcpp/v1/img_gen body into a job. Field names and the
// ignore-unknown-fields leniency follow upstream's SDGenerationParams::from_json_str;
// sampler/scheduler/upscaler names resolve through the engine's own str_to_* APIs
// (an unknown name is ignored, as upstream does).
static bool parse_sdcpp_img_gen(const nlohmann::json &j, const Config &cfg,
                                SdcppJob &job, std::string &err) {
    auto str = [&j](const char *k) {
        return j.contains(k) && j[k].is_string() ? j[k].get<std::string>()
                                                 : std::string();
    };

    job.prompt = str("prompt");
    if (job.prompt.empty()) {
        err = "prompt is required";
        return false;
    }
    job.negative_prompt = str("negative_prompt");
    if (j.contains("clip_skip") && j["clip_skip"].is_number_integer())
        job.clip_skip = j["clip_skip"].get<int>();
    if (j.contains("width") && j["width"].is_number_integer())
        job.width = j["width"].get<int>();
    if (j.contains("height") && j["height"].is_number_integer())
        job.height = j["height"].get<int>();
    if (job.width > 4096 || job.height > 4096) {
        err = "width and height must be at most 4096";
        return false;
    }
    if (j.contains("batch_count") && j["batch_count"].is_number_integer())
        job.batch_count = j["batch_count"].get<int>();
    // upstream's strict-mode clamps
    job.batch_count = std::clamp(job.batch_count, 1, 8);
    if (j.contains("seed") && j["seed"].is_number_integer())
        job.seed = j["seed"].get<long>();
    if (j.contains("strength") && j["strength"].is_number())
        job.strength = (float)j["strength"].get<double>();

    std::string init = str("init_image");
    if (!init.empty() &&
        !decode_sdcpp_image(init, job.init_px, job.init_w, job.init_h)) {
        err = "invalid init_image";
        return false;
    }
    // Inpainting is out of scope for this mirror: the pinned engine's mask path corrupts memory
    // in this APE (segfault or a wedged child, reproduced without the serve layer via the plain
    // diffusion CLI), so the field is rejected rather than passed to the engine.
    if (j.contains("mask_image")) {
        err = "mask_image is not supported by this server";
        return false;
    }

    // sample_params: engine defaults, request fields override (upstream's parse_sample_params_json).
    {
        sd_img_gen_params_t scratch;
        sd_img_gen_params_init(&scratch);
        job.sp = scratch.sample_params;
        job.sp.custom_sigmas = nullptr;  // repointed at the job's vector below
        job.sp.custom_sigmas_count = 0;
        job.sp.extra_sample_args = nullptr;
        if (j.contains("sample_params") && j["sample_params"].is_object()) {
            const nlohmann::json &sp = j["sample_params"];
            if (sp.contains("sample_steps") &&
                sp["sample_steps"].is_number_integer())
                job.sp.sample_steps = sp["sample_steps"].get<int>();
            if (sp.contains("eta") && sp["eta"].is_number())
                job.sp.eta = (float)sp["eta"].get<double>();
            if (sp.contains("shifted_timestep") &&
                sp["shifted_timestep"].is_number_integer())
                job.sp.shifted_timestep = sp["shifted_timestep"].get<int>();
            if (sp.contains("flow_shift") && sp["flow_shift"].is_number())
                job.sp.flow_shift = (float)sp["flow_shift"].get<double>();
            if (sp.contains("custom_sigmas") &&
                sp["custom_sigmas"].is_array())
                job.custom_sigmas =
                    sp["custom_sigmas"].get<std::vector<float>>();
            if (sp.contains("sample_method") &&
                sp["sample_method"].is_string()) {
                sample_method_t m = str_to_sample_method(
                    sp["sample_method"].get<std::string>().c_str());
                if (m != SAMPLE_METHOD_COUNT)
                    job.sp.sample_method = m;
            }
            if (sp.contains("scheduler") && sp["scheduler"].is_string()) {
                scheduler_t s = str_to_scheduler(
                    sp["scheduler"].get<std::string>().c_str());
                if (s != SCHEDULER_COUNT)
                    job.sp.scheduler = s;
            }
            if (sp.contains("guidance") && sp["guidance"].is_object()) {
                const nlohmann::json &g = sp["guidance"];
                if (g.contains("txt_cfg") && g["txt_cfg"].is_number())
                    job.sp.guidance.txt_cfg = (float)g["txt_cfg"].get<double>();
                if (g.contains("img_cfg") && g["img_cfg"].is_number())
                    job.sp.guidance.img_cfg = (float)g["img_cfg"].get<double>();
                if (g.contains("distilled_guidance") &&
                    g["distilled_guidance"].is_number())
                    job.sp.guidance.distilled_guidance =
                        (float)g["distilled_guidance"].get<double>();
            }
        }
        if (job.sp.sample_steps > 0)
            job.sp.sample_steps = std::clamp(job.sp.sample_steps, 1, 100);
    }

    // hires (upstream's resolve(): an empty upscaler means "Latent", "None" disables it, and a
    // name that is not a builtin is an upscaler model file in upscalers_dir).
    if (j.contains("hires") && j["hires"].is_object()) {
        const nlohmann::json &h = j["hires"];
        job.hires_enabled = h.contains("enabled") && h["enabled"].is_boolean() &&
                            h["enabled"].get<bool>();
        std::string up;
        if (h.contains("upscaler") && h["upscaler"].is_string())
            up = h["upscaler"].get<std::string>();
        if (job.hires_enabled) {
            if (up.empty())
                up = "Latent";
            job.hires.upscaler = str_to_sd_hires_upscaler(up.c_str());
            if (job.hires.upscaler == SD_HIRES_UPSCALER_NONE)
                job.hires_enabled = false;
            else if (job.hires.upscaler == SD_HIRES_UPSCALER_COUNT) {
                // not a builtin: an upscaler model file in upscalers_dir
                static const char *exts[] = {".gguf", ".safetensors", ".pt",
                                             ".pth"};
                std::string cand = cfg.sdcpp.upscalers_dir + "/" + up;
                FILE *f = fopen(cand.c_str(), "r");
                if (!f)
                    for (const char *ext : exts) {
                        cand = cfg.sdcpp.upscalers_dir + "/" + up + ext;
                        f = fopen(cand.c_str(), "r");
                        if (f)
                            break;
                    }
                if (!f) {
                    err = "hires upscaler '" + up + "' is not a builtin name "
                          "and no upscaler model file was found for it";
                    return false;
                }
                fclose(f);
                job.hires_model_path = cand;
                job.hires.upscaler = SD_HIRES_UPSCALER_MODEL;
            }
            if (h.contains("scale") && h["scale"].is_number())
                job.hires.scale = (float)h["scale"].get<double>();
            if (h.contains("target_width") &&
                h["target_width"].is_number_integer())
                job.hires.target_width = h["target_width"].get<int>();
            if (h.contains("target_height") &&
                h["target_height"].is_number_integer())
                job.hires.target_height = h["target_height"].get<int>();
            if (h.contains("steps") && h["steps"].is_number_integer())
                job.hires.steps = h["steps"].get<int>();
            if (h.contains("denoising_strength") &&
                h["denoising_strength"].is_number())
                job.hires.denoising_strength =
                    (float)h["denoising_strength"].get<double>();
            if (h.contains("upscale_tile_size") &&
                h["upscale_tile_size"].is_number_integer())
                job.hires.upscale_tile_size =
                    h["upscale_tile_size"].get<int>();
        }
    }

    job.output_format = str("output_format");
    if (job.output_format.empty())
        job.output_format = "png";
    if (job.output_format != "png") {
        err = "output_format '" + job.output_format +
              "' is not supported (png only)";
        return false;
    }
    return true;
}

// Map a parsed job to the engine's params (upstream's to_sd_img_gen_params_t, for the fields
// img_gen exposes). The pointers point into the job, which the run closure keeps alive
// for the whole run.
static void build_sdcpp_params(const SdcppJob &job, sd_img_gen_params_t &sp) {
    sd_img_gen_params_init(&sp);
    sp.prompt = job.prompt.c_str();
    if (!job.negative_prompt.empty())
        sp.negative_prompt = job.negative_prompt.c_str();
    sp.clip_skip = job.clip_skip;
    int w = job.width > 0 ? job.width : (job.init_w > 0 ? job.init_w : 512);
    int h = job.height > 0 ? job.height : (job.init_h > 0 ? job.init_h : 512);
    sp.width = w;
    sp.height = h;
    sp.batch_count = job.batch_count;
    sp.seed = job.seed >= 0 ? job.seed : (int64_t)rand();
    sp.sample_params = job.sp;
    sp.sample_params.custom_sigmas = job.custom_sigmas.empty()
                                         ? nullptr
                                         : const_cast<float *>(
                                               job.custom_sigmas.data());
    sp.sample_params.custom_sigmas_count = (int)job.custom_sigmas.size();
    if (!job.init_px.empty()) {
        sp.init_image.width = (uint32_t)job.init_w;
        sp.init_image.height = (uint32_t)job.init_h;
        sp.init_image.channel = 3;
        sp.init_image.data = const_cast<uint8_t *>(job.init_px.data());
        sp.strength = job.strength;
    }
    if (job.hires_enabled) {
        sp.hires.enabled = true;
        sp.hires.upscaler = job.hires.upscaler;
        sp.hires.model_path = job.hires_model_path.empty()
                                  ? nullptr
                                  : job.hires_model_path.c_str();
        sp.hires.scale = job.hires.scale;
        sp.hires.target_width = job.hires.target_width > 0
                                    ? job.hires.target_width
                                    : (int)(w * job.hires.scale + 0.5);
        sp.hires.target_height = job.hires.target_height > 0
                                     ? job.hires.target_height
                                     : (int)(h * job.hires.scale + 0.5);
        sp.hires.steps = job.hires.steps > 0
                             ? job.hires.steps
                             : sp.sample_params.sample_steps;
        sp.hires.denoising_strength = job.hires.denoising_strength;
        sp.hires.upscale_tile_size = job.hires.upscale_tile_size;
    }
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
        p->in_use.fetch_sub(1);  // the load failed; callers release only on success
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
    SdcppJobs sdcpp_jobs;
    const ModelEntry *sdcpp_model = nullptr;  // resolved at start, stable

    explicit MediaServer(Config c)
        : cfg(std::move(c)), residency(cfg),
          sdcpp_model(cfg.sdcpp.enabled ? cfg.find(cfg.sdcpp.model)
                                        : nullptr) {}

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

        // sdcpp: the native API of stable-diffusion.cpp's own server, mirrored (see the section above); present in the config = on.
        if (cfg.sdcpp.enabled) {
            http.Post("/sdcpp/v1/img_gen",
                      [this](const httplib::Request &req,
                             httplib::Response &res) {
                if (req.body.empty()) {
                    res.status = 400;
                    res.set_content(R"({"error":"empty body"})",
                                    "application/json");
                    return;
                }
                nlohmann::json body;
                try {
                    body = nlohmann::json::parse(req.body);
                } catch (const nlohmann::json::exception &e) {
                    res.status = 400;
                    res.set_content(
                        nlohmann::json({{"error", "invalid json"},
                                        {"message", e.what()}})
                            .dump(-1, ' ', false,
                                  nlohmann::json::error_handler_t::replace),
                        "application/json");
                    return;
                }
                auto job = std::make_shared<SdcppJob>();
                std::string perr;
                if (!parse_sdcpp_img_gen(body, cfg, *job, perr)) {
                    res.status = 400;
                    res.set_content(
                        nlohmann::json({{"error", perr}})
                            .dump(-1, ' ', false,
                                  nlohmann::json::error_handler_t::replace),
                        "application/json");
                    return;
                }
                {
                    std::lock_guard<std::mutex> lk(sdcpp_jobs.mu);
                    sdcpp_purge(sdcpp_jobs, (long)time(nullptr));
                    if (sdcpp_pending(sdcpp_jobs) >= SdcppJobs::MAX_PENDING) {
                        res.status = 429;
                        res.set_content(R"({"error":"job queue is full"})",
                                        "application/json");
                        return;
                    }
                    uint64_t seq = sdcpp_jobs.next_id++;
                    std::ostringstream oss;
                    oss << "job_" << std::hex << (long)time(nullptr) << "_"
                        << std::setw(8) << std::setfill('0') << seq;
                    job->id = oss.str();
                    job->seq = seq;
                    job->created_at = (long)time(nullptr);
                    sdcpp_jobs.jobs[job->id] = job;
                }
                std::string aerr;
                auto p = residency.acquire(*sdcpp_model, aerr);
                if (!p) {
                    std::lock_guard<std::mutex> lk(sdcpp_jobs.mu);
                    sdcpp_jobs.jobs.erase(job->id);
                    res.status = 503;
                    res.set_content(
                        nlohmann::json(
                            {{"error", "model load failed: " + aerr}})
                            .dump(-1, ' ', false,
                                  nlohmann::json::error_handler_t::replace),
                        "application/json");
                    return;
                }
                auto ej = std::make_shared<EngineJob>();
                ej->p = p;
                ej->run = [this, p, job] {
                    {
                        std::lock_guard<std::mutex> lk(sdcpp_jobs.mu);
                        // cancelled while queued: never generate
                        if (job->status == SdcppStatus::Cancelled)
                            return;
                        job->status = SdcppStatus::Generating;
                        job->started_at = (long)time(nullptr);
                    }
                    std::string gerr;
                    {
                        sd_img_gen_params_t sp;
                        build_sdcpp_params(*job, sp);
                        std::vector<ImageResult::Img> imgs;
                        image_generate(*p, sp, imgs, gerr);
                        if (gerr.empty()) {
                            bool encoded_all = true;
                            for (const auto &img : imgs) {
                                std::string png;
                                if (!encode_png(img, png)) {
                                    encoded_all = false;
                                    break;
                                }
                                job->result_b64.push_back(b64_encode(png));
                            }
                            if (!encoded_all)
                                gerr = "failed to encode the generated image(s)";
                        }
                    }
                    std::lock_guard<std::mutex> lk(sdcpp_jobs.mu);
                    job->completed_at = (long)time(nullptr);
                    if (!gerr.empty()) {
                        job->status = SdcppStatus::Failed;
                        job->error_code = "generation_failed";
                        job->error_message = gerr;
                    } else {
                        job->status = SdcppStatus::Completed;
                    }
                };
                if (!residency.submit_engine(ej)) {
                    residency.release(*p);
                    std::lock_guard<std::mutex> lk(sdcpp_jobs.mu);
                    job->completed_at = (long)time(nullptr);
                    job->status = SdcppStatus::Failed;
                    job->error_code = "server_shutdown";
                    job->error_message = "server is shutting down";
                    res.status = 503;
                    res.set_content(
                        nlohmann::json(
                            {{"error", "server is shutting down"}})
                            .dump(-1, ' ', false,
                                  nlohmann::json::error_handler_t::replace),
                        "application/json");
                    return;
                }
                res.status = 202;
                nlohmann::json out;
                out["id"] = job->id;
                out["kind"] = "img_gen";
                out["status"] = "queued";
                out["created"] = job->created_at;
                out["poll_url"] = "/sdcpp/v1/jobs/" + job->id;
                res.set_content(
                    out.dump(-1, ' ', false,
                             nlohmann::json::error_handler_t::replace),
                    "application/json");
            });

            http.Get(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+))",
                     [this](const httplib::Request &req,
                            httplib::Response &res) {
                std::lock_guard<std::mutex> lk(sdcpp_jobs.mu);
                sdcpp_purge(sdcpp_jobs, (long)time(nullptr));
                const std::string id = req.matches[1].str();
                auto it = sdcpp_jobs.jobs.find(id);
                if (it == sdcpp_jobs.jobs.end()) {
                    if (sdcpp_jobs.expired.count(id)) {
                        res.status = 410;
                        res.set_content(R"({"error":"job expired"})",
                                        "application/json");
                    } else {
                        res.status = 404;
                        res.set_content(R"({"error":"job not found"})",
                                        "application/json");
                    }
                    return;
                }
                res.set_content(
                    sdcpp_job_json(sdcpp_jobs, *it->second)
                        .dump(-1, ' ', false,
                              nlohmann::json::error_handler_t::replace),
                    "application/json");
            });

            http.Post(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+)/cancel)",
                      [this](const httplib::Request &req,
                             httplib::Response &res) {
                std::lock_guard<std::mutex> lk(sdcpp_jobs.mu);
                sdcpp_purge(sdcpp_jobs, (long)time(nullptr));
                const std::string id = req.matches[1].str();
                auto it = sdcpp_jobs.jobs.find(id);
                if (it == sdcpp_jobs.jobs.end()) {
                    if (sdcpp_jobs.expired.count(id)) {
                        res.status = 410;
                        res.set_content(R"({"error":"job expired"})",
                                        "application/json");
                    } else {
                        res.status = 404;
                        res.set_content(R"({"error":"job not found"})",
                                        "application/json");
                    }
                    return;
                }
                SdcppJob &job = *it->second;
                if (job.status == SdcppStatus::Queued) {
                    // the queued run no-ops on the pump; this is the client-visible result
                    job.status = SdcppStatus::Cancelled;
                    job.completed_at = (long)time(nullptr);
                    job.error_code = "cancelled";
                    job.error_message = "job cancelled by client";
                } else if (job.status == SdcppStatus::Generating) {
                    res.status = 409;
                    res.set_content(
                        R"({"error":"job is currently generating and cannot be interrupted yet"})",
                        "application/json");
                    return;
                }
                res.status = 200;
                res.set_content(
                    sdcpp_job_json(sdcpp_jobs, job)
                        .dump(-1, ' ', false,
                              nlohmann::json::error_handler_t::replace),
                    "application/json");
            });

            http.Get("/sdcpp/v1/capabilities",
                     [this](const httplib::Request &,
                            httplib::Response &res) {
                const ModelEntry &m = *sdcpp_model;
                std::string name = m.file;
                {
                    const char *slash = strrchr(m.file.c_str(), '/');
                    if (slash)
                        name = slash + 1;
                }
                size_t dot = name.rfind('.');
                std::string stem =
                    dot != std::string::npos ? name.substr(0, dot) : name;

                // The engine's own defaults, reported like upstream's make_img_gen_defaults_json.
                sd_img_gen_params_t d;
                sd_img_gen_params_init(&d);
                auto finite_or_null = [](double v) -> nlohmann::json {
                    return std::isfinite(v) ? nlohmann::json(v)
                                            : nlohmann::json(nullptr);
                };
                const sd_sample_params_t &dsp = d.sample_params;
                nlohmann::json defaults;
                defaults["prompt"] = "";
                defaults["negative_prompt"] = "";
                defaults["clip_skip"] = -1;
                defaults["width"] = 512;
                defaults["height"] = 512;
                defaults["strength"] = 0.75;
                defaults["seed"] = 42;
                defaults["batch_count"] = 1;
                defaults["auto_resize_ref_image"] = true;
                defaults["increase_ref_index"] = false;
                defaults["control_strength"] = 0.9;
                defaults["sample_params"] = {
                    {"scheduler", sd_scheduler_name(dsp.scheduler)},
                    {"sample_method",
                     sd_sample_method_name(dsp.sample_method)},
                    {"sample_steps", dsp.sample_steps},
                    {"eta", finite_or_null(dsp.eta)},
                    {"shifted_timestep", dsp.shifted_timestep},
                    {"flow_shift", finite_or_null(dsp.flow_shift)},
                    {"guidance",
                     {
                         {"txt_cfg", dsp.guidance.txt_cfg},
                         {"img_cfg", finite_or_null(dsp.guidance.img_cfg)},
                         {"distilled_guidance",
                          dsp.guidance.distilled_guidance},
                         {"slg",
                          {
                              {"layers", nlohmann::json::array()},
                              {"layer_start",
                               dsp.guidance.slg.layer_start},
                              {"layer_end", dsp.guidance.slg.layer_end},
                              {"scale", dsp.guidance.slg.scale},
                          }},
                     }},
                };
                defaults["hires"] = {
                    {"enabled", false},
                    {"upscaler", "Latent"},
                    {"scale", 2.0},
                    {"target_width", 0},
                    {"target_height", 0},
                    {"steps", 0},
                    {"denoising_strength", 0.7},
                    {"upscale_tile_size", 128},
                };
                defaults["cache_mode"] = "";
                defaults["cache_option"] = "";
                defaults["scm_mask"] = "";
                defaults["scm_policy_dynamic"] = true;
                defaults["output_format"] = "png";
                defaults["output_compression"] = 100;

                nlohmann::json samplers = nlohmann::json::array();
                for (int i = 0; i < SAMPLE_METHOD_COUNT; i++)
                    samplers.push_back(
                        sd_sample_method_name((sample_method_t)i));
                nlohmann::json schedulers = nlohmann::json::array();
                for (int i = 0; i < SCHEDULER_COUNT; i++)
                    schedulers.push_back(sd_scheduler_name((scheduler_t)i));

                nlohmann::json upscalers = nlohmann::json::array();
                static const char *builtin[] = {
                    "None", "Lanczos", "Nearest", "Latent",
                    "Latent (nearest)", "Latent (nearest-exact)",
                    "Latent (antialiased)", "Latent (bicubic)",
                    "Latent (bicubic antialiased)",
                };
                for (const char *u : builtin)
                    upscalers.push_back({{"name", u}});
                std::vector<std::string> dir_up;
                if (!cfg.sdcpp.upscalers_dir.empty()) {
                    DIR *dir = opendir(cfg.sdcpp.upscalers_dir.c_str());
                    if (dir) {
                        while (struct dirent *de = readdir(dir)) {
                            const std::string n = de->d_name;
                            static const char *exts[] = {".gguf",
                                                         ".safetensors",
                                                         ".pt", ".pth"};
                            for (const char *ext : exts) {
                                size_t el = strlen(ext);
                                if (n.size() > el &&
                                    n.compare(n.size() - el, el, ext) == 0) {
                                    // upstream reports the file's stem
                                    dir_up.push_back(
                                        n.substr(0, n.size() - el));
                                    break;
                                }
                            }
                        }
                        closedir(dir);
                    }
                }
                std::sort(dir_up.begin(), dir_up.end());
                for (const auto &u : dir_up)
                    upscalers.push_back({{"name", u}});

                nlohmann::json out;
                out["model"] = {{"name", name}, {"stem", stem},
                                {"path", m.file}};
                out["current_mode"] = "img_gen";
                out["supported_modes"] = {"img_gen"};
                out["defaults"] = defaults;
                out["defaults_by_mode"] = {{"img_gen", defaults}};
                out["limits"] = {
                    {"min_width", 64},
                    {"max_width", 4096},
                    {"min_height", 64},
                    {"max_height", 4096},
                    {"max_batch_count", 8},
                    {"max_queue_size", (int)SdcppJobs::MAX_PENDING},
                };
                out["samplers"] = samplers;
                out["schedulers"] = schedulers;
                out["output_formats"] = {"png"};
                // features list what this mirror actually exposes; the full sd.cpp server
                // also serves control images, refs, loras, vae tiling and cache modes
                out["features"] = {
                    {"init_image", true},
                    {"mask_image", false},  // rejected: engine's mask path is broken here
                    {"control_image", false},
                    {"ref_images", false},
                    {"lora", false},
                    {"vae_tiling", false},
                    {"hires", true},
                    {"cache", false},
                    {"cancel_queued", true},
                    {"cancel_generating", false},
                };
                out["loras"] = nlohmann::json::array();
                out["upscalers"] = upscalers;
                res.set_content(
                    out.dump(-1, ' ', false,
                             nlohmann::json::error_handler_t::replace),
                    "application/json");
            });
        }

        // /v1/status for the serve layer to merge: the child's queue and
        // its resident rows (media models only -- llama rows come from the
        // router child).
        http.Get("/v1/status", [this](const httplib::Request &,
                                      httplib::Response &res) {
            nlohmann::json out;
            out["listen"] = cfg.host + ":" + std::to_string(cfg.port);
            out["key"] = false;  // the child gates no key; serve does
            out["uptime_s"] = (long)(time(nullptr) - start_at);
            // serve compares it against its own build and warns once on a
            // mismatch (a respawned child after an APE upgrade).
            out["version"] = CHIMERAFILE_VERSION_STRING;
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
    // Test-only knob: run the pump on a dedicated worker thread instead, the shape of
    // sd-server's async_worker (one non-main thread owns all engine work, including the
    // first backend load) -- probes how far the APE threading constraint actually reaches.
    if (const char *w = std::getenv("CHIMERAFILE_MEDIA_PUMP_WORKER");
        w && w[0] == '1') {
        std::thread pt([&server] { server.residency.pump(); });
        pt.join();
    } else {
        server.residency.pump();
    }
    server.shutdown();
    return 0;
}
