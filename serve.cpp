// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// serve.cpp — OpenAI-compatible server mode for Chimerafile
//
// `chimerafile serve --config FILE` reads a declarative JSON config that
// describes the virtual models, which engine each runs on, and how long
// each stays resident.  The server publishes one OpenAI-style API surface
// (chat, transcriptions, images) plus chimerafile extensions (/v1/status),
// routes every request by model id, and keeps loaded models warm for a
// configurable idle TTL.
//
// Config format:
//
//   {
//     "listen": "127.0.0.1:8080",
//     "key": "optional-bearer-token",
//     "models": {
//       "assistant": {
//         "engine": "llama",
//         "file": "qwen2.5-0.5b-q4_0.gguf",
//         "ctx": 4096,
//         "gpu_layers": -1,
//         "ttl": 1800,
//         "pin": false,
//         "args": ["--temp", "0.7"]
//       }
//     }
//   }
//
//   listen     host:port to bind (default 127.0.0.1:8080)
//   key        optional bearer token; when set, required on all routes
//              except /health
//   engine     llama | whisper | diffusion | transcribe
//   file       model path; relative paths resolve against the config file's dir
//   ctx        llama only — context size (default 4096)
//   gpu_layers llama only — layers on GPU, -1 = all, 0 = CPU (default -1)
//   ttl        idle seconds before unload; omitted = engine default
//              (30m llama, 5m others); 0 = never
//   pin        true = never unload, never evict
//   args       engine flags passed through verbatim (llama.cpp options for
//              llama: --ctk, --ctv, --temp, ...)

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <sys/socket.h>

#include <nlohmann/json.hpp>
#include <cpp-httplib/httplib.h>
#include "serve_common.h"
#include "serve_forward.h"
#include "whisper.h"
#include "stable-diffusion.h"
#include "transcribe.h"

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

// The shared types (EngineKind, ModelEntry, Config) live in
// serve_common.h; the llama forwarder in serve_forward.h.

// ── Helpers ──────────────────────────────────────────────────────────────

// Standard base64 (RFC 4648), for OpenAI's b64_json image responses.
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

static bool parse_engine(const std::string &s, EngineKind &out) {
    std::string e = to_lower(s);
    if (e == "llama")      { out = EngineKind::Llama;      return true; }
    if (e == "whisper")    { out = EngineKind::Whisper;    return true; }
    if (e == "diffusion")  { out = EngineKind::Diffusion;  return true; }
    if (e == "transcribe") { out = EngineKind::Transcribe; return true; }
    return false;
}

static std::string fmt_ttl(const ModelEntry &m) {
    int secs = m.effective_ttl();
    if (secs == 0) return "never";
    char buf[32];
    if (secs < 60)                snprintf(buf, sizeof buf, "%ds", secs);
    else if (secs % 3600 == 0)    snprintf(buf, sizeof buf, "%dh", secs / 3600);
    else if (secs % 60 == 0)      snprintf(buf, sizeof buf, "%dm", secs / 60);
    else                          snprintf(buf, sizeof buf, "%ds", secs);
    return buf;
}

static bool parse_listen(const std::string &s, Config &cfg, std::string &err) {
    size_t colon = s.rfind(':');
    std::string host = "127.0.0.1", port_s = "8080";
    if (colon == std::string::npos) {
        port_s = s;
    } else {
        host = s.substr(0, colon);
        port_s = s.substr(colon + 1);
        if (host.empty()) host = "127.0.0.1";
    }
    if (port_s.empty()) {
        err = "listen: missing port in '" + s + "'";
        return false;
    }
    for (char c : port_s)
        if (!isdigit((unsigned char)c)) {
            err = "listen: bad port in '" + s + "'";
            return false;
        }
    int port = atoi(port_s.c_str());
    if (port <= 0 || port > 65535) {
        err = "listen: port out of range in '" + s + "'";
        return false;
    }
    cfg.host = host;
    cfg.port = port;
    return true;
}

// ── Config parsing ───────────────────────────────────────────────────────

inline bool parse_config(const char *path, Config &cfg, std::string &err) {
    std::ifstream f(path);
    if (!f) {
        err = std::string("cannot open config file '") + path + "'";
        return false;
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const nlohmann::json::exception &e) {
        err = std::string("config parse error: ") + e.what();
        return false;
    }
    if (!j.is_object()) {
        err = "config root must be a JSON object";
        return false;
    }

    // Directory of the config file, for relative model paths.
    std::string dir;
    {
        const char *slash = strrchr(path, '/');
        if (slash)
            dir = std::string(path, (size_t)(slash - path + 1));
    }

    if (j.contains("listen")) {
        if (!j["listen"].is_string()) {
            err = "listen must be a string (host:port)";
            return false;
        }
        if (!parse_listen(j["listen"].get<std::string>(), cfg, err))
            return false;
    }

    if (j.contains("key")) {
        if (!j["key"].is_string()) {
            err = "key must be a string";
            return false;
        }
        cfg.key = j["key"].get<std::string>();
    }

    if (j.contains("max_loaded_models")) {
        if (!j["max_loaded_models"].is_number_integer() ||
            j["max_loaded_models"].get<int>() < 1) {
            err = "max_loaded_models must be an integer >= 1";
            return false;
        }
        cfg.max_loaded = j["max_loaded_models"].get<int>();
    }

    if (!j.contains("models") || !j["models"].is_object() || j["models"].empty()) {
        err = "config must contain a non-empty 'models' object";
        return false;
    }

    for (auto &el : j["models"].items()) {
        const nlohmann::json &v = el.value();
        if (!v.is_object()) {
            err = "models." + el.key() + ": must be an object";
            return false;
        }

        ModelEntry m;
        m.id = el.key();

        if (!v.contains("engine") || !v["engine"].is_string()) {
            err = "models." + m.id + ": missing 'engine' (llama|whisper|diffusion|transcribe)";
            return false;
        }
        if (!parse_engine(v["engine"].get<std::string>(), m.engine)) {
            err = "models." + m.id + ": unknown engine '" + v["engine"].get<std::string>() + "'";
            return false;
        }

        if (v.contains("alias_of")) {
            if (!v["alias_of"].is_string() ||
                v["alias_of"].get<std::string>().empty()) {
                err = "models." + m.id + ": 'alias_of' must be a non-empty string";
                return false;
            }
            if (m.engine != EngineKind::Llama) {
                err = "models." + m.id + ": alias_of is only valid for llama models";
                return false;
            }
            m.alias_of = v["alias_of"].get<std::string>();
        }
        if (m.is_alias()) {
            // an alias shares its primary's file, engine and args; the
            // entry carries only the id and alias_of
            cfg.models.push_back(std::move(m));
            continue;
        }

        // Llama ids become sections of the generated --models-preset file
        // and travel in request bodies, so keep them to a safe alphabet.
        if (m.engine == EngineKind::Llama) {
            for (char c : m.id)
                if (!isalnum((unsigned char)c) && c != '-' && c != '_') {
                    err = "models." + m.id +
                          ": llama model ids may only contain letters, digits, '-' and '_'";
                    return false;
                }
        }

        if (!v.contains("file") || !v["file"].is_string() ||
            v["file"].get<std::string>().empty()) {
            err = "models." + m.id + ": missing 'file'";
            return false;
        }
        m.file = v["file"].get<std::string>();
        if (m.file[0] != '/')
            m.file = dir + m.file;
        FILE *mf = fopen(m.file.c_str(), "r");
        if (!mf) {
            err = "models." + m.id + ": cannot open model file '" + m.file + "'";
            return false;
        }
        fclose(mf);

        if (v.contains("ctx")) {
            if (!v["ctx"].is_number_integer() || v["ctx"].get<int>() < 8) {
                err = "models." + m.id + ": 'ctx' must be an integer >= 8";
                return false;
            }
            m.ctx = v["ctx"].get<int>();
            m.ctx_set = true;
        }
        if (v.contains("gpu_layers")) {
            if (!v["gpu_layers"].is_number_integer() ||
                v["gpu_layers"].get<int>() < -1) {
                err = "models." + m.id + ": 'gpu_layers' must be an integer >= -1";
                return false;
            }
            m.gpu_layers = v["gpu_layers"].get<int>();
            m.gpu_set = true;
        }
        if (v.contains("ttl")) {
            if (!v["ttl"].is_number_integer() || v["ttl"].get<int>() < 0) {
                err = "models." + m.id + ": 'ttl' must be an integer >= 0 (seconds)";
                return false;
            }
            m.ttl = v["ttl"].get<int>();
        }
        if (v.contains("pin")) {
            if (!v["pin"].is_boolean()) {
                err = "models." + m.id + ": 'pin' must be a boolean";
                return false;
            }
            m.pinned = v["pin"].get<bool>();
        }
        if (v.contains("args")) {
            if (!v["args"].is_array()) {
                err = "models." + m.id + ": 'args' must be an array of strings";
                return false;
            }
            for (const auto &a : v["args"]) {
                if (!a.is_string()) {
                    err = "models." + m.id + ": 'args' entries must be strings";
                    return false;
                }
                m.args.push_back(a.get<std::string>());
            }
        }

        cfg.models.push_back(std::move(m));
    }

    for (const auto &m : cfg.models) {
        if (!m.is_alias())
            continue;
        const ModelEntry *t = cfg.find(m.alias_of);
        if (!t) {
            err = "models." + m.id + ": alias_of points to unknown model '" +
                  m.alias_of + "'";
            return false;
        }
        if (t->engine != EngineKind::Llama || t->is_alias()) {
            err = "models." + m.id + ": alias_of must name a (non-alias) llama model";
            return false;
        }
    }

    // Pins must fit in the capacity, or the server would thrash from the
    // first request.
    {
        int pinned = 0;
        for (const auto &m : cfg.models)
            if (m.engine == EngineKind::Llama && !m.is_alias() && m.pinned)
                pinned++;
        if (pinned > cfg.max_loaded) {
            err = "max_loaded_models (" + std::to_string(cfg.max_loaded) +
                  ") is below the number of pinned llama models (" +
                  std::to_string(pinned) + ")";
            return false;
        }
    }

    return true;
}

// ── Residency ────────────────────────────────────────────────────────────
//
// A physical model is one loaded copy of (engine, file, args); several
// virtual models in the config that point at the same physical model share
// the one handle.  Loading uses each engine's own native loader -- llama.cpp
// through its official common-params/init machinery, whisper / stable-
// diffusion / transcribe through their C APIs -- so serve never re-implements
// engine internals.  Models load on first use and unload after an idle TTL
// (or on eviction when a bigger model doesn't fit).

void serve_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "serve: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}

static std::string phys_key(const ModelEntry &e) {
    return std::string(engine_name(e.engine)) + "\x01" + e.file;
}

struct PhysModel {
    std::string key;
    EngineKind engine = EngineKind::Llama;
    std::string file;
    std::vector<const ModelEntry *> entries;

    // Union of the policies of every entry pointing here: any pin wins,
    // the most generous TTL wins (0 = never).
    bool pinned = false;
    int ttl = 0;

    // the engines' handles (llama models never reach this struct; they
    // live in the upstream router child, see serve_forward.h)
    whisper_context * wctx = nullptr;
    sd_ctx_t * sctx = nullptr;
    struct transcribe_model * tmodel = nullptr;
    struct transcribe_session * tsess = nullptr;  // borrows tmodel

    std::atomic<int> state{0};    // 0 unloaded, 1 loading, 2 loaded
    std::atomic<long> last_used{0};
    std::atomic<int> in_use{0};   // reserved/running request count

    std::mutex mu;                // serializes load/unload of this handle
    std::condition_variable cv;
    std::string last_err;         // last load failure (set on the main thread)
};

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

struct TranscribeResult {
    bool ok = false;
    std::string err;
    std::string text;
    std::string language;
    double duration_s = 0.0;
    struct Segment {
        double t0 = 0.0;  // seconds
        double t1 = 0.0;
        std::string text;
    };
    std::vector<Segment> segments;
};

// Decode an uploaded audio file to 16 kHz mono float32 PCM (WAV, MP3,
// FLAC and Vorbis containers; resampled and mixed down as needed).
// Same pattern as whisper.cpp's read_audio_data: ask for the length, then
// read it all in one shot (reading past the end of a miniaudio decoder
// reports an error, so the exact-length read is the safe one).
static bool decode_audio(const std::string &data, std::vector<float> &pcm) {
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
static TranscribeResult run_transcribe(PhysModel &p,
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

static std::string build_srt(const std::vector<TranscribeResult::Segment> &segs) {
    std::string out;
    int n = 1;
    for (const auto &s : segs) {
        out += std::to_string(n++) + "\n";
        out += fmt_srt_time(s.t0) + " --> " + fmt_srt_time(s.t1) + "\n";
        out += s.text + "\n\n";
    }
    return out;
}

static std::string build_vtt(const std::vector<TranscribeResult::Segment> &segs) {
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

struct ImageResult {
    bool ok = false;
    std::string err;
    struct Img {
        int w = 0, h = 0, ch = 0;
        std::vector<uint8_t> px;  // row-major, ch bytes per pixel
    };
    std::vector<Img> images;
};

// Run one generation on a loaded model (main thread only).
// edit is the decoded input image for image edits (nullptr = text-to-image).
static ImageResult run_image(PhysModel &p, const std::string &prompt,
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

static bool encode_png(const ImageResult::Img &img, std::string &out) {
    out.clear();
    return stbi_write_png_to_func(png_mem_write, &out, img.w, img.h, img.ch,
                                  img.px.data(), 0) != 0;
}

// Decode an uploaded image (PNG or JPEG) to 3-channel RGB with stb_image.
static bool decode_image(const std::string &data, int &w, int &h,
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
struct EngineJob {
    std::shared_ptr<PhysModel> p;
    std::function<void()> run;  // executed on the main thread

    std::mutex mu;
    std::condition_variable cv;
    bool finished = false;  // run() returned
};

// Per-streaming-chat state shared between the pump (producing pieces)
// and the content provider (draining them to the client).

class Residency {
    Config &cfg;
    mutable std::mutex mu;
    std::map<std::string, std::shared_ptr<PhysModel>> phys;

    // In this APE, every engine call that touches the host-compiled
    // libraries (model load, unload, and inference alike) must run on the
    // process' main thread -- the only thread with a working host-glibc
    // malloc state (the host libraries' TLS is set up at dlopen time, for
    // the dlopening thread only).  So everything engine-side is posted to
    // this queue and executed by the main thread's pump(); the HTTP workers
    // only do HTTP and JSON.
    struct LifeTask {
        std::shared_ptr<PhysModel> p;
        bool is_load;
        std::shared_ptr<std::promise<std::string>> result;  // null = fire and forget
        std::shared_ptr<EngineJob> engine;                  // non-null: engine work
        int priority = 0;  // 0 = normal lane, 1 = fast lane
    };
    mutable std::mutex queue_mu;
    std::condition_variable queue_cv;
    // Two lanes, both FIFO: the normal one, and a fast lane that jumps
    // it.  Loads take the fast lane (their callers block on them and the
    // eviction-retry loop needs them promptly), and so do the
    // interactive engine jobs (chat, transcription); long diffusion
    // generations stay in the normal lane, so a fresh chat or
    // transcription request jumps ahead of queued image work.  Whatever
    // is running always runs to completion -- no preemption in v1.
    std::deque<LifeTask> lo, hi;

    // Call with queue_mu held.
    void push_task(LifeTask t) {
        (t.priority ? hi : lo).push_back(std::move(t));
    }
    std::atomic<bool> pump_running{false};
    std::atomic<bool> task_running{false};   // the pump is handling a task

    std::thread sweeper;
    std::atomic<bool> running{false};

    std::shared_ptr<PhysModel> get_or_create(const ModelEntry &e) {
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

    std::shared_ptr<PhysModel> find(const ModelEntry &e) const {
        std::lock_guard<std::mutex> lk(mu);
        auto it = phys.find(phys_key(e));
        return it == phys.end() ? nullptr : it->second;
    }

    // Evict the least-recently-used model that is not pinned, not in use,
    // and not the one that just failed to load.  Main thread only.
    bool evict_one(const PhysModel &self) {
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

    void sweeper_loop() {
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

    // ── main-thread handlers (run by pump()) ─────────────────────────────

    void handle_load(std::shared_ptr<PhysModel> p) {
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

    void handle_unload(std::shared_ptr<PhysModel> p,
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

    // The engine work itself (chat generation, transcription, ...) lives
    // in the job's run closure, which each route builds; the pump only
    // executes it on the main thread and reports back.
    void handle_engine(std::shared_ptr<EngineJob> job) {
        job->run();
        {
            std::lock_guard<std::mutex> lk(job->mu);
            job->finished = true;
            job->cv.notify_all();
        }
        release(*job->p);  // the model can now be TTL-evicted
    }

public:
    explicit Residency(Config &c) : cfg(c) {}
    ~Residency() { stop(); }

    // Reserve a model for a request (worker thread).  If it is not loaded,
    // dispatches a load to the main thread and waits; the main thread
    // handles the eviction-retry loop.  Callers must release().
    std::shared_ptr<PhysModel> acquire(const ModelEntry &e, std::string &err) {
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

    void release(PhysModel &p) {
        p.last_used = (long)time(nullptr);
        p.in_use.fetch_sub(1);
    }

    // Submit engine work to the main thread (worker thread).  The model
    // must already be acquired; the pump releases it when the run
    // finishes.  Returns false if the pump is shutting down.
    bool submit_engine(std::shared_ptr<EngineJob> job) {
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

    nlohmann::json queue_json() const {
        std::lock_guard<std::mutex> qk(queue_mu);
        nlohmann::json q;
        q["pending"] = (int)(hi.size() + lo.size());
        q["active"] = task_running.load();
        return q;
    }

    // Explicit unload (preload / free-the-GPU control): dispatched to the
    // main thread and waited on.
    bool unload(const ModelEntry &e, std::string &err) {
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

    // Media engines only: llama models never pass through the residency
    // (their rows come from the forwarder's /v1/status merge).
    nlohmann::json resident_json() const {
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

    // Start the TTL sweeper.  The main thread then owns the lifecycle via
    // pump().
    void start() {
        running = true;
        sweeper = std::thread([this] { sweeper_loop(); });
    }

    // Main thread only: process lifecycle tasks until stop_pump().
    void pump() {
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

    // Ask pump() to exit.  Async-signal-safe (atomic store only); the pump
    // wakes within its 500 ms timeout.
    void stop_pump() {
        pump_running = false;
    }

    // Main thread only, after pump() returned: stop the sweeper and release
    // every resident model (waiting briefly for in-flight requests).
    void stop() {
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
};

// ── HTTP server ──────────────────────────────────────────────────────────

// Process-wide counters for the /metrics endpoint.
struct Metrics {
    std::atomic<long> chat_requests{0};
    std::atomic<long> transcribe_requests{0};
    std::atomic<long> image_requests{0};
    std::atomic<long> prompt_tokens{0};
    std::atomic<long> generation_tokens{0};
    std::atomic<long long> prompt_seconds_ms{0};
    std::atomic<long long> generation_seconds_ms{0};

    // Fed from the forwarder out of the child's own usage/timings fields
    // (upstream shape: prompt_ms / predicted_ms).
    void record_forward(const nlohmann::json &usage,
                        const nlohmann::json &timings) {
        chat_requests++;
        if (usage.is_object()) {
            prompt_tokens += usage.value("prompt_tokens", 0);
            generation_tokens += usage.value("completion_tokens", 0);
        }
        if (timings.is_object()) {
            prompt_seconds_ms += (long long)timings.value("prompt_ms", 0.0);
            generation_seconds_ms +=
                (long long)timings.value("predicted_ms", 0.0);
        }
    }
};

struct Server {
    Config cfg;
    Residency residency;
    Metrics metrics;
    Forwarder forwarder;
    httplib::Server http;
    time_t start_at = 0;
    std::thread http_thread;
    std::string argv0;   // the APE file; the router child re-runs it

    Server(Config c, std::string a0)
        : cfg(std::move(c)), residency(cfg), forwarder(cfg),
          argv0(std::move(a0)) {}

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
        metrics.image_requests++;
        nlohmann::json out;
        out["created"] = (long)time(nullptr);
        out["model"] = e->id;
        out["data"] = data;
        res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
    }

    void register_routes() {
        // httplib's default socket options set SO_REUSEPORT, which would let
        // two chimerafile serve instances coexist on one port and silently
        // split connections between them.  Use SO_REUSEADDR instead: the
        // kernel then refuses the second bind (our single-instance guard),
        // while still allowing an immediate re-bind after a restart.
        http.set_socket_options([](socket_t sock) {
            int yes = 1;
            setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
        });

        // Auth gate: /health stays open (liveness for shell wrappers and the
        // GUI later); everything else needs the bearer key when one is set.
        // A "req" line with no matching "resp" is the request that took the
        // process down, so log at the start of routing.
        http.set_logger([](const httplib::Request &req,
                           const httplib::Response &res) {
            serve_log("resp %s %s %d\n", req.method.c_str(), req.path.c_str(),
                      res.status);
        });
        http.set_pre_routing_handler([this](const httplib::Request &req,
                                            httplib::Response &res)
                                            -> httplib::Server::HandlerResponse {
            serve_log("req %s %s\n", req.method.c_str(), req.path.c_str());
            if (req.path == "/health")
                return httplib::Server::HandlerResponse::Unhandled;
            if (!cfg.key.empty()) {
                const std::string auth = req.get_header_value("Authorization");
                if (auth != "Bearer " + cfg.key) {
                    res.status = 401;
                    res.set_content(
                        "{\"error\":{\"message\":\"invalid or missing API key "
                        "(send Authorization: Bearer <key>)\","
                        "\"type\":\"authentication_error\",\"code\":null}}",
                        "application/json");
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });

        http.Get("/health", [](const httplib::Request &, httplib::Response &res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });

        // OpenAI-shaped model list, extended with the engine each id routes to.
        http.Get("/v1/models", [this](const httplib::Request &,
                                      httplib::Response &res) {
            nlohmann::json data = nlohmann::json::array();
            for (const auto &m : cfg.models) {
                data.push_back({
                    {"id", m.id},
                    {"object", "model"},
                    {"created", 0},
                    {"owned_by", "chimerafile"},
                    {"engine", engine_name(m.engine)},
                });
            }
            nlohmann::json out = {{"object", "list"}, {"data", data}};
            res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
        });

        // OpenAI: chat completions (llama engine models).  The request is
        // forwarded to the upstream router child as-is (the child owns the
        // template, grammar, streaming and error semantics); the serve side
        // validates the model, does capacity admission, and rewrites the
        // reply's model id back to what the client asked for.
        http.Post("/v1/chat/completions",
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

            nlohmann::ordered_json body;
            try {
                body = nlohmann::ordered_json::parse(req.body);
            } catch (...) {
                send_err(res, 400, "invalid_request_error", "invalid_json",
                         "request body is not valid JSON");
                return;
            }
            if (!body.contains("model") || !body["model"].is_string()) {
                send_err(res, 400, "invalid_request_error", "invalid_request",
                         "'model' is required");
                return;
            }
            const ModelEntry *e = cfg.find(body["model"].get<std::string>());
            if (!e) {
                send_err(res, 404, "invalid_request_error", "model_not_found",
                         "unknown model id");
                return;
            }
            if (e->engine != EngineKind::Llama) {
                send_err(res, 400, "invalid_request_error",
                         "unsupported_model",
                         std::string("model '") + e->id +
                         "' runs on the " + engine_name(e->engine) +
                         " engine, not llama");
                return;
            }
            forwarder.handle_chat(res, *e, body);
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
            metrics.transcribe_requests++;

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

        // Chimerafile extension: what is loaded, what is queued, how long
        // the server has been up.  Resident rows merge the media engines'
        // in-process state with the llama models' state from the router
        // child (which also carries the model child's port).
        http.Get("/v1/status", [this](const httplib::Request &,
                                      httplib::Response &res) {
            nlohmann::json out;
            out["listen"] = cfg.host + ":" + std::to_string(cfg.port);
            out["key"] = !cfg.key.empty();
            out["uptime_s"] = (long)(time(nullptr) - start_at);
            out["queue"] = residency.queue_json();
            out["resident"] = resident_json();
            res.set_content(out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace), "application/json");
        });

        // Prometheus-format metrics (like llama-server's /metrics): process
        // counters, per-model residency, and the derived average throughput.
        http.Get("/metrics", [this](const httplib::Request &,
                                    httplib::Response &res) {
            std::string m;
            auto gauge = [&m](const std::string &name, const char *help,
                              double v) {
                m += "# HELP " + name + " " + help +
                     "\n# TYPE " + name + " gauge\n" + name + " " +
                     std::to_string(v) + "\n";
            };
            auto counter = [&m](const std::string &name, const char *help,
                                double v) {
                m += "# HELP " + name + " " + help +
                     "\n# TYPE " + name + " counter\n" + name + " " +
                     std::to_string(v) + "\n";
            };
            double gen_s = metrics.generation_seconds_ms.load() / 1000.0;
            gauge("chimerafile_uptime_seconds",
                  "Seconds since the server started.",
                  (double)(time(nullptr) - start_at));
            counter("chimerafile_chat_requests_total",
                    "Chat completions served.", metrics.chat_requests);
            counter("chimerafile_transcribe_requests_total",
                    "Transcriptions served.", metrics.transcribe_requests);
            counter("chimerafile_image_requests_total",
                    "Image generations served.", metrics.image_requests);
            counter("chimerafile_prompt_tokens_total",
                    "Prompt tokens processed.", metrics.prompt_tokens);
            counter("chimerafile_generation_tokens_total",
                    "Generated tokens produced.", metrics.generation_tokens);
            counter("chimerafile_prompt_processing_seconds_total",
                    "Seconds spent decoding prompts.",
                    metrics.prompt_seconds_ms / 1000.0);
            counter("chimerafile_generation_seconds_total",
                    "Seconds spent generating tokens.",
                    metrics.generation_seconds_ms / 1000.0);
            gauge("chimerafile_tokens_per_second",
                  "Average generation throughput (process lifetime).",
                  gen_s > 0.0 ? metrics.generation_tokens.load() / gen_s
                              : 0.0);
            for (const auto &r : resident_json()) {
                const std::string id = r["id"].get<std::string>();
                gauge("chimerafile_model_state{model=\"" + id + "\"}",
                      "Model residency state: 0 unloaded, 1 loading, 2 loaded.",
                      r["state"] == "loaded" ? 2.0
                      : r["state"] == "loading" ? 1.0 : 0.0);
                gauge("chimerafile_model_in_use{model=\"" + id + "\"}",
                      "In-flight requests on the model.",
                      (double)r["in_use"].get<int>());
            }
            res.set_content(m, "text/plain; version=0.0.4; charset=utf-8");
        });

        // Chimerafile extensions: preload / explicit-unload controls.
        // Llama models go through the forwarder (the router child loads and
        // unloads them); media models use the in-process residency.
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
            std::string err;
            if (e->engine == EngineKind::Llama) {
                if (!forwarder.preload(*e, err)) {
                    res.status = 503;
                    res.set_content(
                        nlohmann::json({{"error", {{"message", err},
                                                   {"type", "server_error"},
                                                   {"code", "model_load_failed"}}}})
                            .dump(-1, ' ', false, nlohmann::json::error_handler_t::replace),
                        "application/json");
                    return;
                }
            } else {
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
            }
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
            std::string err;
            bool ok = e->engine == EngineKind::Llama
                ? forwarder.unload(*e, err)
                : residency.unload(*e, err);
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
                    "{\"error\":{\"message\":\"no such route "
                    "(engine routing lands in the next steps)\","
                    "\"type\":\"invalid_request_error\",\"code\":null}}",
                    "application/json");
            }
        });
    }

    // The /v1/status and /metrics rows: media engines from the in-process
    // residency, llama models from the router child.
    nlohmann::json resident_json() {
        nlohmann::json out = residency.resident_json();
        auto llama = forwarder.resident_json();
        for (auto &r : llama)
            out.push_back(std::move(r));
        return out;
    }

    // Binds the port (our single-instance guard) and starts the accept loop
    // on a background thread; the calling (main) thread then owns the model
    // lifecycle via residency.pump().  Returns false if the port could not
    // be bound or the router child would not come up.
    bool start(std::string &err) {
        register_routes();
        std::string ierr;
        if (!forwarder.init(argv0, ierr)) {
            err = ierr;
            return false;
        }
        forwarder.record_chat = [this](const nlohmann::json &u,
                                       const nlohmann::json &t) {
            metrics.record_forward(u, t);
        };
        start_at = time(nullptr);
        residency.start();
        fprintf(stderr, "serve: listening on http://%s:%d\n",
                cfg.host.c_str(), cfg.port);
        if (!http.bind_to_port(cfg.host.c_str(), cfg.port)) {
            err = "cannot bind " + cfg.host + ":" + std::to_string(cfg.port) +
                  " (is another chimerafile serve running?)";
            residency.stop();
            return false;
        }
        // the router child is up before any request can arrive
        if (!forwarder.start(err)) {
            residency.stop();
            http.stop();
            return false;
        }
        http_thread = std::thread([this] { http.listen_after_bind(); });
        return true;
    }

    // Main thread, after the pump exits: stop the server and release every
    // resident model, then let the router child (and with it the model
    // children) go.
    void shutdown() {
        http.stop();
        if (http_thread.joinable())
            http_thread.join();
        residency.stop();
        forwarder.stop();
    }
};

} // namespace serve

// ── Entry point ──────────────────────────────────────────────────────────

static void print_serve_usage(FILE *f, const char *prog) {
    fprintf(f, "usage: %s serve --config FILE\n\n", prog);
    fprintf(f, "  FILE    declarative server config (JSON)\n");
    fprintf(f, "          see README.md for the format\n");
}

static void print_config_summary(const serve::Config &cfg) {
    fprintf(stderr, "serve: listen %s:%d\n", cfg.host.c_str(), cfg.port);
    fprintf(stderr, "serve: key %s\n", cfg.key.empty() ? "(none)" : "(set)");
    fprintf(stderr, "serve: max_loaded_models %d\n", cfg.max_loaded);
    fprintf(stderr, "serve: models\n");
    for (const auto &m : cfg.models) {
        if (m.is_alias()) {
            fprintf(stderr, "  %-14s %-10s alias of %s\n",
                    m.id.c_str(), serve::engine_name(m.engine),
                    m.alias_of.c_str());
            continue;
        }
        fprintf(stderr, "  %-14s %-10s %s",
                m.id.c_str(), serve::engine_name(m.engine), m.file.c_str());
        if (m.engine == serve::EngineKind::Llama) {
            fprintf(stderr, "  ctx=%d", m.ctx);
            if (m.gpu_layers != -1)
                fprintf(stderr, " gpu_layers=%d", m.gpu_layers);
        }
        fprintf(stderr, "  ttl=%s", serve::fmt_ttl(m).c_str());
        if (m.pinned)
            fprintf(stderr, " pinned");
        if (!m.args.empty())
            fprintf(stderr, " args(%zu)", m.args.size());
        fprintf(stderr, "\n");
    }
}

static serve::Server *g_server = nullptr;
static void on_signal(int) {
    if (g_server) {
        g_server->residency.stop_pump();
        g_server->http.stop();
    }
}

int serve_main(int argc, char **argv) {
    const char *config = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --config requires a path\n");
                return 1;
            }
            config = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_serve_usage(stdout, argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    if (!config) {
        fprintf(stderr, "error: serve requires --config FILE\n\n");
        print_serve_usage(stderr, argv[0]);
        return 1;
    }

    serve::Config cfg;
    std::string err;
    if (!serve::parse_config(config, cfg, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    print_config_summary(cfg);

    // argv[0] is the APE file; the router child re-runs it through the
    // APE runtime (/proc/self/exe).
    serve::Server server(cfg, argv[0]);
    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    if (!server.start(err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    // The main thread owns every engine call (load, unload, inference -- in
    // this APE only the main thread has a working host-glibc malloc state,
    // which the host-compiled GPU backend depends on); the HTTP workers only
    // do HTTP and JSON.  pump() returns when a signal asks for shutdown.
    server.residency.pump();
    server.shutdown();
    fprintf(stderr, "serve: stopped\n");
    return 0;
}
