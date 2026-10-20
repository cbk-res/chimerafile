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

#include <atomic>
#include <cstdarg>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
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
#include "serve_media.h"

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

void serve_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "serve: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}

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
