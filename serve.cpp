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

namespace serve {

// The shared types (EngineKind, ModelEntry, Config) live in
// serve_common.h; the llama forwarder in serve_forward.h.

// ── Helpers ──────────────────────────────────────────────────────────────

// Standard base64 (RFC 4648), for OpenAI's b64_json image responses.
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

// Rebuild a multipart body from the parts the httplib fork parsed into
// req.form.  This fork parses multipart uploads (leaving req.body empty),
// so relaying the upload to the media child verbatim is not possible; the
// child re-parses the rebuilt form.  Part-level headers are dropped (the
// child decodes uploads by content, not by announced type).
static const char *kMultipartBoundary = "chimerafile-serve-boundary";
static const char *kMultipartContentType =
    "multipart/form-data; boundary=chimerafile-serve-boundary";

static std::string build_multipart_body(const httplib::FormFields &fields,
                                        const httplib::FormFiles &files) {
    std::string out;
    const char *boundary = kMultipartBoundary;
    for (const auto &kv : fields) {
        out += "--" + std::string(boundary) + "\r\n" +
               "Content-Disposition: form-data; name=\"" + kv.second.name +
               "\"\r\n\r\n" + kv.second.content + "\r\n";
    }
    for (const auto &kv : files) {
        out += "--" + std::string(boundary) + "\r\n" +
               "Content-Disposition: form-data; name=\"" + kv.second.name +
               "\"; filename=\"" + kv.second.filename +
               "\"\r\n\r\n" + kv.second.content + "\r\n";
    }
    out += "--" + std::string(boundary) + "--\r\n";
    return out;
}

struct Server {
    Config cfg;
    Metrics metrics;
    Forwarder forwarder;
    MediaForwarder media_forwarder;
    httplib::Server http;
    time_t start_at = 0;
    std::thread http_thread;
    std::string argv0;   // the APE file; the children re-run it
    std::string config_path;  // the config file; the media child re-parses it

    Server(Config c, std::string a0, std::string cp)
        : cfg(std::move(c)), forwarder(cfg), media_forwarder(cfg),
          argv0(std::move(a0)), config_path(std::move(cp)) {}

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
        // The request is relayed to the media child, which owns the audio
        // decode and the engine; serve validates the model up front (the
        // child re-validates) so a config without any media models -- and
        // thus without a media child -- still answers with the right
        // 400/404.  The multipart upload is rebuilt from the parts this
        // httplib fork parsed into req.form (it leaves req.body empty for
        // multipart); the child re-parses the rebuilt form.
        http.Post("/v1/audio/transcriptions",
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
            std::string model;
            for (const auto &kv : req.form.fields)
                if (kv.first == "model") {
                    model = kv.second.content;
                    break;
                }
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
            media_forwarder.forward("/v1/audio/transcriptions",
                                    build_multipart_body(req.form.fields,
                                                         req.form.files),
                                    kMultipartContentType, res);
            if (res.status == 200)
                metrics.transcribe_requests++;
        });

        // OpenAI: image generation (diffusion engine models).  Relayed to
        // the media child (JSON body as-is); the child owns the generation
        // parameters and the engine.  Serve validates the model up front,
        // like the transcriptions route.
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
            const ModelEntry *e =
                cfg.find(body["model"].get<std::string>());
            if (!e) {
                send_err(res, 404, "invalid_request_error", "model_not_found",
                         "unknown model id");
                return;
            }
            if (e->engine != EngineKind::Diffusion) {
                send_err(res, 400, "invalid_request_error",
                         "unsupported_model",
                         std::string("model '") + e->id +
                         "' runs on the " + engine_name(e->engine) +
                         " engine, not diffusion");
                return;
            }
            media_forwarder.forward("/v1/images/generations", req.body,
                                    "application/json", res);
            if (res.status == 200)
                metrics.image_requests++;
        });

        // OpenAI: image edits (diffusion engine models, img2img path).
        // Relayed to the media child (multipart rebuilt from req.form and
        // the model validated up front -- see the transcriptions route).
        http.Post("/v1/images/edits",
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
            std::string model;
            for (const auto &kv : req.form.fields)
                if (kv.first == "model") {
                    model = kv.second.content;
                    break;
                }
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
            if (e->engine != EngineKind::Diffusion) {
                send_err(res, 400, "invalid_request_error",
                         "unsupported_model",
                         std::string("model '") + e->id +
                         "' runs on the " + engine_name(e->engine) +
                         " engine, not diffusion");
                return;
            }
            media_forwarder.forward("/v1/images/edits",
                                    build_multipart_body(req.form.fields,
                                                         req.form.files),
                                    kMultipartContentType, res);
            if (res.status == 200)
                metrics.image_requests++;
        });

        // Chimerafile extension: what is loaded, what is queued, how long
        // the server has been up.  Resident rows merge the media child's
        // state with the llama models' state from the router child (which
        // also carries the model child's port); the queue is the media
        // child's pump queue (the serve process runs no engine work).
        http.Get("/v1/status", [this](const httplib::Request &,
                                      httplib::Response &res) {
            nlohmann::json out;
            out["listen"] = cfg.host + ":" + std::to_string(cfg.port);
            out["key"] = !cfg.key.empty();
            out["uptime_s"] = (long)(time(nullptr) - start_at);
            nlohmann::json media = media_forwarder.status();
            out["queue"] =
                media.value("queue",
                            nlohmann::json{{"pending", 0}, {"active", false}});
            out["resident"] = resident_json(media);
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
            for (const auto &r : resident_json(media_forwarder.status())) {
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

        // Chimerafile extensions: preload / explicit-unload controls,
        // dispatched by engine: llama models to the router child (which
        // loads and unloads them), media models to the media child.
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
            bool ok = e->engine == EngineKind::Llama
                          ? forwarder.preload(*e, err)
                          : media_forwarder.preload(*e, err);
            if (!ok) {
                res.status = 503;
                res.set_content(
                    nlohmann::json({{"error", {{"message", err},
                                               {"type", "server_error"},
                                               {"code", "model_load_failed"}}}})
                        .dump(-1, ' ', false, nlohmann::json::error_handler_t::replace),
                    "application/json");
                return;
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
                : media_forwarder.unload(*e, err);
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

    // The /v1/status and /metrics rows: media engines from the media
    // child, llama models from the router child.  The child's status is
    // passed in when the caller already fetched it (the status route also
    // needs it for the queue).
    nlohmann::json resident_json(const nlohmann::json &media_status) {
        nlohmann::json out = media_status.value("resident",
                                                nlohmann::json::array());
        auto llama = forwarder.resident_json();
        for (auto &r : llama)
            out.push_back(std::move(r));
        return out;
    }

    // Binds the port (our single-instance guard), starts both children, and
    // runs the accept loop on a background thread.  The serve process
    // itself runs no engine work (the media engines live in the media
    // child), so the main thread only waits for a signal.  Returns false
    // if the port could not be bound or a child would not come up.
    bool start(std::string &err) {
        register_routes();
        std::string ierr;
        if (!forwarder.init(argv0, ierr)) {
            err = ierr;
            return false;
        }
        if (!media_forwarder.init(argv0, config_path, ierr)) {
            err = ierr;
            return false;
        }
        forwarder.record_chat = [this](const nlohmann::json &u,
                                       const nlohmann::json &t) {
            metrics.record_forward(u, t);
        };
        start_at = time(nullptr);
        fprintf(stderr, "serve: listening on http://%s:%d\n",
                cfg.host.c_str(), cfg.port);
        if (!http.bind_to_port(cfg.host.c_str(), cfg.port)) {
            err = "cannot bind " + cfg.host + ":" + std::to_string(cfg.port) +
                  " (is another chimerafile serve running?)";
            return false;
        }
        // both children are up before any request can arrive
        if (!forwarder.start(err)) {
            http.stop();
            return false;
        }
        if (!media_forwarder.start(err)) {
            forwarder.stop();
            http.stop();
            return false;
        }
        http_thread = std::thread([this] { http.listen_after_bind(); });
        return true;
    }

    // Stop the accept loop, then let the media child go (it releases its
    // own models) and the router child (which takes the model children
    // with it).
    void shutdown() {
        http.stop();
        if (http_thread.joinable())
            http_thread.join();
        media_forwarder.stop();
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
static std::atomic<bool> g_stopping{false};
static void on_signal(int) {
    g_stopping = true;
    if (g_server)
        g_server->http.stop();
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

    // argv[0] is the APE file the children re-run through the APE runtime
    // (/proc/self/exe); config is the file the media child re-parses.
    serve::Server server(cfg, argv[0], config);
    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    if (!server.start(err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    // The serve process runs no engine work (llama lives in the router
    // child, the media engines in the media child), so the main thread
    // only waits for a signal; shutdown() then stops the accept loop and
    // the children.
    while (!g_stopping.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    server.shutdown();
    fprintf(stderr, "serve: stopped\n");
    return 0;
}
