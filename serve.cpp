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
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <sys/socket.h>

#include <nlohmann/json.hpp>
#include <cpp-httplib/httplib.h>

namespace serve {

enum class EngineKind { Llama, Whisper, Diffusion, Transcribe };

inline const char *engine_name(EngineKind e) {
    switch (e) {
        case EngineKind::Llama:      return "llama";
        case EngineKind::Whisper:    return "whisper";
        case EngineKind::Diffusion:  return "diffusion";
        case EngineKind::Transcribe: return "transcribe";
    }
    return "?";
}

// Engine defaults: the user-facing contract is 30 min for LLMs, 5 min for
// everything else.
inline int default_ttl(EngineKind e) {
    return e == EngineKind::Llama ? 1800 : 300;
}

struct ModelEntry {
    std::string id;
    EngineKind engine = EngineKind::Llama;
    std::string file;       // resolved
    int ctx = 4096;         // llama
    int gpu_layers = -1;    // llama
    int ttl = -1;           // idle seconds; -1 = engine default, 0 = never
    bool pinned = false;
    std::vector<std::string> args;

    int effective_ttl() const { return ttl < 0 ? default_ttl(engine) : ttl; }
};

struct Config {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string key;                 // empty = no auth
    std::vector<ModelEntry> models;

    const ModelEntry *find(const std::string &id) const {
        for (const auto &m : models)
            if (m.id == id) return &m;
        return nullptr;
    }
};

// ── Helpers ──────────────────────────────────────────────────────────────

static std::string to_lower(std::string s) {
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
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
        }
        if (v.contains("gpu_layers")) {
            if (!v["gpu_layers"].is_number_integer() ||
                v["gpu_layers"].get<int>() < -1) {
                err = "models." + m.id + ": 'gpu_layers' must be an integer >= -1";
                return false;
            }
            m.gpu_layers = v["gpu_layers"].get<int>();
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

    return true;
}

// ── HTTP server ──────────────────────────────────────────────────────────

struct Server {
    Config cfg;
    httplib::Server http;
    time_t start_at = 0;

    explicit Server(Config c) : cfg(std::move(c)) {}

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
        http.set_pre_routing_handler([this](const httplib::Request &req,
                                            httplib::Response &res)
                                            -> httplib::Server::HandlerResponse {
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
            res.set_content(out.dump(), "application/json");
        });

        // Chimerafile extension: what is loaded, what is queued, how long
        // the server has been up.  The residency manager fills this in with
        // live state in a later step.
        http.Get("/v1/status", [this](const httplib::Request &,
                                      httplib::Response &res) {
            nlohmann::json resident = nlohmann::json::array();
            for (const auto &m : cfg.models) {
                nlohmann::json r;
                r["id"] = m.id;
                r["engine"] = engine_name(m.engine);
                r["state"] = "unloaded";
                r["ttl_s"] = m.effective_ttl();
                r["pinned"] = m.pinned;
                resident.push_back(r);
            }
            nlohmann::json out;
            out["listen"] = cfg.host + ":" + std::to_string(cfg.port);
            out["key"] = !cfg.key.empty();
            out["uptime_s"] = (long)(time(nullptr) - start_at);
            out["queue"] = {{"pending", 0}, {"active", nullptr}};
            out["resident"] = resident;
            res.set_content(out.dump(), "application/json");
        });

        http.set_error_handler([](const httplib::Request &,
                                  httplib::Response &res) {
            if (res.status == 404 || res.status == 405) {
                res.set_content(
                    "{\"error\":{\"message\":\"no such route "
                    "(engine routing lands in the next steps)\","
                    "\"type\":\"invalid_request_error\",\"code\":null}}",
                    "application/json");
            }
        });
    }

    // Blocks until stop() (from a signal).  Returns false if the port could
    // not be bound — our single-instance guard.
    bool start(std::string &err) {
        register_routes();
        start_at = time(nullptr);
        fprintf(stderr, "serve: listening on http://%s:%d\n",
                cfg.host.c_str(), cfg.port);
        if (!http.listen(cfg.host.c_str(), cfg.port)) {
            err = "cannot bind " + cfg.host + ":" + std::to_string(cfg.port) +
                  " (is another chimerafile serve running?)";
            return false;
        }
        return true;
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
    fprintf(stderr, "serve: models\n");
    for (const auto &m : cfg.models) {
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

    serve::Server server(cfg);
    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    if (!server.start(err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    // listen() blocks here until SIGINT/SIGTERM.
    fprintf(stderr, "serve: stopped\n");
    return 0;
}
