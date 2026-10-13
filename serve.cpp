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
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <sys/socket.h>

// llama.cpp's official client machinery (common params, model init, sampler)
// comes from common.h/arg.h/sampling.h; it also pulls in llama.h and the json
// fwd header.
#include "common.h"
#include "arg.h"
#include "sampling.h"
#include "chat.h"
#include <nlohmann/json.hpp>
#include <cpp-httplib/httplib.h>
#include "whisper.h"
#include "stable-diffusion.h"
#include "transcribe.h"

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
    bool ctx_set = false;
    bool gpu_set = false;
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

static void serve_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "serve: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void serve_parse_usage(int, char **) {
    fprintf(stderr, "  (llama.cpp engine args: see 'chimerafile llama --cli --help')\n");
}

static std::string phys_key(const ModelEntry &e) {
    std::string k = std::string(engine_name(e.engine)) + "\x01" + e.file;
    if (e.engine == EngineKind::Llama)
        for (const auto &a : e.args)
            k += "\x01" + a;
    return k;
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

    // llama.cpp handles; cparams must outlive the init result
    common_params cparams;
    common_init_result_ptr init;
    common_chat_templates_ptr chat_tmpl;  // llama: the model's chat template
    // the other engines' handles
    whisper_context * wctx = nullptr;
    sd_ctx_t * sctx = nullptr;
    struct transcribe_model * tmodel = nullptr;

    std::atomic<int> state{0};    // 0 unloaded, 1 loading, 2 loaded
    std::atomic<long> last_used{0};
    std::atomic<int> in_use{0};   // reserved/running request count

    std::mutex mu;                // serializes load/unload of this handle
    std::condition_variable cv;
    std::string last_err;         // last load failure (set on the main thread)
};

static void load_llama(PhysModel &p, std::string &err) {
    const ModelEntry &e = *p.entries.front();
    // A re-load after TTL eviction starts from fresh defaults: the parser
    // asserts on stale values a previous load of this handle left behind.
    p.cparams = common_params();
    std::vector<std::string> strs = {"chimerafile serve", "-m", e.file};
    for (const auto &a : e.args)
        strs.push_back(a);
    std::vector<char *> argv;
    for (auto &s : strs)
        argv.push_back(s.data());
    argv.push_back(nullptr);

    // The engine's own CLI parser fills the params, so the config 'args'
    // are genuine llama.cpp options (--temp, --ctk, --ctv, ...).
    if (!common_params_parse((int)argv.size() - 1, argv.data(), p.cparams,
                             LLAMA_EXAMPLE_COMMON, serve_parse_usage)) {
        err = "llama: invalid engine args for model '" + e.id + "'";
        return;
    }
    // For what serve owns, the config is authoritative.
    if (e.ctx_set)
        p.cparams.n_ctx = e.ctx;
    if (e.gpu_set)
        p.cparams.n_gpu_layers = e.gpu_layers;

    p.init = common_init_from_params(p.cparams, false);
    if (!p.init) {
        err = "llama: failed to load '" + e.file + "' (out of memory?)";
        return;
    }
    p.chat_tmpl = common_chat_templates_init(p.init->model(), "");
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
    if (st != TRANSCRIBE_OK)
        err = std::string("transcribe: ") + transcribe_status_string(st);
}

static void load_engine(PhysModel &p, std::string &err) {
    switch (p.engine) {
        case EngineKind::Llama:      load_llama(p, err);      break;
        case EngineKind::Whisper:    load_whisper(p, err);    break;
        case EngineKind::Diffusion:  load_diffusion(p, err);  break;
        case EngineKind::Transcribe: load_transcribe(p, err); break;
    }
}

static void unload_engine(PhysModel &p) {
    // the chat template may reference the model; drop it first
    p.chat_tmpl.reset();
    if (p.init)
        p.init.reset();
    if (p.wctx) { whisper_free(p.wctx); p.wctx = nullptr; }
    if (p.sctx) { free_sd_ctx(p.sctx); p.sctx = nullptr; }
    if (p.tmodel) { transcribe_model_free(p.tmodel); p.tmodel = nullptr; }
}

// ── Chat generation (llama) ──────────────────────────────────────────────
//
// One OpenAI chat completion, built entirely from official llama.cpp
// client machinery: OpenAI messages in through
// common_chat_msgs_parse_oaicompat, the model's chat template applied
// through common_chat_templates_apply, sampling through a common_sampler
// built from the config's engine args (which llama.cpp's own CLI parser
// interpreted at load time) with the request's sampling fields applied
// on top.  In this APE the generation must run on the process' main
// thread (see Residency below), so run_chat is executed by the pump
// through an InferJob; the HTTP workers only parse the request, wait,
// and serialize the response.

struct ChatRequest {
    nlohmann::ordered_json messages;
    float temperature = -1.0f;  // < 0 = not set in the request
    float top_p = -1.0f;
    int top_k = -1;
    int max_tokens = 0;         // 0 = as many as the context allows
    bool stream = false;
    bool seed_set = false;
    uint32_t seed = 0;
    std::vector<std::string> stop;
};

static bool parse_chat_request(const nlohmann::ordered_json &body,
                               ChatRequest &out, std::string &err) {
    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        err = "'messages' must be a non-empty array";
        return false;
    }
    out.messages = body["messages"];
    if (body.contains("temperature") && body["temperature"].is_number())
        out.temperature = body["temperature"].get<float>();
    if (body.contains("top_p") && body["top_p"].is_number())
        out.top_p = body["top_p"].get<float>();
    if (body.contains("top_k") && body["top_k"].is_number())
        out.top_k = body["top_k"].get<int>();
    if (body.contains("max_tokens") && body["max_tokens"].is_number())
        out.max_tokens = body["max_tokens"].get<int>();
    if (body.contains("max_completion_tokens") &&
        body["max_completion_tokens"].is_number())
        out.max_tokens = body["max_completion_tokens"].get<int>();
    if (body.contains("stream") && body["stream"].is_boolean())
        out.stream = body["stream"].get<bool>();
    if (body.contains("seed") && body["seed"].is_number()) {
        out.seed = body["seed"].get<uint32_t>();
        out.seed_set = true;
    }
    if (body.contains("stop")) {
        if (body["stop"].is_string())
            out.stop.push_back(body["stop"].get<std::string>());
        else if (body["stop"].is_array())
            for (const auto &s : body["stop"])
                if (s.is_string())
                    out.stop.push_back(s.get<std::string>());
    }
    if (body.contains("n") && body["n"].is_number() && body["n"].get<int>() != 1) {
        err = "'n' > 1 is not supported";
        return false;
    }
    return true;
}

struct GenResult {
    bool ok = false;
    std::string err;
    std::string text;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::string finish;  // "stop" | "length"
};

// Run one chat generation on a loaded model.  Must run on the process'
// main thread (the host-compiled GPU backend only has working
// host-glibc malloc state there).  on_piece is called for every decoded
// response piece; returning false aborts the generation early (e.g. the
// client went away).
static GenResult run_chat(PhysModel &p, const ChatRequest &req,
                          const std::function<bool(const char *, size_t)> &on_piece) {
    GenResult out;
    auto *model = p.init->model();
    auto *ctx = p.init->context();
    auto *vocab = llama_model_get_vocab(model);
    if (!p.chat_tmpl) {
        out.err = "model has no chat template";
        return out;
    }

    // Per-request sampler: the config's engine args with the OpenAI
    // sampling fields applied on top.
    common_params_sampling sp = p.cparams.sampling;
    if (req.temperature >= 0.0f) sp.temp = req.temperature;
    if (req.top_p >= 0.0f) sp.top_p = req.top_p;
    if (req.top_k >= 0) sp.top_k = req.top_k;
    if (req.seed_set) sp.seed = req.seed;
    auto *gs = common_sampler_init(model, sp);
    if (!gs) {
        out.err = "failed to initialize the sampler";
        return out;
    }

    // OpenAI messages -> the model's chat template -> prompt
    common_chat_templates_inputs in;
    in.messages = common_chat_msgs_parse_oaicompat(req.messages);
    auto chat = common_chat_templates_apply(p.chat_tmpl.get(), in);

    int32_t n_prompt = llama_tokenize(vocab, chat.prompt.data(),
                                      (int32_t)chat.prompt.size(), nullptr, 0,
                                      true, false);
    if (n_prompt < 0)
        n_prompt = -n_prompt;
    if (n_prompt > (int32_t)llama_n_ctx(ctx)) {
        out.err = "prompt is longer than the context (" +
                  std::to_string(llama_n_ctx(ctx)) + " tokens)";
        common_sampler_free(gs);
        return out;
    }
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, chat.prompt.data(), (int32_t)chat.prompt.size(),
                   prompt_tokens.data(), n_prompt, true, false);
    out.prompt_tokens = n_prompt;

    // Decode the prompt in batches (token positions track themselves).
    int n_batch = llama_n_batch(ctx);
    for (int i = 0; i < n_prompt; i += n_batch) {
        int n = std::min(n_batch, n_prompt - i);
        if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data() + i, n)) != 0) {
            out.err = "failed to decode the prompt";
            common_sampler_free(gs);
            return out;
        }
    }

    // Generate: sample, accept, emit, decode, repeat.  The sampler takes the
    // batch-token index of the last decoded token; -1 is the last output row,
    // which is the right one both after the prompt batch (only its final
    // token has logits) and after each 1-token generation batch.
    int max_tokens = req.max_tokens > 0 ? req.max_tokens
                                        : (int)llama_n_ctx(ctx) - n_prompt;
    for (int i = 0; i < max_tokens; i++) {
        llama_token tok = common_sampler_sample(gs, ctx, -1);
        common_sampler_accept(gs, tok, true);
        if (llama_vocab_is_eog(vocab, tok)) {
            out.finish = "stop";
            break;
        }
        char buf[1024];
        int n = llama_token_to_piece(vocab, tok, buf, (int32_t)sizeof buf, 0, false);
        if (n < 0) {
            out.err = "failed to decode a response token";
            common_sampler_free(gs);
            return out;
        }
        out.text.append(buf, n);
        out.completion_tokens++;
        if (!on_piece(buf, n))
            break;  // client went away
        bool stopped = false;
        for (const auto &s : req.stop) {
            if (!s.empty() && out.text.size() >= s.size() &&
                out.text.compare(out.text.size() - s.size(), s.size(), s) == 0) {
                out.text.erase(out.text.size() - s.size());
                stopped = true;
                break;
            }
        }
        if (stopped) {
            out.finish = "stop";
            break;
        }
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) {
            out.err = "failed to decode a response token";
            common_sampler_free(gs);
            return out;
        }
    }
    if (out.finish.empty())
        out.finish = "length";
    out.ok = true;
    common_sampler_free(gs);
    return out;
}

// One chat generation submitted to the main thread's pump.  The pump
// owns the run (in this APE only the main thread has a working
// host-glibc malloc state, which the host-compiled GPU backend
// allocates through); the submitting HTTP worker waits for `finished`
// (non-streaming) or drains `pieces` as they arrive (streaming).
struct InferJob {
    std::shared_ptr<PhysModel> p;
    ChatRequest req;

    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> pieces;  // pump -> streaming provider
    bool consumer_gone = false;      // the provider gave up (client gone)
    GenResult result;                // valid once `finished`
    bool finished = false;           // run_chat returned
};

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
        std::shared_ptr<InferJob> infer;                    // non-null: run a generation
    };
    mutable std::mutex queue_mu;
    std::condition_variable queue_cv;
    std::deque<LifeTask> tasks;
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
                tasks.push_back(LifeTask{p, false, nullptr});
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

    void handle_infer(std::shared_ptr<InferJob> job) {
        {
            std::lock_guard<std::mutex> lk(job->mu);
            if (job->consumer_gone) {
                // the client went away before the run could even start
                job->result.err = "client disconnected";
                job->finished = true;
                job->cv.notify_all();
            }
        }
        if (job->finished) {
            release(*job->p);
            return;
        }
        auto on_piece = [&job](const char *s, size_t n) -> bool {
            if (n == 0)
                return true;
            std::unique_lock<std::mutex> lk(job->mu);
            // bounded queue: backpressure on the pump if the provider is slow
            while (job->pieces.size() >= 256 && !job->consumer_gone)
                job->cv.wait(lk);
            if (job->consumer_gone)
                return false;
            job->pieces.emplace_back(s, n);
            job->cv.notify_all();
            return true;
        };
        job->result = run_chat(*job->p, job->req, on_piece);
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
                    tasks.push_back(LifeTask{p, true, nullptr});
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

    // Submit a generation to the main thread (worker thread).  The model
    // must already be acquired; the pump releases it when the run finishes.
    // Returns null if the pump is shutting down.
    std::shared_ptr<InferJob> submit_infer(std::shared_ptr<PhysModel> p,
                                           const ChatRequest &req) {
        if (!pump_running.load())
            return nullptr;
        auto job = std::make_shared<InferJob>();
        job->p = p;
        job->req = req;
        std::lock_guard<std::mutex> qk(queue_mu);
        tasks.push_back(LifeTask{p, false, nullptr, job});
        queue_cv.notify_one();
        return job;
    }

    nlohmann::json queue_json() const {
        std::lock_guard<std::mutex> qk(queue_mu);
        nlohmann::json q;
        q["pending"] = (int)tasks.size();
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
            tasks.push_back(LifeTask{p, false, result});
            queue_cv.notify_one();
        }
        err = result->get_future().get();
        return err.empty();
    }

    nlohmann::json resident_json() const {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto &e : cfg.models) {
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
                    return !tasks.empty() || !pump_running.load();
                });
                if (tasks.empty()) {
                    if (!pump_running.load())
                        break;
                    continue;
                }
                t = tasks.front();
                tasks.pop_front();
            }
            task_running = true;
            if (t.infer)
                handle_infer(t.infer);
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

struct Server {
    Config cfg;
    Residency residency;
    httplib::Server http;
    time_t start_at = 0;
    std::thread http_thread;

    Server(Config c) : cfg(std::move(c)), residency(cfg) {}

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

        // OpenAI: chat completions (llama engine models).
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
                res.set_content(out.dump(), "application/json");
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
            ChatRequest cr;
            std::string perr;
            if (!parse_chat_request(body, cr, perr)) {
                send_err(res, 400, "invalid_request_error", "invalid_request",
                         perr);
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
            std::string err;
            auto p = residency.acquire(*e, err);
            if (!p) {
                send_err(res, 503, "server_error", "model_load_failed", err);
                return;
            }

            std::string id = "chatcmpl-" + std::to_string(time(nullptr));
            long created = (long)time(nullptr);

            if (!cr.stream) {
                auto job = residency.submit_infer(p, cr);
                if (!job) {
                    residency.release(*p);
                    send_err(res, 503, "server_error", "server_busy",
                             "server is shutting down");
                    return;
                }
                GenResult g;
                {
                    std::unique_lock<std::mutex> lk(job->mu);
                    job->cv.wait(lk, [&] { return job->finished; });
                }
                g = job->result;
                if (!g.ok) {
                    send_err(res, 500, "server_error", "generation_failed",
                             g.err);
                    return;
                }
                nlohmann::json out;
                out["id"] = id;
                out["object"] = "chat.completion";
                out["created"] = created;
                out["model"] = e->id;
                nlohmann::json choice;
                choice["index"] = 0;
                choice["message"]["role"] = "assistant";
                choice["message"]["content"] = g.text;
                choice["finish_reason"] = g.finish;
                out["choices"] = {choice};
                out["usage"]["prompt_tokens"] = g.prompt_tokens;
                out["usage"]["completion_tokens"] = g.completion_tokens;
                out["usage"]["total_tokens"] =
                    g.prompt_tokens + g.completion_tokens;
                res.set_content(out.dump(), "application/json");
                return;
            }

            // Streaming: Server-Sent Events via a chunked content provider.
            // The generation itself runs on the main thread (pump); the
            // provider only copies pieces to the client as they arrive.
            auto job = residency.submit_infer(p, cr);
            if (!job) {
                residency.release(*p);
                send_err(res, 503, "server_error", "server_busy",
                         "server is shutting down");
                return;
            }
            res.set_chunked_content_provider(
                "text/event-stream",
                [job, id, created, model_id = e->id](
                    size_t, httplib::DataSink &sink) -> bool {
                    auto chunk = [&](const nlohmann::json &delta,
                                     const char *finish) {
                        nlohmann::json c;
                        c["id"] = id;
                        c["object"] = "chat.completion.chunk";
                        c["created"] = created;
                        c["model"] = model_id;
                        nlohmann::json ch;
                        ch["index"] = 0;
                        ch["delta"] = delta;
                        ch["finish_reason"] = finish
                            ? nlohmann::json(finish)
                            : nlohmann::json(nullptr);
                        c["choices"] = {ch};
                        std::string frame = "data: " + c.dump() + "\n\n";
                        return sink.write(frame.data(), frame.size());
                    };
                    auto gone = [&job] {
                        std::lock_guard<std::mutex> lk(job->mu);
                        job->consumer_gone = true;
                        job->cv.notify_all();
                    };
                    if (!chunk({{"role", "assistant"}}, nullptr)) {
                        gone();
                        return true;  // client gone
                    }
                    bool ok = false;
                    std::string finish;
                    {
                        std::unique_lock<std::mutex> lk(job->mu);
                        for (;;) {
                            if (!job->pieces.empty()) {
                                std::string piece =
                                    std::move(job->pieces.front());
                                job->pieces.pop_front();
                                lk.unlock();
                                nlohmann::json d;
                                d["content"] = std::move(piece);
                                if (!chunk(d, nullptr)) {
                                    gone();
                                    return true;  // client gone
                                }
                                lk.lock();
                                continue;
                            }
                            if (job->finished) {
                                ok = job->result.ok;
                                finish = job->result.finish;
                                break;
                            }
                            // a timeout so a stuck job cannot hang the
                            // provider forever
                            job->cv.wait_for(lk,
                                             std::chrono::milliseconds(200));
                        }
                    }
                    if (ok) {
                        chunk(nlohmann::json::object(), finish.c_str());
                        std::string done = "data: [DONE]\n\n";
                        sink.write(done.data(), done.size());
                    }
                    sink.done();
                    return true;
                });
        });

        // Chimerafile extension: what is loaded, what is queued, how long
        // the server has been up.
        http.Get("/v1/status", [this](const httplib::Request &,
                                      httplib::Response &res) {
            nlohmann::json out;
            out["listen"] = cfg.host + ":" + std::to_string(cfg.port);
            out["key"] = !cfg.key.empty();
            out["uptime_s"] = (long)(time(nullptr) - start_at);
            out["queue"] = residency.queue_json();
            out["resident"] = residency.resident_json();
            res.set_content(out.dump(), "application/json");
        });

        // Chimerafile extensions: preload / explicit-unload controls.
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
            auto p = residency.acquire(*e, err);
            if (!p) {
                res.status = 503;
                res.set_content(
                    nlohmann::json({{"error", {{"message", err},
                                               {"type", "server_error"},
                                               {"code", "model_load_failed"}}}})
                        .dump(),
                    "application/json");
                return;
            }
            residency.release(*p);  // keep it warm; the TTL clock starts now
            nlohmann::json out;
            out["id"] = e->id;
            out["state"] = "loaded";
            res.set_content(out.dump(), "application/json");
        });

        http.Post(R"(^/v1/residency/([A-Za-z0-9_-]+)/unload$)",
                  [this](const httplib::Request &req, httplib::Response &res) {
            const ModelEntry *e = cfg.find(req.matches[1].str());
            nlohmann::json out;
            out["id"] = e ? e->id : req.matches[1].str();
            if (!e) {
                res.status = 404;
                out["state"] = "unknown";
                res.set_content(out.dump(), "application/json");
                return;
            }
            std::string err;
            if (!residency.unload(*e, err)) {
                res.status = 409;
                out["state"] = "busy";
                out["error"] = err;
                res.set_content(out.dump(), "application/json");
                return;
            }
            out["state"] = "unloaded";
            res.set_content(out.dump(), "application/json");
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

    // Binds the port (our single-instance guard) and starts the accept loop
    // on a background thread; the calling (main) thread then owns the model
    // lifecycle via residency.pump().  Returns false if the port could not
    // be bound.
    bool start(std::string &err) {
        register_routes();
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

    serve::Server server(cfg);
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
