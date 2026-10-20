// Shared types for the serve layer.  serve.cpp owns the config parsing,
// the in-process residency (media engines) and the HTTP surface;
// serve_forward.cpp owns the upstream router child and the llama
// forwarder.  Both work on these types, so they live here rather than in
// either translation unit.

#ifndef CHIMERAFILE_SERVE_COMMON_H
#define CHIMERAFILE_SERVE_COMMON_H

#include <cctype>
#include <string>
#include <vector>

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
    std::string alias_of;   // non-empty: this id is an alias for another model

    bool is_alias() const { return !alias_of.empty(); }
    int effective_ttl() const { return ttl < 0 ? default_ttl(engine) : ttl; }
};

// The sdcpp surface of the media child: the native /sdcpp/v1/* API of
// stable-diffusion.cpp's own server, mirrored so clients that point at an sd.cpp server work unchanged. Present in the config = enabled.
struct SdcppConfig {
    bool enabled = false;
    std::string model;          // diffusion model id that serves img_gen
    std::string upscalers_dir;  // hires upscaler models (optional)
};

struct Config {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string key;                 // empty = no auth
    int max_loaded = 1;              // llama models resident at once
    std::vector<ModelEntry> models;
    SdcppConfig sdcpp;

    const ModelEntry *find(const std::string &id) const {
        for (const auto &m : models)
            if (m.id == id) return &m;
        return nullptr;
    }
    // The model a request for `e` actually runs on: itself, or the model
    // an alias points at (an alias and its primary share one physical
    // child).  nullptr if the chain is broken.
    const ModelEntry *primary_of(const ModelEntry &e) const {
        const ModelEntry *m = &e;
        for (int hops = 0; m->is_alias() && hops < 16; ++hops) {
            const ModelEntry *t = find(m->alias_of);
            if (!t) return nullptr;
            m = t;
        }
        return m->is_alias() ? nullptr : m;
    }
};

// "serve: " prefixed stderr logging, shared with serve_forward.cpp.
void serve_log(const char *fmt, ...);

// Parse the declarative server config (JSON) at `path` into `cfg`
// (serve_config.cpp).  Relative model files resolve against the config
// file's directory and must exist; alias chains and pin-vs-capacity are
// validated here.  Shared by the serve layer and the media child.
bool parse_config(const char *path, Config &cfg, std::string &err);

inline std::string to_lower(std::string s) {
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

} // namespace serve

#endif
