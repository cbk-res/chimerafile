// serve_config.cpp -- parsing of the declarative serve config (JSON).
// Shared by the serve layer and the media child, which parses the same
// config file (its --host/--port flags override the listen entry).

#include "serve_common.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace serve {

static bool parse_engine(const std::string &s, EngineKind &out) {
    std::string e = to_lower(s);
    if (e == "llama")      { out = EngineKind::Llama;      return true; }
    if (e == "whisper")    { out = EngineKind::Whisper;    return true; }
    if (e == "diffusion")  { out = EngineKind::Diffusion;  return true; }
    if (e == "transcribe") { out = EngineKind::Transcribe; return true; }
    return false;
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

bool parse_config(const char *path, Config &cfg, std::string &err) {
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

} // namespace serve
