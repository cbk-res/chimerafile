// The llama forwarder.  One upstream router child (llama.cpp router mode)
// owns the model fleet -- spawn, readiness, crash respawns of the model
// children, proxying -- and this file owns the front half: it spawns that
// router child with a generated --models-preset, forwards the llama API to
// it, and carries the llama side of residency (TTL, pins, capacity
// admission).  Media engines never come here; they stay on serve.cpp's
// in-process pump.

#ifndef CHIMERAFILE_SERVE_FORWARD_H
#define CHIMERAFILE_SERVE_FORWARD_H

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <thread>

#include <nlohmann/json.hpp>
#include "serve_common.h"

namespace httplib { struct Request; struct Response; }

namespace serve {

struct Forwarder {
    // Set by Server before the first request: keeps the process-wide
    // /metrics counters fed from forwarded chat traffic.
    std::function<void(const nlohmann::json &usage, const nlohmann::json &timings)> record_chat;

    explicit Forwarder(const Config &c) : cfg(c) {}

    bool active() const { return has_llama && running.load(); }

    // Write the generated preset INI; called from Server::start() before
    // the port is bound.  `ape` is argv[0] of the serve process (the APE
    // file the children re-run).  No-op (and success) when the config has
    // no llama models.
    bool init(const std::string &ape, std::string &err);

    // Spawn the router child and wait for it to answer /health, then start
    // the TTL sweeper.  No-op (and success) with no llama models.
    bool start(std::string &err);
    void stop();

    // One chat completion through the router child.  `body` is the parsed
    // client request; the model id has already been validated by the
    // route.  Streams are relayed frame by frame (no buffering).
    void handle_chat(httplib::Response &res, const ModelEntry &e,
                     const nlohmann::ordered_json &body);

    // Residency controls for llama models (route: /v1/residency/...).
    bool preload(const ModelEntry &e, std::string &err);
    bool unload(const ModelEntry &e, std::string &err);

    // /v1/status rows for the llama models (the media rows come from
    // Residency).
    nlohmann::json resident_json() const;

private:
    const Config &cfg;

    bool has_llama = false;
    std::string ape_path;
    std::string work_dir;      // the generated INI and the router log
    std::string ini_path;
    std::string log_path;
    std::atomic<int> router_port{0};
    std::atomic<int> router_pid{0};
    int respawns = 0;           // auto-respawns in the last quarantine window
    long last_respawn = 0;

    struct Rec {
        std::atomic<long> last_used{0};
        std::atomic<int> in_use{0};
        std::atomic<bool> evicting{false};  // an unload is in flight for it
    };
    std::map<std::string, Rec> recs;

    std::mutex mu;              // router_pid/port and respawn bookkeeping
    // Last /v1/models snapshot, published by the sweeper and read by the
    // deep chat path (see fetch_models).
    mutable std::mutex models_mu;
    nlohmann::json models_cache;
    std::thread sweeper;
    std::atomic<bool> running{false};

    bool spawn_router(std::string &err);
    bool router_healthy() const;
    bool wait_healthy(long timeout_s);
    bool respawn_router();

    // One GET /v1/models from the router, reduced to what the front half
    // needs: per model name the mapped state (loaded|loading|unloaded),
    // the raw state, and the child's port while running (the router
    // reports it in the status args).  The deep chat path
    // (ensure_capacity/wait_for_state) must not call this: a json parse
    // at that call depth overflows the small worker thread stacks, so it
    // reads the sweeper's snapshot instead.  Handler-depth routes
    // (status, residency controls) fetch live.
    nlohmann::json fetch_models() const;
    nlohmann::json cached_models() const;
    bool wait_for_state(const std::string &id, const std::string &want,
                        long timeout_s, std::string &err);
    bool ensure_capacity(const ModelEntry &e, std::string &err);
    void sweep_loop();
    static bool write_preset_ini(const Config &cfg, const std::string &path,
                                 std::string &err);
};

} // namespace serve

#endif
