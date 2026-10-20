// The children of the serve layer.  One upstream router child (llama.cpp
// router mode) owns the llama fleet -- spawn, readiness, crash respawns of
// the model children, proxying -- and one media child (chimerafile media)
// owns the whisper / diffusion / transcribe engines.  This file owns the
// front half for both: it spawns the children, proxies the API to them, and
// carries the llama side of residency (TTL, pins, capacity admission).

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

    // Set by Server before the first request: reports whether the media
    // child has a model in "loading" state.  The llama side asks it in
    // ensure_capacity (the GPU lease): while the media child is loading a
    // model and a GPU is in play, a llama model load is deferred.  An
    // unset callback never defers.
    std::function<bool()> other_child_loading;

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

    // True while any llama model is loading (from the sweeper's snapshot):
    // the media side's lease gate asks this.
    bool any_loading() const;

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

// The media forwarder: the media child (chimerafile media) hosts the
// whisper / diffusion / transcribe engines on its own main thread; this
// owns the process that talks to it -- spawn, readiness, crash respawns,
// and the verbatim relay of the /v1/audio/* and /v1/images/* requests
// (the child re-parses the same config file and is the authority on
// validation).  Mirrors the llama forwarder; the shared spawn machinery
// lives in serve_forward.cpp.
struct MediaForwarder {
    // Set by Server before the first request: reports whether the router
    // (llama) child has a model in "loading" state.  The media side asks it
    // in forward/preload (the GPU lease, the mirror of
    // Forwarder::ensure_capacity): while the llama child is loading a model
    // and a GPU is in play, a media model load is deferred.
    std::function<bool()> other_child_loading;

    explicit MediaForwarder(const Config &c) : cfg(c) {}

    bool active() const { return has_media && running.load(); }

    // True while any media model is loading (from the status sweeper's
    // snapshot): the llama side's lease gate asks this.
    bool any_loading() const;

    // No-op (and success) when the config has no media models.  `ape` is
    // argv[0] of the serve process; `config` the config file path the
    // child re-parses.  Called from Server::start() before the port is
    // bound.
    bool init(const std::string &ape, const std::string &config,
              std::string &err);

    // Spawn the media child and wait for it to answer /health.  No-op
    // (and success) with no media models.
    bool start(std::string &err);
    void stop();

    // Relay one request to the child verbatim (path, body, Content-Type)
    // and copy the child's response (status, Content-Type, body) back.
    // On a dead child: respawn, retry once; still dead -> 502
    // media_unavailable.
    void forward(const std::string &path, const std::string &body,
                 const std::string &content_type, httplib::Response &res);

    // /v1/status merge inputs: the child's queue and resident rows (live
    // fetch).  Empty object when the child is absent or down.
    nlohmann::json status() const;

    // Residency controls for media models (route: /v1/residency/...).
    bool preload(const ModelEntry &e, std::string &err);
    bool unload(const ModelEntry &e, std::string &err);

private:
    const Config &cfg;

    bool has_media = false;
    std::string ape_path;
    std::string config_path;
    std::string work_dir;      // shared with the router child: <pid> dir
    std::string log_path;
    std::atomic<int> media_port{0};
    std::atomic<int> media_pid{0};
    int respawns = 0;           // auto-respawns in the last quarantine window
    long last_respawn = 0;

    std::mutex mu;              // media_pid/port and respawn bookkeeping
    std::atomic<bool> running{false};
    // Status sweeper interval in seconds (default 5); CHIMERAFILE_MEDIA_SWEEP_S
    // shortens it -- the lease tests run with a 1 s sweep so they can observe
    // the gate inside the model's load window.
    int sweep_s = 5;

    // Last /v1/status resident rows, published by the status sweeper and
    // read by the lease gate (model_state) and the llama side
    // (any_loading).  Stale by up to one sweep interval (~5 s by default).
    mutable std::mutex status_mu;
    nlohmann::json resident_cache;
    std::thread sweeper;
    // The child version the upgrade-skew warning was already printed for
    // (sweeper-only, no lock needed).
    std::string warned_version;

    bool spawn_media(std::string &err);
    bool media_healthy() const;
    bool wait_healthy(long timeout_s);
    bool respawn_media();

    // The status sweeper: every sweep_s (default 5) seconds pull the
    // child's /v1/status and keep its resident rows for the lease gate.  A
    // failed pull (child down) leaves the last known rows in place.
    void sweep_loop();
    nlohmann::json cached_resident() const;
    // The last known state of one media model ("unloaded" when unknown):
    // the lease gate defers a request whose model is not loaded.
    std::string model_state(const std::string &id) const;
};

} // namespace serve

#endif
