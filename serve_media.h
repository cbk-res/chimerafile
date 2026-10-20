// The media engines' in-process home.  A physical model is one loaded
// copy of (engine, file); several virtual models in the config that point
// at the same physical model share the one handle.  Loading uses each
// engine's own native loader -- whisper / stable-diffusion / transcribe
// through their C APIs -- so the layer never re-implements engine
// internals.  Models load on first use and unload after an idle TTL (or on
// eviction when a bigger model doesn't fit).
//
// In this APE, every engine call that touches the host-compiled libraries
// (load, unload and inference alike) must run on the process' main thread
// -- the only thread with a working host-glibc malloc state.  So
// everything engine-side is posted to the Residency queue and executed by
// the main thread's pump(); the HTTP workers only do HTTP and JSON.  The
// media child (serve_media.cpp) runs exactly that: its own main thread is
// the pump.

#ifndef CHIMERAFILE_SERVE_MEDIA_H
#define CHIMERAFILE_SERVE_MEDIA_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include "serve_common.h"

#include "whisper.h"
#include "stable-diffusion.h"
#include "transcribe.h"

namespace serve {

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

// One OpenAI image request, built on stable-diffusion.cpp's own
// generate_image(): an sd_img_gen_params_t carries the prompt, size,
// batch and (for /v1/images/edits) the decoded input image, and the raw
// pixels come straight back out of the engine.
struct ImageResult {
    bool ok = false;
    std::string err;
    struct Img {
        int w = 0, h = 0, ch = 0;
        std::vector<uint8_t> px;  // row-major, ch bytes per pixel
    };
    std::vector<Img> images;
};

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
    void push_task(LifeTask t);
    std::atomic<bool> pump_running{false};
    std::atomic<bool> task_running{false};   // the pump is handling a task

    std::thread sweeper;
    std::atomic<bool> running{false};

    std::shared_ptr<PhysModel> get_or_create(const ModelEntry &e);
    std::shared_ptr<PhysModel> find(const ModelEntry &e) const;

    // Evict the least-recently-used model that is not pinned, not in use,
    // and not the one that just failed to load.  Main thread only.
    bool evict_one(const PhysModel &self);
    void sweeper_loop();

    // ── main-thread handlers (run by pump()) ─────────────────────────────
    void handle_load(std::shared_ptr<PhysModel> p);
    void handle_unload(std::shared_ptr<PhysModel> p,
                       std::shared_ptr<std::promise<std::string>> result);
    // The engine work itself (transcription, generation, ...) lives in
    // the job's run closure, which each route builds; the pump only
    // executes it on the main thread and reports back.
    void handle_engine(std::shared_ptr<EngineJob> job);

public:
    explicit Residency(Config &c) : cfg(c) {}
    ~Residency() { stop(); }

    // Reserve a model for a request (worker thread).  If it is not loaded,
    // dispatches a load to the main thread and waits; the main thread
    // handles the eviction-retry loop.  Callers must release().
    std::shared_ptr<PhysModel> acquire(const ModelEntry &e, std::string &err);

    void release(PhysModel &p);

    // Submit engine work to the main thread (worker thread).  The model
    // must already be acquired; the pump releases it when the run
    // finishes.  Returns false if the pump is shutting down.
    bool submit_engine(std::shared_ptr<EngineJob> job);

    nlohmann::json queue_json() const;

    // Explicit unload (preload / free-the-GPU control): dispatched to the
    // main thread and waited on.
    bool unload(const ModelEntry &e, std::string &err);

    // Media engines only: llama models never pass through the residency
    // (their rows come from the forwarder's /v1/status merge).
    nlohmann::json resident_json() const;

    // Start the TTL sweeper.  The main thread then owns the lifecycle via
    // pump().
    void start();

    // Main thread only: process lifecycle tasks until stop_pump().
    void pump();

    // Ask pump() to exit.  Async-signal-safe (atomic store only); the pump
    // wakes within its 500 ms timeout.
    void stop_pump();

    // Main thread only, after pump() returned: stop the sweeper and release
    // every resident model (waiting briefly for in-flight requests).
    void stop();
};

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
bool decode_audio(const std::string &data, std::vector<float> &pcm);

// Run one transcription on a loaded model (main thread only).  pcm is 16
// kHz mono float32, as decode_audio produces.
TranscribeResult run_transcribe(PhysModel &p,
                                const std::vector<float> &pcm,
                                const std::string &language,
                                const std::string &prompt,
                                float temperature);

std::string build_srt(const std::vector<TranscribeResult::Segment> &segs);
std::string build_vtt(const std::vector<TranscribeResult::Segment> &segs);

// ── Image generation (diffusion) ─────────────────────────────────────────
//
// The engine context is one-shot: the pinned engine crashes on a second
// generate_image() against the same context (reproduced outside this
// server with a minimal harness against the engine alone), so each
// request loads a fresh context and frees it again afterwards -- a few
// seconds per request, in exchange for the rest of the design staying
// identical to the other engines.

// Run one generation on a loaded model (main thread only).
// edit is the decoded input image for image edits (nullptr = text-to-image).
ImageResult run_image(PhysModel &p, const std::string &prompt,
                      const std::string &negative, int w, int h,
                      int n, int steps, float cfg, long seed,
                      const std::vector<uint8_t> *edit,
                      int edit_w, int edit_h);

// Encode one image to PNG with stb_image_write (linked in via stb.a).
bool encode_png(const ImageResult::Img &img, std::string &out);

// Decode an uploaded image (PNG or JPEG) to 3-channel RGB with stb_image.
bool decode_image(const std::string &data, int &w, int &h,
                  std::vector<uint8_t> &px);

} // namespace serve

#endif
