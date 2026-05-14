// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// llama_dispatch.cpp — llamafile mode dispatch for Chimerafile
//
// Contains the mode-dispatch logic from llamafile/main.cpp, wrapped as
// a regular function.  We do NOT compile llamafile/main.cpp because
// -Dmain=llamafile would corrupt lf::chatbot::main calls.
//
// All called functions are provided by LLAMAFILE_OBJS / LLAMAFILE_DEPS.

#include <cstdio>
#include <cstring>
#include <vector>

#include "args.h"
#include "chatbot.h"
#include "llamafile.h"
#include "version.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <vector>
#include <condition_variable>

#ifdef COSMOCC
#include <cosmo.h>
#endif

// Forward declarations — same as in llamafile/main.cpp
extern int server_main(int argc, char **argv,
                       std::function<void(const std::string &)> on_ready,
                       std::function<void(std::function<void()>)> on_shutdown_available);

// server_main is already forward-declared above using std::function signatures.

// cli_main is not in chatbot.h — forward-declare it here
namespace lf { namespace chatbot { extern int cli_main(int argc, char **argv); } }

// =============================================================================
// Help text — these are static in llamafile/main.cpp so we reproduce them here
// =============================================================================

static void print_general_help() {
    printf("chimerafile: llama engine — run LLMs locally\n"
           "(based on llamafile v" LLAMAFILE_VERSION_STRING ")\n"
           "\n"
           "usage: chimerafile llama -m MODEL.gguf [options]\n"
           "\n"
           "modes:\n"
           "  (default)   combined TUI chat + HTTP server\n"
           "  --server    HTTP server only (OpenAI-compatible API)\n"
           "  --chat      TUI chat only (no server)\n"
           "  --cli       single prompt/response (requires -p)\n"
           "\n"
           "common options:\n"
           "  -m FILE          path to GGUF model file (required)\n"
           "  -p TEXT          system prompt (in --cli mode: user prompt)\n"
           "  --gpu MODE       GPU backend (auto, nvidia, amd, apple, disable)\n"
           "  -ngl N           number of layers to offload to GPU\n"
           "  --verbose        enable verbose logging\n"
           "  --version        show version information\n"
           "  --help           show this help\n"
           "\n"
           "for mode-specific help and options:\n"
           "  chimerafile llama --server --help\n"
           "  chimerafile llama --chat --help\n"
           "  chimerafile llama --cli --help\n");
}

static void print_chat_help() {
    printf("chimerafile: llama engine --chat mode\n"
           "\n"
           "usage: chimerafile llama -m MODEL.gguf --chat [options]\n"
           "\n"
           "Interactive terminal chat. For full help see:\n"
           "  chimerafile llama --cli --help\n");
}

static void print_cli_help() {
    printf("chimerafile: llama engine --cli mode\n"
           "\n"
           "usage: chimerafile llama -m MODEL.gguf --cli -p \"prompt\" [options]\n"
           "\n"
           "Single prompt/response mode. All llama.cpp options accepted.\n"
           "For full help see:\n"
           "  chimerafile llama --help\n");
}

namespace lf {

// Context passed to the TUI thread via pthread (from main.cpp)
struct TuiThreadCtx {
    std::function<void()> *shutdown_fn;
    std::mutex *mu;
    std::condition_variable *cv;
    bool *shutdown_ready;
    std::string listen_addr;
    std::string system_prompt;
    std::string model_path;
};

static void *tui_thread_fn(void *arg) {
    auto *ctx = static_cast<TuiThreadCtx *>(arg);
    {
        std::unique_lock<std::mutex> lock(*ctx->mu);
        ctx->cv->wait(lock, [&] { return *ctx->shutdown_ready; });
    }
    chatbot::api_main(ctx->listen_addr, ctx->system_prompt, ctx->model_path, *ctx->shutdown_fn);
    delete ctx;
    return nullptr;
}

static int combined_main(const LlamafileArgs &args) {
    std::function<void()> shutdown_fn;
    pthread_t tui_tid = 0;
    std::mutex mu;
    std::condition_variable cv;
    bool shutdown_ready = false;

    auto on_ready = [&](const std::string &listen_addr) {
        auto *ctx = new TuiThreadCtx{
            &shutdown_fn, &mu, &cv, &shutdown_ready,
            listen_addr, args.system_prompt, args.model_path
        };
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
        pthread_create(&tui_tid, &attr, tui_thread_fn, ctx);
        pthread_attr_destroy(&attr);
    };

    auto on_shutdown = [&](std::function<void()> fn) {
        std::lock_guard<std::mutex> lock(mu);
        shutdown_fn = std::move(fn);
        shutdown_ready = true;
        cv.notify_one();
    };

    int rc = server_main(args.llama_argc, args.llama_argv, on_ready, on_shutdown);
    if (tui_tid)
        pthread_join(tui_tid, nullptr);
    return rc;
}

} // namespace lf

int llama_dispatch(int argc, char **argv) {
    // Load bundled default args from ZIP section (if any)
    argc = cosmo_args("/zip/.args", &argv);

    // Handle --version before anything else
    if (llamafile_has(argv, "--version")) {
        printf("chimerafile llama v" LLAMAFILE_VERSION_STRING "\n");
        return 0;
    }

    // Handle --help for the chimerafile llama prefix
    // This is called after cosmo_args but before parse_llamafile_args
    // so that bundled .args files can add -m automatically.

    lf::LlamafileArgs args = lf::parse_llamafile_args(argc, argv);

    if (llamafile_has(argv, "--help") || llamafile_has(argv, "-h")) {
        switch (args.mode) {
            case lf::ProgramMode::SERVER: break;
            case lf::ProgramMode::AUTO: print_general_help(); return 0;
            case lf::ProgramMode::CHAT: print_chat_help();    return 0;
            case lf::ProgramMode::CLI:  print_cli_help();     return 0;
        }
    }

    if (args.model_path.empty() &&
        !llamafile_has(argv, "--help") && !llamafile_has(argv, "-h")) {
        fprintf(stderr, "error: missing required -m MODEL.gguf\n\n");
        switch (args.mode) {
            case lf::ProgramMode::SERVER: print_general_help(); break;
            case lf::ProgramMode::AUTO:   print_general_help(); break;
            case lf::ProgramMode::CHAT:   print_chat_help();    break;
            case lf::ProgramMode::CLI:    print_cli_help();     break;
        }
        return 1;
    }

    if (!FLAG_verbose) {
        llama_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);
        llamafile_metal_log_set(llamafile_log_callback_null, NULL);
        llamafile_cuda_log_set(llamafile_log_callback_null, NULL);
    }

    if (args.mode == lf::ProgramMode::CLI)
        FLAG_nologo = 1;

    static char log_flag[] = "--log-verbosity";
    static char log_val[] = "1";
    std::vector<char *> quiet_argv;
    if (!FLAG_verbose && args.mode != lf::ProgramMode::SERVER) {
        for (int i = 0; i < args.llama_argc; i++)
            quiet_argv.push_back(args.llama_argv[i]);
        quiet_argv.push_back(log_flag);
        quiet_argv.push_back(log_val);
        quiet_argv.push_back(nullptr);
        args.llama_argc = static_cast<int>(quiet_argv.size()) - 1;
        args.llama_argv = quiet_argv.data();
    }

    llamafile_has_gpu();

    switch (args.mode) {
        case lf::ProgramMode::SERVER:
            return server_main(args.llama_argc, args.llama_argv, nullptr, nullptr);
        case lf::ProgramMode::CHAT:
            return lf::chatbot::main(args.llama_argc, args.llama_argv);
        case lf::ProgramMode::CLI:
            return lf::chatbot::cli_main(args.llama_argc, args.llama_argv);
        case lf::ProgramMode::AUTO:
            return lf::combined_main(args);
    }
    return 1;
}
