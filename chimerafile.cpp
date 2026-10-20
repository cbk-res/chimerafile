// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// chimerafile.cpp — unified entry point for Chimerafile
//
// Dispatches to llamafile, whisperfile, or diffusionfile based on the
// first argument.  Build-time toggles (CHIMERAFILE_NO_WHISPER,
// CHIMERAFILE_NO_DIFFUSION) are set by BUILD.mk when the corresponding
// engine is disabled.

#include <cosmo.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <signal.h>
#include <thread>
#include <dlfcn.h>
#if defined(__unix__)
#include <sys/prctl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "llamafile/llamafile.h"

#ifndef CHIMERAFILE_VERSION_STRING
#define CHIMERAFILE_VERSION_STRING "0.1.0"
#endif

// ── Banner ─────────────────────────────────────────────────────────────────

static void print_banner(void) {
    fprintf(stderr,
        " ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓██████████████▓▒░░▒▓████████▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░▒▓█▓▒░      ░▒▓████████▓▒░\n"
        "░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░       \n"
        "░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░       \n"
        "░▒▓█▓▒░      ░▒▓████████▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓███████▓▒░░▒▓████████▓▒░▒▓██████▓▒░ ░▒▓█▓▒░▒▓█▓▒░      ░▒▓██████▓▒░  \n"
        "░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░       \n"
        "░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░       \n"
        " ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓████████▓▒░▒▓████████▓▒░\n"
        "\n"
        "  chimerafile v" CHIMERAFILE_VERSION_STRING "  —  llama + whisper + diffusion + transcribe in one APE\n"
        "\n"
    );
}

static void print_engines(void) {
    fprintf(stderr, "  engines:\n"
                    "    llama      — LLM inference (chat, server, CLI)\n");
#ifndef CHIMERAFILE_NO_WHISPER
    fprintf(stderr, "    whisper    — speech-to-text transcription\n");
#else
    fprintf(stderr, "    whisper    — (not built; set CHIMERAFILE_WITH_WHISPER=1)\n");
#endif
#ifndef CHIMERAFILE_NO_DIFFUSION
    fprintf(stderr, "    diffusion  — text-to-image / video generation\n");
#else
    fprintf(stderr, "    diffusion  — (not built; set CHIMERAFILE_WITH_DIFFUSION=1)\n");
#endif
#ifndef CHIMERAFILE_NO_TRANSCRIBE
    fprintf(stderr, "    transcribe — speech-to-text (transcribe.cpp)\n");
#else
    fprintf(stderr, "    transcribe — (not built; set CHIMERAFILE_WITH_TRANSCRIBE=1)\n");
#endif
    fprintf(stderr, "    serve      — OpenAI-compatible server (registry, TTL, one port)\n");
    fprintf(stderr, "\n");
}

static void print_gpu_backends(void) {
    // Probe /zip for bundled GPU backends, respecting --gpu mode
    fprintf(stderr, "  gpu backends:");
    bool any = false;
    bool probe_all = (FLAG_gpu == LLAMAFILE_GPU_AUTO);
    if ((probe_all || FLAG_gpu == LLAMAFILE_GPU_NVIDIA) && llamafile_file_exists("/zip/ggml-cuda.so")) {
        fprintf(stderr, " cuda");
        any = true;
    }
    if ((probe_all || FLAG_gpu == LLAMAFILE_GPU_VULKAN) && llamafile_file_exists("/zip/ggml-vulkan.so")) {
        fprintf(stderr, " vulkan");
        any = true;
    }
    if ((probe_all || FLAG_gpu == LLAMAFILE_GPU_AMD) && llamafile_file_exists("/zip/ggml-rocm.so")) {
        fprintf(stderr, " rocm");
        any = true;
    }
#ifdef __APPLE__
    if (probe_all || FLAG_gpu == LLAMAFILE_GPU_APPLE) {
        fprintf(stderr, " metal");
        any = true;
    }
#endif
    if (FLAG_gpu == LLAMAFILE_GPU_DISABLE)
        fprintf(stderr, " none (--gpu off)");
    else if (!any)
        fprintf(stderr, " none (CPU only)");
    fprintf(stderr, "\n\n");
}

// ── Backend entry points ───────────────────────────────────────────────────

extern int llama_dispatch(int argc, char **argv);
#ifndef CHIMERAFILE_NO_WHISPER
extern int whisperfile_main(int argc, char **argv);
#endif
#ifndef CHIMERAFILE_NO_DIFFUSION
extern int diffusionfile_main(int argc, char **argv);
#endif
#ifndef CHIMERAFILE_NO_TRANSCRIBE
extern int transcribefile_main(int argc, char **argv);
#endif
extern int serve_main(int argc, char **argv);
// The media child: the in-process home of the whisper / diffusion /
// transcribe engines, spawned by the serve layer (not a user entry point).
extern int media_main(int argc, char **argv);

// The GPU backend this process resolved in main(); the serve layer hands it
// to the router child via LLAMAFILE_GPU.  nullptr = auto-detect, which the
// serve layer expresses by leaving the variable unset (the child then
// auto-detects too).
extern "C" const char *chimerafile_gpu_mode(void) {
    switch (FLAG_gpu) {
        case LLAMAFILE_GPU_DISABLE: return "off";
        case LLAMAFILE_GPU_AUTO:    return nullptr;
        case LLAMAFILE_GPU_AMD:     return "amd";
        case LLAMAFILE_GPU_APPLE:   return "apple";
        case LLAMAFILE_GPU_NVIDIA:  return "nvidia";
        case LLAMAFILE_GPU_VULKAN:  return "vulkan";
        default:                    return nullptr;
    }
}

// ── Usage ──────────────────────────────────────────────────────────────────

static void print_usage(FILE *f, const char *prog) {
    fprintf(f, "usage: %s <engine> [engine-args...]\n\n", prog);
    fprintf(f, "examples:\n");
    fprintf(f, "  %s llama      --cli -m model.gguf -p \"Hello\"\n", prog);
#ifndef CHIMERAFILE_NO_WHISPER
    fprintf(f, "  %s whisper    -m whisper.gguf -f audio.wav --output-csv\n", prog);
#endif
#ifndef CHIMERAFILE_NO_DIFFUSION
    fprintf(f, "  %s diffusion  -m model.safetensors -p \"a cat\"\n", prog);
#endif
#ifndef CHIMERAFILE_NO_TRANSCRIBE
    fprintf(f, "  %s transcribe -m model.bin -f audio.wav\n", prog);
#endif
    fprintf(f, "  %s serve      --config serve.json\n", prog);
    fprintf(f, "  %s --version\n\n", prog);
    fprintf(f, "Pass --help after the engine name for engine-specific help:\n");
    fprintf(f, "  %s llama      --help\n", prog);
#ifndef CHIMERAFILE_NO_WHISPER
    fprintf(f, "  %s whisper    --help\n", prog);
#endif
#ifndef CHIMERAFILE_NO_DIFFUSION
    fprintf(f, "  %s diffusion  --help\n", prog);
#endif
#ifndef CHIMERAFILE_NO_TRANSCRIBE
    fprintf(f, "  %s transcribe --help\n", prog);
#endif
}

// ── Host-glibc heap isolation ─────────────────────────────────────────────
//
// cosmo_dlopen() runs the host's ld-linux inside this APE. The host glibc then
// initialises its brk-based heap from the process brk, which sits at the base
// of this APE's own reserved heap arena (the ~30 GB region right after the
// data segment). Under sustained allocation (e.g. diffusion + the Vulkan
// driver) the two heap managers interleave in the same region and corrupt
// each other, destroying the in-process loader's link_map and segfaulting in
// _dl_fixup() on the next lazy PLT binding. Routing every host-glibc
// allocation through mmap (M_MMAP_THRESHOLD = 0) keeps the host heap out of
// the shared brk region and prevents the collision.
static void isolate_host_glibc_heap(void) {
#if defined(__linux__)
    void *libc = cosmo_dlopen("libc.so.6", RTLD_LAZY);
    if (libc) {
        typedef int (*mallopt_fn)(int, int);
        mallopt_fn host_mallopt = (mallopt_fn)cosmo_dlsym(libc, "mallopt");
        if (host_mallopt)
            host_mallopt(1 /* M_MMAP_THRESHOLD */, 0);
    }
#endif
}

// ── Parent-death watcher ─────────────────────────────────────────────────
//
// PR_SET_PDEATHSIG alone is not a usable guarantee in the APE: it was
// never even compiled there (cosmocc does not define __linux__), and the
// runtime's handling of it is unverified.  So a child whose parent dies
// would be orphaned: a hard kill of the serve process (task manager, a
// crash) would leave the router, the model children, and the media child
// running, holding their ports and the GPU.
// Every child of the serve layer is spawned with LLAMA_APP_CMD set (the
// router and media children directly, the router's model children by
// environment inheritance), so each of them installs this watcher on its
// direct parent.  When the parent dies the kernel reparents us: getppid()
// changes, or, if the pid was reused, the parent's start time (field 22 of
// /proc/<pid>/stat, which a reused pid cannot forge) stops matching.  On
// detection the child sends itself SIGTERM so its own graceful shutdown
// runs -- and a router's shutdown takes its model children down with it, so
// a death at any level cascades to the whole tree.  __unix__, not
// __linux__: the cosmocc APE toolchain does not define __linux__.
#if defined(__unix__)

// The process identity that survives pid reuse: field 22 (starttime) of
// /proc/<pid>/stat, which never changes for the process's lifetime.
static std::string proc_starttime(pid_t pid) {
    char path[64];
    snprintf(path, sizeof path, "/proc/%d/stat", (int)pid);
    FILE *f = fopen(path, "r");
    if (!f)
        return "";
    char buf[4096];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = '\0';
    // comm (field 2) may contain spaces and parentheses, so anchor on the
    // last ')': what follows is fields 3.., single-space separated.
    char *p = strrchr(buf, ')');
    if (!p)
        return "";
    p += 2;  // skip ") "
    for (int field = 3; field < 22; field++) {
        while (*p == ' ')
            p++;
        while (*p && *p != ' ')
            p++;
    }
    // Return field 22 alone: the rest of the line (vsize, rss, ...) is
    // mutable and would read as a changed identity.
    while (*p == ' ')
        p++;
    char *end = p;
    while (*end && *end != ' ')
        end++;
    if (end == p)
        return "";
    return std::string(p, (size_t)(end - p));
}

static void parent_death_watcher(void) {
    pid_t parent = getppid();
    std::string parent_start = proc_starttime(parent);
    if (parent_start.empty())
        return;  // cannot establish the parent's identity; do not act
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        pid_t cur = getppid();
        bool dead = cur != parent;
        if (!dead) {
            std::string now = proc_starttime(cur);
            dead = !now.empty() && now != parent_start;  // pid was reused
        }
        if (!dead)
            continue;
        fprintf(stderr, "parent process died; shutting down with it\n");
        // The child's own SIGTERM handler runs its graceful shutdown.  If
        // it is wedged, the backstop kills the process anyway -- any
        // in-flight work was already lost with the parent.
        raise(SIGTERM);
        for (int i = 0; i < 50; i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        _exit(1);
    }
}
#endif

// ── Entry point ────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    // Best effort: when our parent dies, get signalled instead of
    // orphaned.  Whether the cosmo runtime honours it inside the APE is
    // unverified -- the parent-death watcher below is the guarantee that
    // actually holds for serve-layer children either way, plus the
    // launcher killing the whole process group on a hard stop (systemd
    // KillMode=control-group, job control).  Native builds honour the
    // prctl for every tree generation.
#if defined(__unix__)
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif

    if (llamafile_has(argv, "--version")) {
        printf("chimerafile v" CHIMERAFILE_VERSION_STRING "\n");
        return 0;
    }

    // ── Parse --gpu MODE before banner (so GPU backend list is accurate) ──
    bool gpu_flag_seen = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu") && i + 1 < argc) {
            const char *mode = argv[i + 1];
            if (!strcasecmp(mode, "off"))
                mode = "disable";
            FLAG_gpu = llamafile_gpu_parse(mode);
            memmove(&argv[i], &argv[i + 2], (argc - i - 2 + 1) * sizeof(char *));
            argc -= 2;
            gpu_flag_seen = true;
            break;
        }
    }
    if (!gpu_flag_seen) {
        // Router-spawned model children carry no --gpu; the front router
        // exports the chosen backend via LLAMAFILE_GPU so grandchildren
        // pick up the same one.
        const char * gpu_env = getenv("LLAMAFILE_GPU");
        if (gpu_env != nullptr && gpu_env[0] != '\0') {
            const char *mode = gpu_env;
            if (!strcasecmp(mode, "off"))
                mode = "disable";
            FLAG_gpu = llamafile_gpu_parse(mode);
        }
    }

    // ── GPU backend loading ──────────────────────────────────────────
    if (FLAG_gpu != LLAMAFILE_GPU_DISABLE && llamafile_has_gpu())
        isolate_host_glibc_heap();

    // ── Banner (skip for --version, show for everything else) ────────
    print_banner();
    print_engines();
    print_gpu_backends();

    if (argc < 2 ||
        strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0) {
        print_usage(argc < 2 ? stderr : stdout, argv[0]);
        return argc < 2 ? 1 : 0;
    }

    const char *engine = argv[1];

    // Shift: consume argv[1] (the engine name), keep argv[0] (program name).
    argv[1] = argv[0];
    argv++;
    argc--;

#if defined(__unix__)
    // Children of the serve layer carry LLAMA_APP_CMD (spawn_child sets
    // it; the router's model children inherit it).  They must not outlive
    // their parent.  Installed just before the dispatch, so every engine
    // runs with the watcher in place.  Best effort: a failed start leaves
    // the child with the pre-watcher guarantees only (graceful stop, group
    // kill).
    if (getenv("LLAMA_APP_CMD")) {
        try {
            std::thread(parent_death_watcher).detach();
        } catch (...) {
            fprintf(stderr, "warning: cannot start the parent-death "
                            "watcher thread; the child will outlive a "
                            "hard kill of its parent\n");
        }
    }
#endif

    if (strcmp(engine, "llama") == 0)
        return llama_dispatch(argc, argv);

#ifndef CHIMERAFILE_NO_WHISPER
    if (strcmp(engine, "whisper") == 0)
        return whisperfile_main(argc, argv);
#endif

#ifndef CHIMERAFILE_NO_DIFFUSION
    if (strcmp(engine, "diffusion") == 0)
        return diffusionfile_main(argc, argv);
#endif

#ifndef CHIMERAFILE_NO_TRANSCRIBE
    if (strcmp(engine, "transcribe") == 0)
        return transcribefile_main(argc, argv);
#endif

    if (strcmp(engine, "serve") == 0)
        return serve_main(argc, argv);

    if (strcmp(engine, "media") == 0)
        return media_main(argc, argv);

    fprintf(stderr, "chimerafile: unknown engine '%s'\n\n", engine);
    print_usage(stderr, argv[0]);
    return 1;
}
