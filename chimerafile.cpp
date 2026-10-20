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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <dlfcn.h>
#if defined(__linux__)
#include <sys/prctl.h>
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

// ── Entry point ────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    // Best effort: when our parent dies, get signalled instead of
    // orphaned.  The current APE runtime (.ape-1.10) swallows the prctl,
    // so on the APE this does not fire -- the guarantees that actually
    // hold there are serve's graceful stop (it reaps router and model
    // children) and the launcher killing the whole process group on a
    // hard stop (systemd KillMode=control-group, job control).  Native
    // builds honour it for every tree generation.
#if defined(__linux__)
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

    fprintf(stderr, "chimerafile: unknown engine '%s'\n\n", engine);
    print_usage(stderr, argv[0]);
    return 1;
}
