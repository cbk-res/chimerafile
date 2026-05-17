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
#include <string.h>

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
        "  chimerafile v" CHIMERAFILE_VERSION_STRING "  —  llama + whisper + diffusion in one APE\n"
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
    fprintf(stderr, "    diffusion  — text-to-image generation\n");
#else
    fprintf(stderr, "    diffusion  — (not built; set CHIMERAFILE_WITH_DIFFUSION=1)\n");
#endif
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
extern "C" int diffusionfile_main(int argc, char **argv);
#endif

// ── Usage ──────────────────────────────────────────────────────────────────

static void print_usage(FILE *f, const char *prog) {
    fprintf(f,
        "usage: %s <engine> [engine-args...]\n"
        "\n"
        "examples:\n"
        "  %s llama      --cli -m model.gguf -p \"Hello\"\n"
#ifndef CHIMERAFILE_NO_WHISPER
        "  %s whisper    -m whisper.gguf -f audio.wav --output-csv\n"
#endif
#ifndef CHIMERAFILE_NO_DIFFUSION
        "  %s diffusion  -m model.safetensors -p \"a cat\"\n"
#endif
        "  %s --version\n"
        "\n"
        "Pass --help after the engine name for engine-specific help:\n"
        "  %s llama      --help\n"
#ifndef CHIMERAFILE_NO_WHISPER
        "  %s whisper    --help\n"
#endif
#ifndef CHIMERAFILE_NO_DIFFUSION
        "  %s diffusion  --help\n"
#endif
        ,
        prog, prog
#ifndef CHIMERAFILE_NO_WHISPER
        , prog
#endif
#ifndef CHIMERAFILE_NO_DIFFUSION
        , prog
#endif
        , prog, prog
#ifndef CHIMERAFILE_NO_WHISPER
        , prog
#endif
#ifndef CHIMERAFILE_NO_DIFFUSION
        , prog
#endif
    );
}

// ── Entry point ────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (llamafile_has(argv, "--version")) {
        printf("chimerafile v" CHIMERAFILE_VERSION_STRING "\n");
        return 0;
    }

    // ── Parse --gpu MODE before banner (so GPU backend list is accurate) ──
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu") && i + 1 < argc) {
            const char *mode = argv[i + 1];
            if (!strcasecmp(mode, "off"))
                mode = "disable";
            FLAG_gpu = llamafile_gpu_parse(mode);
            memmove(&argv[i], &argv[i + 2], (argc - i - 2 + 1) * sizeof(char *));
            argc -= 2;
            break;
        }
    }

    // ── GPU backend loading ──────────────────────────────────────────
    if (FLAG_gpu != LLAMAFILE_GPU_DISABLE)
        llamafile_has_gpu();

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

    fprintf(stderr, "chimerafile: unknown engine '%s'\n\n", engine);
    print_usage(stderr, argv[0]);
    return 1;
}
