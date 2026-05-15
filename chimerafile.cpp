// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// chimerafile.cpp — unified entry point for Chimerafile
//
// Dispatches to llamafile or whisperfile based on the first argument:
//
//   ./chimerafile llama   [llamafile args...]
//   ./chimerafile whisper [whisperfile args...]
//
// The first argument is consumed; the remainder are passed verbatim to the
// chosen backend, with argv[0] preserved as the program name.
//
// Both backends are compiled with -Dmain=<backend>_main so their translation
// units do not define a competing main().  No modifications to the llamafile
// source tree are required.

#include <cosmo.h>
#include <stdio.h>
#include <string.h>

#include "llamafile/llamafile.h"

#ifndef CHIMERAFILE_VERSION_STRING
#define CHIMERAFILE_VERSION_STRING "0.1.0"
#endif

// ── Backend entry points ───────────────────────────────────────────────────
//
// whisperfile/whisperfile.cpp is compiled with -Dmain=whisperfile_main.
// That file calls whisper_cli_main() (already renamed by the whisper.cpp
// patch), so the chain is:
//   chimerafile main()
//     → whisperfile_main()          (whisperfile/whisperfile.cpp)
//       → whisper_cli_main()        (whisper.cpp/examples/cli/cli.cpp)
//
// For the llama backend, we do NOT compile llamafile/main.cpp because
// -Dmain=llamafile_main would corrupt lf::chatbot::main calls.  Instead,
// llama_dispatch.cpp contains the same mode-switching logic.

extern int llama_dispatch(int argc, char **argv);
extern int whisperfile_main(int argc, char **argv);
extern "C" int diffusionfile_main(int argc, char **argv);

// ── Usage ──────────────────────────────────────────────────────────────────

static void print_usage(FILE *f, const char *prog) {
    fprintf(f,
        "chimerafile v" CHIMERAFILE_VERSION_STRING "\n"
        "\n"
        "usage: %s <engine> [engine-args...]\n"
        "\n"
        "engines:\n"
        "  llama      LLM inference (chat, server, CLI)  — wraps llamafile\n"
        "  whisper    Speech-to-text transcription       — wraps whisperfile\n"
        "  diffusion  Image generation (txt2img / img2img)\n"
        "\n"
        "examples:\n"
        "  %s llama      --cli -m model.gguf -p \"Hello\"\n"
        "  %s llama      --server -m model.gguf\n"
        "  %s whisper    -m whisper.gguf -f audio.wav --output-csv\n"
        "  %s diffusion  -m model.safetensors -p \"a cat\"\n"
        "  %s --version\n"
        "\n"
        "Pass --help after the engine name for engine-specific help:\n"
        "  %s llama      --help\n"
        "  %s whisper    --help\n"
        "  %s diffusion  --help\n",
        prog, prog, prog, prog, prog, prog, prog, prog, prog);
}

// ── Entry point ────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    // Each backend handles its own init (cosmo_args, CPU checks, --version).
    // We only parse the engine name, shift argv, and dispatch.

    if (llamafile_has(argv, "--version")) {
        printf("chimerafile v" CHIMERAFILE_VERSION_STRING "\n");
        return 0;
    }

    if (argc < 2 ||
        strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0) {
        print_usage(argc < 2 ? stderr : stdout, argv[0]);
        return argc < 2 ? 1 : 0;
    }

    const char *engine = argv[1];

    // Shift: consume argv[1] (the engine name), keep argv[0] (program name).
    //   Before: argv = [ "chimerafile", "whisper", "-f", "audio.wav", ... ]
    //   After:  argv = [ "chimerafile",            "-f", "audio.wav", ... ]
    argv[1] = argv[0];
    argv++;
    argc--;

    if (strcmp(engine, "llama") == 0)
        return llama_dispatch(argc, argv);

    if (strcmp(engine, "whisper") == 0)
        return whisperfile_main(argc, argv);

    if (strcmp(engine, "diffusion") == 0)
        return diffusionfile_main(argc, argv);

    fprintf(stderr, "chimerafile: unknown engine '%s'\n\n", engine);
    print_usage(stderr, argv[0]);
    return 1;
}
