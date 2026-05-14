# Chimerafile

A unified inference binary that combines Mozilla's **llamafile** (LLM) and **whisperfile** (speech-to-text) projects into a single [APE](https://justine.lol/ape.html) executable.

```
./chimerafile llama   [llamafile args...]
./chimerafile whisper [whisperfile args...]
```

Chimerafile is intentionally a minimal dispatcher addon, which extends an unmodified version of the original [llamafile](https://github.com/mozilla-ai/llamafile) repository.

---

## Requirements

- Linux; macOS and Windows haven't been tested (yet)
- Git
- GNU Make ≥ 4.0

No other dependencies — the build system downloads [cosmocc](https://github.com/jart/cosmopolitan) automatically.

---

## Quick Start

```sh
git clone https://github.com/you/chimerafile
cd chimerafile
make setup          # initialise submodule + download cosmocc (~50 MB)
make                # build (debug)
make MODE=rel       # build (optimised)
```

The binary is written to `llamafile/o//chimerafile/chimerafile` (debug) or `llamafile/o/rel/chimerafile/chimerafile` (release).

```sh
make install        # install to /usr/local/bin/chimerafile
make install PREFIX=~/.local   # or a custom prefix
```

---

## Usage

```sh
# LLM — all llamafile modes work unchanged
./chimerafile llama --cli   -m model.gguf -p "Summarise this podcast"
./chimerafile llama --server -m model.gguf --port 8080
./chimerafile llama --chat  -m model.gguf

# Speech-to-text — all whisperfile flags work unchanged
./chimerafile whisper -m whisper.gguf -f audio.wav
./chimerafile whisper -m whisper.gguf -f audio.wav --output-csv -of transcript
./chimerafile whisper --help

# Version / help
./chimerafile --version
./chimerafile --help
./chimerafile llama --help
./chimerafile whisper --help
```

---

## Architecture

```
chimerafile/              ← this repo
├── chimerafile.cpp       ← dispatcher (owns main(), ~80 lines)
├── BUILD.mk              ← build rules (injected into llamafile's make)
├── Makefile              ← thin wrapper, delegates to llamafile/Makefile
├── .gitmodules           ← pins llamafile submodule
└── llamafile/            ← git submodule (stock, unmodified)
    ├── llamafile/        ← LLM TUI + server
    ├── whisperfile/      ← speech-to-text CLI
    ├── llama.cpp/        ← LLM backend (submodule)
    ├── whisper.cpp/      ← ASR backend (submodule, patched by llamafile)
    └── ...
```

### How the dispatch works

`chimerafile.cpp` owns `main()`. The two backend entry points are renamed at
compile time using `-Dmain=<name>`, requiring **zero changes** to the llamafile
source tree:

| Source file | Compiled as |
|---|---|
| `llamafile/main.cpp` | `-Dmain=llamafile_main` |
| `whisperfile/whisperfile.cpp` | `-Dmain=whisperfile_main` |

`argv[1]` (`llama` or `whisper`) is consumed by the dispatcher; the rest of
the arguments are passed verbatim to the chosen backend.

### Single GGML

`whisper.cpp` ships its own GGML fork. Linking both GGMLs into one binary
produces duplicate symbols. Chimerafile resolves this by recompiling the
whisper sources against `llama.cpp`'s GGML headers, giving both backends a
single, GPU-capable GGML implementation.

### APE format

Because Chimerafile uses the same cosmocc toolchain as llamafile, the output
binary is automatically an [APE](https://justine.lol/ape.html) (Actually
Portable Executable) — a single file that runs natively on Linux, macOS,
Windows, and FreeBSD, on both x86_64 and aarch64.

---

## Bundling a model (optional)

A bare `chimerafile` binary requires `-m model.gguf` at runtime. To create a
self-contained single-file distribution with a model baked in, use llamafile's
`zipalign` tool after the build (same workflow as llamafile itself):

```sh
# Example: bundle a Qwen3 GGUF for LLM mode
cp llamafile/o/rel/chimerafile/chimerafile my-podcast-assistant.chimerafile
echo '--cli -m qwen3.gguf' > .args
llamafile/o//third_party/zipalign/zipalign -j0 \
    my-podcast-assistant.chimerafile \
    qwen3.gguf \
    .args
chmod +x my-podcast-assistant.chimerafile
```

---

## Roadmap

- **v0.1** — llama + whisper dispatch (this release)
- **v0.2** — diffusion image generation forced endpoint (`./chimerafile diffusion`) - next time!

## License

This project is provided under the GNU General Public License version 3 (GPLv3), and does not include code sourced directly from Mozilla's llamafile, nor is this project associated with such.

## AI-Usage Disclosure

Unfortunately, I was but the project manager, bullying Deepseek v4 into complying with my less than reasonable demands. I couldn't write code to save my life, but I hope it is of some use to you :)
