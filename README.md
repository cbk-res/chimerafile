# Chimerafile

[![build](https://github.com/cbk-res/chimerafile/actions/workflows/build.yml/badge.svg)](https://github.com/cbk-res/chimerafile/actions/workflows/build.yml)

A unified inference binary that combines **llamafile** (LLM), **whisperfile** (speech-to-text), and **stable-diffusion.cpp** ('diffusionfile' for image generation) into a single [APE](https://justine.lol/ape.html) executable, while making zero changes to the upstream source.

```
./chimerafile llama      [llamafile args...]
./chimerafile whisper    [whisperfile args...]
./chimerafile diffusion  [diffusion args...]
```

All three engines share a **single GGML runtime** — reducing duplication which might occur when projects wish to make use of one or more 'llamafiles' (e.g. an app which uses llamafile, whisperfile, and diffusionfile). We include several GPU backends (Vulkan, CUDA, ROCm, Metal), which are bundled as `.so` files inside the APE and loaded at runtime.

---

## Downloads

Pre-built binaries from [Releases](https://github.com/cbk-res/chimerafile/releases) (CI-built on every tag):

| Artifact | Backends | Use case |
|---|---|---|
| `chimerafile-VERSION-cpu` | CPU | Lightest, works everywhere |
| `chimerafile-VERSION-cpu+vulkan` | CPU + Vulkan | Linux (and probably Windows) with Vulkan drivers |
| `chimerafile-VERSION-cpu+cuda` | CPU + CUDA | Linux (and probably Windows) with NVIDIA GPU |
| `chimerafile-VERSION-cpu+rocm` | CPU + ROCm | Linux with AMD GPU |
| `chimerafile-VERSION-cpu+metal` | CPU + Metal | macOS (Intel & Apple Silicon) |
| `chimerafile-VERSION-cpu+all-gpu` | CPU + Vulkan + CUDA + ROCm | Universal GPU support (outwith macOS) |

---

## Quick Start

```sh
git clone https://github.com/cbk-res/chimerafile
cd chimerafile
make setup          # initialise submodule + download cosmocc (~50 MB)
make                # build (debug)
make MODE=rel       # build (optimised, ~20 min first build)
```

The binary is written to `llamafile/o//chimerafile/chimerafile`.

---

## Usage

### LLM — all existing llamafile functionality is preserved
```sh
./chimerafile llama --cli   -m model.gguf -p "Summarise this podcast"
./chimerafile llama --server -m model.gguf --port 8080
./chimerafile llama --chat  -m model.gguf
```

### Speech-to-text — whsiperfile is similarly unchanged
```sh
./chimerafile whisper -m whisper.gguf -f audio.wav
./chimerafile whisper -m whisper.gguf -f audio.wav --output-csv -of transcript
```

### Image generation — 'diffusionfile' builds on the patches for stable-diffusion.cpp present within the llamafile submodule, providing text-to-image generation and more!
```sh
./chimerafile diffusion -m model.safetensors -p "a cat in a spacesuit"
./chimerafile diffusion -m model.safetensors -p "portrait" -W 1024 -H 1024 --steps 30
./chimerafile diffusion -m model.safetensors -p "turn into van gogh" \
    --input photo.jpg --strength 0.6
./chimerafile diffusion -m model.gguf -p "donkey" --verbose
```

---

## GPU Backends

GPU acceleration is automatic — no flags needed.  Use `--gpu` to override:

```sh
# Central flag (applies to all engines):
./chimerafile --gpu nvidia llama -m model.gguf -p "Hello"
./chimerafile --gpu off diffusion -m model.gguf -p "cat"

# Per-engine override:
./chimerafile llama --gpu nvidia -m model.gguf -p "Hello"
./chimerafile diffusion --gpu vulkan -m model.gguf -p "cat"
```

Modes: `auto` (default), `nvidia`, `amd`, `vulkan`, `apple`, `disable`, `off`.

> **Known issue:** The Vulkan backend may crash on Intel integrated GPUs
> (Mesa ANV driver).  If you see a segfault on `--gpu auto`, try specifying a different GPU mode (e.g. `--gpu nvidia`) if avaialble or use `--gpu off` as a fallback.

Run with `--verbose` to see detected backends.

Building GPU backends requires the respective SDK:
```sh
# Vulkan (needs Vulkan SDK from lunarg.com)
make MODE=rel vulkan && make MODE=rel

# CUDA (needs CUDA toolkit)
make MODE=rel cuda && make MODE=rel

# ROCm (needs ROCm stack)
make MODE=rel rocm && make MODE=rel
```

Each produces a `.so` that gets zipaligned into the APE automatically.

---

## Project Structure

```
chimerafile/                    ← this repo (chimerafile layer only)
├── chimerafile.cpp             ← dispatcher (owns main())
├── llama_dispatch.cpp          ← llamafile mode (not compiled from submodule)
├── diffusionfile/
│   ├── diffusionfile.cpp       ← diffusion engine entry point
│   ├── sd_core.cpp             ← stb conflict wrapper
│   └── sd_ggml_compat.h        ← GGML API compatibility shim
├── BUILD.mk                    ← build rules (injected into llamafile's make)
├── Makefile                    ← thin wrapper, delegates to llamafile/Makefile
├── .github/workflows/build.yml ← CI: 6 variant builds
└── llamafile/                  ← git submodule (stock, unmodified)
    ├── llamafile/              ← LLM TUI + server
    ├── whisperfile/            ← speech-to-text CLI
    ├── stable-diffusion.cpp/   ← image generation library
    ├── llama.cpp/              ← LLM backend (submodule)
    └── whisper.cpp/            ← ASR backend (submodule)
```

### Single GGML — all three backends share one runtime

`whisper.cpp` and `stable-diffusion.cpp` each ship their own GGML fork.
Linking both into one binary would produce duplicate symbols. Chimerafile
resolves this by recompiling both backends against **llama.cpp's GGML**
headers, giving all three engines a single, GPU-capable GGML implementation
(Vulkan / CUDA / ROCm / Metal).

For `stable-diffusion.cpp`, a compatibility shim (`sd_ggml_compat.h`) bridges
six API differences between its upstream GGML fork and llama.cpp's GGML:

| API | Resolution |
|---|---|
| `ggml_cpu_has_blas()` | Stub (returns 0) |
| `ggml_internal_get_type_traits()` → `ggml_get_type_traits()` | Struct-by-value wrapper |
| `ggml_upscale(ctx, a, 2)` (3-arg) → 4-arg | Macro appends `GGML_SCALE_MODE_NEAREST` |
| `ggml_set_f32()` / `ggml_set_f32_1d()` | Provided by `ggml-cpu.h` |
| `ggml_graph_compute_with_ctx()` | Provided by `ggml-cpu.h` |
| `ggml_backend_cpu_init()` → GPU-first | Linker `--wrap` tries Vulkan/CUDA first |

### Dispatch model

Backend entry points are renamed at compile time (`-Dmain=<name>`),
requiring **zero changes** to the llamafile source tree:

| Source file | Compiled as |
|---|---|
| `llamafile/llamafile/main.cpp` | `-Dmain=llamafile_main` (via `llama_dispatch.cpp`) |
| `whisperfile/whisperfile.cpp` | `-Dmain=whisperfile_main` |
| `diffusionfile/diffusionfile.cpp` | `-Dmain=diffusionfile_main` |

`argv[1]` is consumed by the dispatcher; the remainder passes verbatim to
the chosen backend.

---


---
# Custom Build Options

### build-time engine toggles:
make MODE=rel CHIMERAFILE_WITH_WHISPER=0               # exclude whisper
make MODE=rel CHIMERAFILE_WITH_DIFFUSION=0              # exclude diffusion
make MODE=rel CHIMERAFILE_WITH_WHISPER=0 CHIMERAFILE_WITH_DIFFUSION=0  # llama only

# Disable specific GPU backends:
make MODE=rel CHIMERAFILE_WITH_VULKAN=0                # skip Vulkan
make MODE=rel CHIMERAFILE_WITH_CUDA=0 CHIMERAFILE_WITH_ROCM=0  # skip CUDA + ROCm

## Bundling a model (optional)

The workflow here should behave the same as it does within Mozilla's llamafile — using `zipalign` to embed a GGUF into the APE:

```sh
cp llamafile/o/rel/chimerafile/chimerafile my-model
echo '--cli -m qwen3.gguf' > .args
llamafile/o/rel/third_party/zipalign/zipalign -j0 \
    my-model \
    qwen3.gguf \
    .args
chmod +x my-model
```
However, this has not yet been tested within chimerafile itself.

---

## Roadmap

- **v0.1** — llama + whisper dispatch
- **v0.2** — diffusion image generation (`stable-diffusion.cpp` backend, single GGML, GPU compat shim)
- **v0.3** — support for further, optional backends is planned

## License

This project is provided under the GNU General Public License version 3 (GPLv3), and does not include code sourced directly from Mozilla's llamafile, nor is this project associated with such.

## AI-Usage Disclosure

Unfortunately, I was but the project manager, bullying Deepseek v4 into complying with my less than reasonable demands. I couldn't write code to save my life, but I hope it is of some use to you :)
