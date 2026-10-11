# Chimerafile

[![build](https://github.com/cbk-res/chimerafile/actions/workflows/build.yml/badge.svg)](https://github.com/cbk-res/chimerafile/actions/workflows/build.yml)

A unified inference binary that combines **llamafile** (LLM), **whisperfile** (speech-to-text), **diffusionfile** (stable-diffusion.cpp image/video generation), and **transcribefile** (transcribe.cpp speech-to-text) into a single [APE](https://justine.lol/ape.html) executable, while making zero changes to the upstream source.

```
./chimerafile llama      [llamafile args...]
./chimerafile whisper    [whisperfile args...]
./chimerafile diffusion  [diffusionfile args...]
./chimerafile transcribe [transcribefile args...]
```

All engines share one **llama.cpp GGML runtime** (whisper and stable-diffusion.cpp are compiled against it; transcribe.cpp keeps its own private copy — see below), reducing duplication for projects that would otherwise bundle several standalone 'llamafiles'. GPU backends (Vulkan, CUDA, ROCm, Metal) are bundled as `.so` files inside the APE and loaded at runtime.

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

The binary is written to `llamafile/o/$(MODE)/chimerafile/chimerafile` (`o//` in debug builds, i.e. the empty-MODE path).

---

## Usage

### LLM — all existing llamafile functionality is preserved
```sh
./chimerafile llama --cli   -m model.gguf -p "Summarise this podcast"
./chimerafile llama --server -m model.gguf --port 8080
./chimerafile llama --chat  -m model.gguf
```

### Speech-to-text — whisperfile is unchanged
```sh
./chimerafile whisper -m whisper.gguf -f audio.wav
./chimerafile whisper -m whisper.gguf -f audio.wav --output-csv -of transcript
```

### Image/video generation — diffusionfile, as provided upstream
The diffusion engine is stable-diffusion.cpp's own CLI (built by the
`diffusionfile` package inside the llamafile submodule, with its `main()`
renamed for our dispatcher). Run `--help` for the full option list:
```sh
./chimerafile diffusion -m model.safetensors -p "a cat in a spacesuit"
./chimerafile diffusion -m model.safetensors -p "a cat" -o out.png
./chimerafile diffusion -M upscale -m model.safetensors -i photo.jpg
./chimerafile diffusion -m model.safetensors --version
```
Modes (`-M`): `img_gen` (default), `vid_gen`, `upscale`, `convert`, `metadata`.

### Speech-to-text — transcribefile (transcribe.cpp)
```sh
./chimerafile transcribe -m model.gguf audio.wav
./chimerafile transcribe -m model.gguf -l en audio.wav
./chimerafile transcribe --help
```
Note: the transcribe engine runs on its own private GGML copy (see below)
and is therefore CPU-only.

---

## GPU Backends

GPU acceleration for the llama/whisper/diffusion engines is automatic — no
flags needed. Use `--gpu` to override:

```sh
# Central flag (applies to all engines):
./chimerafile --gpu nvidia llama -m model.gguf -p "Hello"
./chimerafile --gpu off diffusion -m model.safetensors -p "cat"

# Per-engine override (engines that accept --gpu):
./chimerafile llama --gpu nvidia -m model.gguf -p "Hello"
```

Modes: `auto` (default), `nvidia`, `amd`, `vulkan`, `apple`, `disable`, `off`.

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

### Troubleshooting: "Vulkan is not usable" on SELinux Fedora

On SELinux-enforcing Fedora you may see:

```
fatal error: --gpu vulkan was explicitly requested but Vulkan is not usable
on this system
```

even though `vulkaninfo` lists your GPU. This is a host policy issue, not a
build issue. Cosmopolitan's `cosmo_dlopen` runs the host `ld-linux`
in-process; its loader zero-fills a page of the mapped loader before making
it executable, and SELinux's `file_mprotect` hook then requires the
`execmod` permission on the loader's file type. Fedora's policy grants
`unconfined_t` `execmod` on `user_home_t`/`user_tmp_t` but **not** on
`ld_so_t` (`/lib64`), so the `mprotect` fails with EACCES and the probe
aborts.

Workaround: relocate the loader into `$HOME` (type `user_home_t`) and point
the cached `dlopen-helper` at it — the APE reuses the helper when it is newer
than the binary. This repo ships the fixer at
`scripts/fix-cosmo-dlopen-selinux.sh` (no root needed); run it after
rebuilding the APE if the error returns.


---

## Project Structure

```
chimerafile/                    ← this repo (chimerafile layer only)
├── chimerafile.cpp             ← dispatcher (owns main())
├── llama_dispatch.cpp          ← llamafile mode dispatch
├── BUILD.mk                    ← build rules (injected into llamafile's make)
├── Makefile                    ← thin wrapper, delegates to llamafile/Makefile
├── ci-patches/                 ← CI-only patches (Pascal sm_61 for cuda.sh)
├── .github/workflows/build.yml ← CI: 6 variant builds
└── llamafile/                  ← git submodule (stock, unmodified, v0.10.6)
    ├── llamafile/              ← LLM TUI + server
    ├── whisperfile/            ← speech-to-text CLI wrapper
    ├── diffusionfile/          ← image generation CLI wrapper
    ├── transcribefile/         ← transcribe.cpp CLI wrapper
    ├── llama.cpp/              ← LLM backend (nested submodule)
    ├── whisper.cpp/            ← ASR backend (nested submodule)
    ├── stable-diffusion.cpp/   ← image generation library (nested submodule)
    └── transcribe.cpp/         ← ASR backend (nested submodule)
```

### Single GGML — mostly

`llamafile`, `whisperfile` and `diffusionfile` share **llama.cpp's GGML**:
whisper.cpp's core/common sources are recompiled against llama.cpp's GGML
headers, and stable-diffusion.cpp is built by the upstream diffusionfile
package against the same GGML — one GPU-capable runtime (Vulkan / CUDA /
ROCm / Metal) for all three engines.

`transcribe.cpp` is the exception: it vendors its own GGML (ggml-org/ggml
0.15.2), a different version from llama.cpp's GGML (0.20). Rather than
recompile the whole engine, the transcribe build renames every symbol
its static library shares with the rest of the binary — C `ggml*`/`gguf*`
functions *and* the mangled C++ symbols they both define
(`scripts/transcribe-rename-map.sh`, applied with `objcopy --redefine-syms`)
— so the binary carries one private, isolated copy for the transcribe
engine. Consequences:

- no symbol collisions with the shared GGML;
- the llamafile GPU loaders register with llama.cpp's GGML, so the
  transcribe engine stays CPU-only in this build.

### Dispatch model

Backend entry points are renamed at compile time (`-Dmain=<name>`),
requiring **zero changes** to the llamafile source tree:

| Source file | Compiled as |
|---|---|
| `llamafile/llamafile/main.cpp` (via `llama_dispatch.cpp`) | `llama_dispatch()` |
| `whisperfile/whisperfile.cpp` | `-Dmain=whisperfile_main` |
| `diffusionfile/diffusionfile.cpp` | `-Dmain=diffusionfile_main` |
| `transcribefile/main.cpp` | `-Dmain=transcribefile_main` |

`argv[1]` is consumed by the dispatcher; the remainder passes verbatim to
the chosen backend.

---

## Custom Build Options

### build-time engine toggles:
make MODE=rel CHIMERAFILE_WITH_WHISPER=0               # exclude whisper
make MODE=rel CHIMERAFILE_WITH_DIFFUSION=0              # exclude diffusion
make MODE=rel CHIMERAFILE_WITH_TRANSCRIBE=0             # exclude transcribe
make MODE=rel CHIMERAFILE_WITH_WHISPER=0 CHIMERAFILE_WITH_DIFFUSION=0 CHIMERAFILE_WITH_TRANSCRIBE=0  # llama only

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
- **v0.2** — diffusion image generation (custom stable-diffusion.cpp backend, GPU compat shim)
- **v0.3** — aligned with llamafile v0.10.6: upstream diffusionfile pattern, transcribe engine, CI builds the pinned submodule

## License

This project is provided under the GNU General Public License version 3 (GPLv3), and does not include code sourced directly from Mozilla's llamafile, nor is this project associated with such.

## AI-Usage Disclosure

Unfortunately, I was but the project manager, bullying Deepseek v4 into complying with my less than reasonable demands. I couldn't write code to save my life, but I hope it is of some use to you :)
