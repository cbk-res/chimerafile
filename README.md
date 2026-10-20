# Chimerafile

[![build](https://github.com/cbk-res/chimerafile/actions/workflows/build.yml/badge.svg)](https://github.com/cbk-res/chimerafile/actions/workflows/build.yml)

A unified inference binary that combines **llamafile** (LLM), **whisperfile** (speech-to-text), **diffusionfile** (stable-diffusion.cpp image/video generation), and **transcribefile** (transcribe.cpp speech-to-text) into a single [APE](https://justine.lol/ape.html) executable, while making zero changes to the upstream source.

```
./chimerafile llama      [llamafile args...]
./chimerafile whisper    [whisperfile args...]
./chimerafile diffusion  [diffusionfile args...]
./chimerafile transcribe [transcribefile args...]
```

All engines share one **llama.cpp GGML runtime** (whisper, stable-diffusion.cpp and transcribe.cpp are all compiled against it), reducing duplication for projects that would otherwise bundle several standalone 'llamafiles'. GPU backends (Vulkan, CUDA, ROCm, Metal) are bundled as `.so` files inside the APE and loaded at runtime.

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

---

## Serve — OpenAI-compatible server

All four engines behind one port, driven by a declarative config file:

```sh
./chimerafile serve --config config.json
```

### Config

```json
{
  "listen": "127.0.0.1:8080",
  "key": "optional-bearer-token",
  "sdcpp": {
    "model": "artist",
    "upscalers_dir": "upscalers"
  },
  "models": {
    "assistant": {
      "engine": "llama", "file": "gemma.gguf",
      "ctx": 4096, "gpu_layers": -1,
      "args": ["--temp", "0.7"]
    },
    "scribe":    { "engine": "whisper", "file": "ggml-tiny.bin" },
    "subtitles": { "engine": "transcribe", "file": "qwen3-asr.gguf" },
    "artist":    { "engine": "diffusion", "file": "sd15.gguf", "ttl": 0 }
  }
}
```

- `listen` — `host:port`. The port must be free; the server refuses to start if it is taken.
- `key` — optional. When set, every request needs `Authorization: Bearer <key>` (401 without it).
- `engine` — `llama`, `whisper`, `diffusion` or `transcribe`. `file` is resolved relative to the config file's directory.
- `ctx`, `gpu_layers` — llama only (defaults 4096, `-1` = all layers on GPU).
- `ttl` — seconds to stay in RAM when idle. Omitted = engine default (30 min llama, 5 min everything else); `0` = never.
- `pin` — never unload, never evict.
- `args` — CLI flags passed through verbatim, **llama only**: they land in the model child's own `llama-server` command line (`--temp`, `--ctk`, `--no-mmap`, `--flash-attn on`, `--cache-type-k q8_0`, ...), so the full llama-server option set is available, including server-tool-only flags like `--cache-prompt`. The other engines have no flag passthrough; their behaviour is steered by the request fields instead.
- `sdcpp` — optional; the block's presence enables the `/sdcpp/v1/*` surface (see Endpoints). `model` — the diffusion model it serves; defaults to the first diffusion model in the config, and must name a diffusion model. `upscalers_dir` — optional directory of extra hires upscaler models (`.gguf`, `.safetensors`, `.pt`, `.pth`), resolved relative to the config file's directory.

### Endpoints

| Endpoint | Engine | Notes |
|---|---|---|
| `GET /health` | — | liveness |
| `GET /v1/models` | — | the configured model ids |
| `POST /v1/chat/completions` | llama | non-stream and SSE streaming; `temperature`, `top_p`, `max_tokens`, `stop`; `tools`, `tool_choice` (string and object forms), `parallel_tool_calls`; `n > 1` is rejected |
| `POST /v1/audio/transcriptions` | whisper, transcribe | multipart `file` + `model`; `response_format` = `json` (default) \| `text` \| `srt` \| `vtt` \| `verbose_json`; `language` |
| `POST /v1/images/generations` | diffusion | JSON; see extensions below |
| `POST /v1/images/edits` | diffusion | multipart `image` + `prompt` (img2img); optional `size`, defaults to the input image's |
| `POST /sdcpp/v1/img_gen` | diffusion | sdcpp surface (config-gated): async image job; 202 with `poll_url` |
| `GET /sdcpp/v1/jobs/<id>` | diffusion | job status/result; 410 once expired |
| `POST /sdcpp/v1/jobs/<id>/cancel` | diffusion | cancel a queued job (409 while generating) |
| `GET /sdcpp/v1/capabilities` | diffusion | model, modes, defaults, limits, samplers, upscalers |
| `GET /v1/status` | — | what is loaded, what is queued, uptime |
| `POST /v1/residency/<id>` | — | preload / keep warm (starts the TTL clock) |
| `POST /v1/residency/<id>/unload` | — | explicit unload (409 while in use) |
| `GET /metrics` | — | Prometheus text format: request/token/second counters, derived average t/s, per-model residency |

### Metrics and timing

`GET /metrics` returns a Prometheus text-format document, like llama-server's: request counters per surface (`chimerafile_chat_requests_total`, `chimerafile_transcribe_requests_total`, `chimerafile_image_requests_total`), prompt and generation token totals, prompt-processing and generation seconds, the `chimerafile_tokens_per_second` average derived from them, uptime, and per-model residency gauges (`chimerafile_model_state`, `chimerafile_model_in_use`).

Chat completions carry the model child's own per-request `timings` object (the upstream llama-server shape — `prompt_ms`, `predicted_ms`, `per_second`, ...) as a top-level field of the JSON response and in the final SSE chunk of a stream; clients that do not look for it ignore it.

### How it runs

- Llama models run behind an upstream router child: serve spawns one `llama-server` router per port (a generated `--models-preset` names the fleet) and forwards the llama API to it. The router owns model spawning, loading, queueing and crash respawns; serve keeps the front half — admission against `max_loaded_models`, TTL eviction, pins, and the status merge. The first request for a model waits for its load, and concurrent requests for a full fleet get a retryable 503.
- Media engines (whisper, diffusion, transcribe) run in a media child process: a config with media models makes serve spawn a second child (`chimerafile media`) and forward `/v1/audio/*`, `/v1/images/*` and (when `sdcpp` is enabled) `/sdcpp/*` to it. The child keeps the single-pump discipline of the in-process days — loads, inference and unloads run on its main thread, because in this APE only the thread that dlopen'd the host engine libraries has a working host-glibc TLS state — so the first request for a model waits for its load and concurrent media jobs queue behind it; interactive work (transcription) takes a fast lane ahead of queued image generations. A crashed media child is respawned on the next request (bounded: five in any five-minute window) and that request is retried once.
- GPU lease: with a GPU in play, model loads are serialised across the two children — while one child is loading a model, the other defers starting its own loads with a retryable 503 `model_loading` (`Retry-After: 5`). The other side's state is polled (the router snapshot every 1 s, the media child's `/v1/status` every 5 s), so the lease is coarse and best-effort, with a false-positive window of up to one sweep; with `--gpu off` it is a complete no-op.
- No orphans: the APE runtime swallows PDEATHSIG, so every child carries its own parent-death watcher instead — when serve dies (hard kill, crash), each child notices within about a second and shuts down through its own SIGTERM path, and the tree cascades: the router takes its model children down with it. A launcher that kills the whole process group (systemd `KillMode=control-group`, job control) takes everything down immediately.
- KV retention is the child's own: it keeps prompt tokens in its KV cache between requests (unified cache, matched by content), so a prompt that repeats a previous prefix only prefills the new tail (clients like opencode resend the whole conversation each turn); the cache never clears on its own, and each model has its own child and its own cache.
- Diffusion models get a fresh engine context per request: the pinned stable-diffusion.cpp cannot run a second generation on the same context, so each image request pays a few seconds of model load.
- Streaming caveat: `stop` strings are checked by llama.cpp at token boundaries, so a stream can overshoot a stop string by the rest of the token the match landed inside.
- Tool calling is the child's: the template renders the tool schemas into the prompt and carries a PEG parser for its tool-call markup, so replies come back as structured `tool_calls` (in non-streaming responses and as streaming deltas alike) and the raw markup never leaks into `content`. The serve side only validates the request shape before forwarding (`tools` must be an array, `tool_choice` a string or an object of type `none`/`function`).
- A prompt that does not fit the model's context — alone, or together with `max_tokens` — is a client error: 400 `context_length_exceeded` before the reply starts. In streaming mode the 400 arrives before the first chunk, not mid-stream.
- `temperature` and `prompt` on `/v1/audio/transcriptions` are whisper-only: whisper.cpp has both, transcribe.cpp has neither, and the server passes down only what the engine supports.
- Image generation takes the OpenAI fields `prompt`, `size` (`WxH`), `n` (1–4), `seed`, `response_format` (`b64_json` default | `url`, returned as a `data:` URL) plus the chimerafile extensions `negative_prompt`, `steps` and `cfg_scale` — all optional.
- The `sdcpp` block mirrors the native API of stable-diffusion.cpp's own server (`/sdcpp/v1/*`) so clients pointed at an sd.cpp server work unchanged: `img_gen` jobs (txt2img, img2img via `init_image` + `strength`, hires upscale, batches of up to 8) with the upstream's async poll/cancel semantics, request field names and response shapes. What the mirror does not expose: video, control images, reference images, loras, vae tiling, and `mask_image` (rejected with a 400 — the pinned engine's mask path corrupts memory in this APE, with or without the serve layer).

An end-to-end smoke test of the whole surface lives at `scripts/serve-smoke-test.py` (standard library only):

```sh
python3 scripts/serve-smoke-test.py \
    --bin llamafile/o/rel/chimerafile/chimerafile \
    --llama gemma.gguf --whisper ggml-tiny.bin \
    --transcribe qwen3-asr.gguf --diffusion sd15.gguf --audio jfk.wav
```

A tools-focused companion test, `scripts/serve-tools-test.py`, spawns the
server from a serve config with one llama model (or reuses one already
listening) and checks the tool path: `tool_choice=none`, structured
extraction, a tool-result round trip, streaming deltas and request-shape
validation:

```sh
python3 scripts/serve-tools-test.py --config config.json
```

---

## GPU Backends

GPU acceleration for all engines (llama, whisper, diffusion, transcribe) is
automatic — no flags needed. Use `--gpu` to override:

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

ROCm builds are TinyBLAS-only: `rocm.sh` passes `-DGGML_USE_TINYBLAS=1`, and
`make setup` applies `ci-patches/rocm-tinyblas.patch` to llama.cpp so that path
is actually self-contained — no hipBLAS installation needed, and the resulting
`ggml-rocm.so` has no `libhipblas` dependency.

Each produces a `.so` that gets zipaligned into the APE automatically.

**Vulkan gotcha:** the shader compiler `glslc` must be on your `PATH` (or set
`GLSLC=/path/to/glslc`) — with neither available, `vulkan.sh` fails *silently*
and the build continues with a CPU-only APE. If you installed the SDK to a
non-standard prefix, also export `VULKAN_SDK=/prefix/usr` so the SPIR-V
headers are found (e.g. from unpacked F44 RPMs:
`VULKAN_SDK=/tmp/vulksdk/root/usr GLSLC=/tmp/vulksdk/root/usr/bin/glslc`).

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

### Diffusion (Vulkan) segfault at image save — fixed in the APE

A `--gpu vulkan` diffusion run used to complete generation ("generate_image
completed") then segfault in the host loader's `_dl_fixup` while writing the
PNG. Cause: `cosmo_dlopen` runs the host `ld-linux` in-process, and the host
glibc initialises its **brk-based** heap from the process brk — which sits at
the base of this APE's own reserved heap arena. Under sustained allocation
(the 2.7 GB SD model + Vulkan driver) the two heap managers interleave in the
same region and corrupt each other, destroying the loader's `link_map`. The
dispatcher now calls the host `mallopt(M_MMAP_THRESHOLD, 0)` right after a GPU
backend loads, routing every host-glibc allocation through `mmap` and keeping
it out of the shared brk region. Nothing to do — it's built in.


---

## Project Structure

```
chimerafile/                    ← this repo (chimerafile layer only)
├── chimerafile.cpp             ← dispatcher (owns main())
├── llama_dispatch.cpp          ← llamafile mode dispatch
├── serve.cpp                   ← serve layer: OpenAI surface, routing to the children
├── serve_common.h              ← shared serve types (config, model entries)
├── serve_config.cpp            ← config parsing (shared with the media child)
├── serve_forward.{h,cpp}       ← the child forwarders: router (llama) + media
├── serve_media.{h,cpp}         ← the media child: whisper / diffusion / transcribe
├── BUILD.mk                    ← build rules (injected into llamafile's make)
├── Makefile                    ← thin wrapper, delegates to llamafile/Makefile
├── ci-patches/                 ← builder patches (Pascal sm_61; ROCm TinyBLAS; router child spawn)
├── setup-baseline/             ← content hashes of the patched submodule trees
├── .github/workflows/build.yml ← CI: 6 variant builds
└── llamafile/                  ← git submodule (pinned v0.10.6, patched in place by make setup)
    ├── llamafile/              ← LLM TUI + server
    ├── whisperfile/            ← speech-to-text CLI wrapper
    ├── diffusionfile/          ← image generation CLI wrapper
    ├── transcribefile/         ← transcribe.cpp CLI wrapper
    ├── llama.cpp/              ← LLM backend (nested submodule)
    ├── whisper.cpp/            ← ASR backend (nested submodule)
    ├── stable-diffusion.cpp/   ← image generation library (nested submodule)
    └── transcribe.cpp/         ← ASR backend (nested submodule)
```

### Single GGML

All four engines share **llama.cpp's GGML** — one GPU-capable runtime
(Vulkan / CUDA / ROCm / Metal) for everything:

- `whisperfile`: whisper.cpp's core/common sources are recompiled against
  llama.cpp's GGML headers.
- `diffusionfile`: stable-diffusion.cpp is built by the upstream
  diffusionfile package against the same GGML.
- `transcribefile`: transcribe.cpp vendors its own GGML (ggml-org/ggml
  0.15.2), a different version from llama.cpp's (0.20), so chimerafile
  builds transcribe's library sources directly against llama.cpp's GGML
  (`BUILD.mk`) instead of using the upstream `transcribe.cpp.a` (which
  bundles the private copy). The llamafile submodule stays untouched —
  its transcribe build is never run or linked — and the transcribe
  engine uses the same bundled GPU backends as the rest.

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

### Submodule state — baselines and `make fresh`

The llamafile submodule is pinned to an upstream commit and `make setup`
patches its nested submodules **in place** (the `.patches` apply as
working-tree changes), so a correctly set up tree is *dirty by design*.
`setup-baseline/` records exactly what that dirt looks like — content-hashed
per file — and every build first runs `scripts/verify-setup.sh`, which fails
loudly on any drift: stray edits (even to files setup already patches), stray
files, or moved checkouts.

```sh
make verify-setup   # check the submodule state only
make fresh          # reset trees to the pins, re-apply setup, wipe o/, full rebuild
```

`make setup` and `make fresh` re-record the baselines automatically. If you
deliberately change the pinned state (new llamafile commit, new patches),
re-run setup and commit the updated `setup-baseline/`.

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
- **v0.2.99** — diffusion Vulkan save-crash fix (host glibc heap routed via mmap); transcribe ported onto the shared GGML — all four engines GPU-capable from one backend
- **v0.3** — aligned with llamafile v0.10.6: upstream diffusionfile pattern, transcribe engine, CI builds the pinned submodule
- **v0.4** — `chimerafile serve`: OpenAI-compatible server over all four engines (declarative config, TTL residency, serial execution)
- **v0.5** — llama path moved behind an upstream router child (one router per serve, front-side admission, TTL/pins kept); media engines still in-process
- **v0.6** — media engines moved behind a media child process (serve is now a thin router over the router child and the media child); GPU lease serialises model loads across the two children

## License

This project is provided under the GNU General Public License version 3 (GPLv3), and does not include code sourced directly from Mozilla's llamafile, nor is this project associated with such.

## AI-Usage Disclosure

Chimerafile is highly experimental and largely a proof of concept. Substantial portions of the codebase have been written with the assistance of large language models (including Deepseek V4 and Qwen 3.8), and as such, bugs or instabilities should be expected for the moment. If you have any problems, please feel free to open an issue!
