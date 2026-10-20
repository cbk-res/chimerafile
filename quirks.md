# Quirks and known limitations

Chimerafile runs four upstream engines inside one APE. Most of the time that
is invisible; this page collects the places where it is not — engine quirks,
APE constraints, and the workarounds already baked into the binary.

## Engines

### Each image generation gets a fresh engine context

The pinned stable-diffusion.cpp cannot run a second generation on the same
context, so every image request (`/v1/images/generations` or an sdcpp
`img_gen`) pays a few seconds of model load. That is an upstream limitation,
not a serve-layer choice.

### `mask_image` is rejected on the sdcpp surface

`POST /sdcpp/v1/img_gen` with `mask_image` returns 400. The pinned engine's
mask path corrupts memory in this APE — with or without the serve layer.
Every other field works.

### What the sdcpp mirror does not expose

Video generation, control images, reference images, loras and VAE tiling are
not part of the mirror (and `mask_image` is rejected, above). They exist in
upstream stable-diffusion.cpp's server; the mirror covers the img_gen
workflow only.

### img2img can crash the media child

img2img (init-image) jobs can overflow the media child's 80 KB guest thread
stack in the VAE-encode path and crash the child; for the test-suite inputs
the overflow is deterministic. The serve layer respawns the child on the next
request, so a dead child surfaces as a 502 on the in-flight job, not as a
dead service. This is an engine-side limitation of the pinned
stable-diffusion.cpp on this APE (pass-through-plan §14.10); the img2img test
is marked xfail on purpose.

### Oversized chat prompts are rejected before they are forwarded

The pinned router child can, in rare cases, crash (segfault) while
tokenising a very large prompt instead of returning its normal 400. One bad
request must not be able to take the child — and every in-flight chat — down,
so serve checks first: message content plus `tools` beyond `4 × ctx` bytes
cannot possibly fit (a token is 1–4 bytes of UTF-8), and such requests get a
400 `context_length_exceeded` without reaching the child. No fitting prompt
can be rejected: anything that fits is at most `4 × ctx` bytes.

Residual window: the child's effective context is capped at the model's
training context, which can sit below the configured `ctx`. Prompts between
the two can still reach the child and still carry the rare crash risk.
Closing the window needs the child's effective `n_ctx`, which the status
sweep already reports — deferred (pass-through-plan §14.11).

### Stop strings can overshoot by a token

`stop` strings are checked by llama.cpp at token boundaries, so a stream can
run past a stop string by the rest of the token the match landed inside.
Inherited from llama.cpp.

### `temperature` and `prompt` on `/v1/audio/transcriptions` are whisper-only

whisper.cpp has both fields, transcribe.cpp has neither, and the server
passes down only what the target engine supports.

## Serve

### The GPU lease is coarse

The lease defers a child's model loads while the *other* child is loading a
model, but each side learns about the other by polling (the router snapshot
every 1 s, the media child's `/v1/status` every 5 s) rather than by a
handshake. A load can therefore be deferred with a 503 `model_loading` for
up to one sweep even when the GPU is already free — the request is retried
and goes through. Best-effort by design; with `--gpu off` the lease does
nothing at all.

### Thinking models can come back with an empty `content`

A thinking model spends `max_tokens` on reasoning first. When the budget
runs out inside the thinking, the reply is a 200 with an empty `content`
and the reasoning itself in the message's `reasoning_content` field. In the
v0.4 gauntlet, six of twelve turns came back that way with `max_tokens=320`;
the same question answered on the first try once the budget allowed for
thinking (mellum can burn several thousand tokens of it). Budget
`max_tokens` for thinking plus answer, and read `reasoning_content` if you
care where the budget went.

## GPU and memory budgets

### Fixed-VRAM GPUs have no spill

The memory budget for a serve is the GPU's *local* memory: there is no
fallback to system RAM. A mellum-class model at 131k context (Q5_K_M weights
plus the full KV cache) sits at about 7.1G resident, and flash-attn working
buffers on top of that. On an 8G eGPU (Vega 64) that is within reach of the
wall: a 96k-token prefill wedged the card (a child died mid-tensor-upload in
`ggml_vk_wait_for_fence` — fence error, uncaught exception, core dump — and
the eGPU dropped off the bus until a reboot; the in-flight request got a
clean 500, the service itself stayed up). The child logs were lost in the
reboot, so the exact failing allocation is unconfirmed, but the budget math
points at the 8G wall, and no media model fits alongside such a model on 8G
in any case. If a serve targets a fixed-VRAM card, size the config so
weights + KV + the largest concurrent generation fit inside it.

### iGPUs with shared memory are the easy case

On an iGPU (e.g. AMD 780M) the "VRAM" is system RAM, so the budget is the
machine's RAM (24G here) and mellum at 131k plus every media model fits
comfortably. This is the intended deployment shape.

## APE and builds

### A Vulkan build without `glslc` is silently CPU-only

`vulkan.sh` needs the shader compiler `glslc` on the `PATH` (or
`GLSLC=/path/to/glslc`); with neither available it fails *silently* and the
build continues, producing a CPU-only APE. If the SDK is installed at a
non-standard prefix, also export `VULKAN_SDK=/prefix/usr` so the SPIR-V
headers are found (e.g. from unpacked F44 RPMs:
`VULKAN_SDK=/tmp/vulksdk/root/usr GLSLC=/tmp/vulksdk/root/usr/bin/glslc`).

### "Vulkan is not usable" on SELinux Fedora

On SELinux-enforcing Fedora you may see:

```
fatal error: --gpu vulkan was explicitly requested but Vulkan is not usable
on this system
```

even though `vulkaninfo` lists the GPU. This is a host policy issue, not a
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

### The diffusion Vulkan save crash is fixed in the binary

A `--gpu vulkan` diffusion run used to complete generation ("generate_image
completed") and then segfault in the host loader's `_dl_fixup` while writing
the PNG. Cause: `cosmo_dlopen` runs the host `ld-linux` in-process, and the
host glibc initialises its **brk-based** heap from the process brk — which
sits at the base of this APE's own reserved heap arena. Under sustained
allocation (the 2.7 GB SD model + Vulkan driver) the two heap managers
interleave in the same region and corrupt each other, destroying the
loader's `link_map`. The dispatcher now calls the host
`mallopt(M_MMAP_THRESHOLD, 0)` right after a GPU backend loads, routing every
host-glibc allocation through `mmap` and keeping it out of the shared brk
region. Nothing to do — it is built in.

## Testing

### The transcribe fixture is voice-sensitive

The canary-180m-flash model transcribes espeak-ng's default voice (en-gb,
175 wpm) to **empty** text, on every quant and backend tested; `en-us` at
120 wpm is reliable. The success-path tests therefore synthesise their
fixture with `espeak-ng -v en-us -s 120` (the comment in
`tests/test_media_transcribe.py` explains). A whisper turbo model transcribed
the default voice perfectly, so the sensitivity is the canary model's, not
the pipeline's.
