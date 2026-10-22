# serve tests

A pytest suite for the **serve integration layer** — the part chimerafile
owns: the OpenAI surface (chat and media), model residency, queueing,
and per-model KV retention.  Engine-level behaviour (sampling, chat
templates, PEG, quantisations) is covered by llama.cpp's own test suite
against the pinned submodule; this suite deliberately does not re-test it.

## Layout

| File | What it checks |
| --- | --- |
| `test_compat.py` | The OpenAI surface against whatever model the config points at: chat, streaming, `max_tokens`, thinking split, tools (auto + forced `required`), the error matrix, overflow-with-tools (the load-crash regression), Bearer auth. Doubles as the model/backend acceptance check. |
| `test_tools_nested.py` | Tool schemas with a doubly-nested array of objects (the `question` tool shape): the cosmo build used to crash the model child on them (SIGSEGV, the request died). Requests carrying such a schema must complete repeatedly and a plain follow-up must survive; a flat array of scalars must complete and must not be dropped. The last test prints which fix mode the build is in ("works", or "dropped" - the child logs one warning per dropped tool). |
| `test_kv_retention.py` | Per-model KV cache via the child's `timings.prompt_ms`: identical prompt decodes nothing, shared prefix decodes only the tail, divergent prefix pays a full prefill. Every test uses sentences no other test uses (the child's KV cache is content-addressed and never clears). |
| `test_queueing.py` | Load: concurrent chats all succeed, `/health` stays responsive in flight, stream + non-stream interleave, recovery after a burst. |
| `test_residency.py` | Multi-model: both models load and serve, per-model caches survive interleaving, TTL unload + on-demand reload (via `GET /v1/status`). |
| `test_gizmo_lifecycle.py` | The process tree, on a fresh port per test: spawn failure, crash self-heal, TTL release, orphans after a group kill, after a graceful stop, and after a bare hard kill of serve (the parent-death watcher), disconnect cancellation, alias routing, load coalescing, capacity eviction, pinned-fleet 503 — the GPU lease deferring a llama load while the media child is on the GPU (`gpu_exclusive`), and the media child's own lifecycle (crash self-heal, TTL unload + reload, group kill with a media config). |
| `test_media_audio.py` | The Whisper engine via `/v1/audio/transcriptions`: a real transcription of an espeak-ng–synthesised clip, the `json` / `verbose_json` / `text` formats, and the error matrix. |
| `test_media_images.py` | The diffusion engine via `/v1/images/generations` + `/v1/images/edits`: small CPU-friendly generations (256×256, 8 steps) return a valid PNG of the requested size, plus the error matrix. |
| `test_media_sdcpp.py` | The sdcpp mirror of stable-diffusion.cpp's server API: `/sdcpp/v1/capabilities`, async `img_gen` jobs (txt2img, img2img, hires upscale, cancel of a queued job), and the error matrix (including the `mask_image` rejection). Uses the tiny-media-sdcpp config. Note: img2img (init-image) jobs can crash the media child (engine stack-overflow during VAE encode — [quirks.md](../quirks.md)); a dead child on an img2img job is that limitation, not a serve-layer regression. |
| `test_media_transcribe.py` | The transcribe.cpp engine on the same `/v1/audio/transcriptions` surface as whisper: a real transcription of an espeak-ng clip (voice picked for the model — [quirks.md](../quirks.md)) with a canary-180m model, `json` + `verbose_json`, and the 400 for a non-speech model. Uses the tiny-media-sdcpp config. |
| `test_child_kv.py` | The KV retention scenarios again, but against a raw `llama --server` child with no serve layer in between, asserting on upstream's exact `timings.cache_n` counter — the check that warm continuation is inherited from upstream. |

Configs live in `tests/configs/`.  Each is a normal serve config; a run
points at one of them, so "a different model/backend" is a different
config (plus `--gpu` for the spawned server).  Relative paths in a config
(model `file`, `--chat-template-file`) resolve against the repo root.

The optional top-level `test_profile` key (ignored by the server) marks a
config's model.  `"tiny"` means weak weights: the plumbing is still
tested, but model-quality expectations drop out — the auto tool-call test
skips and the forced `required` test degrades to a mechanism check
(grammar builds and the request completes; the model itself will not call
a tool of its own volition).

## Running

The default config is the repo's **tiny test model** (TinyLLama-5M from
the llamafile submodule, 10 MB) on the CPU, so a full run costs nothing
on the GPU and can happen while the live service is up.  The spawned
server always gets an explicit `--gpu`: the default (`cpu`) maps to the
binary's `off`, so no GPU backend loads at all — without the flag the APE
auto-detects, and the media engines (which have no layer knob) would land
on the GPU.

```sh
pytest tests/
```

**The full matrix** — the default suite, the media, residency, and
raw-child suites, in one go (`make test` runs the same thing):

```sh
scripts/run-test-matrix.sh               # the CPU lane (safe while the
                                         # live service is up)
GPU=vulkan scripts/run-test-matrix.sh    # + the GPU lane (full suite with
                                         # --gpu vulkan; the gpu_exclusive
                                         # tests skip while the live service
                                         # holds the GPU)
```

Each suite spawns its own server on a free port; the script exits
non-zero when any suite fails.  `PYTEST=/path/to/pytest` overrides the
interpreter.

**Your own model / quant / backend** is a config file (plus `--gpu` for
the spawned server):

```sh
pytest tests/ --serve-config my-model.json --gpu vulkan
pytest tests/ --reuse --serve-config my-live-service.json
```

**Residency** uses the dedicated two-model config.  The tiny one costs
nothing, so the live service can stay up:

```sh
pytest tests/test_residency.py --serve-config tests/configs/two-models.json
```

(A GPU two-model config still needs the GPU to itself; while the live
service is listening, residency skips itself and says so.)

`two-models.json` loads the same model file twice: the router gives each
model id its own child, so one file can back several ids with different
`args` and `ctx` — each id is its own model, with its own KV cache.

**Media** uses the tiny-media config (tiny llama + whisper + diffusion):

```sh
pytest tests/test_media_audio.py tests/test_media_images.py \
    --serve-config tests/configs/tiny-media.json
```

The sdcpp and transcribe suites use the tiny-media-sdcpp config (tiny
llama + whisper + diffusion + a transcribe model, with `"sdcpp"` enabled
on the diffusion model):

```sh
pytest tests/test_media_sdcpp.py tests/test_media_transcribe.py \
    --serve-config tests/configs/tiny-media-sdcpp.json
```

The media model files live in `tests/fixtures/` (gitignored — they are
hundreds of MB): a whisper `.bin`, an SD 1.5 GGUF and a
canary-180m-flash GGUF.  The success-path tests need them; without them
(or against a config with no media entries) the module skips.  The one
committed exception is the `upscalers/` placeholder: a labeled 67-byte
file that `test_capabilities` asserts on through the capabilities
listing (it is never loaded).

The GPU lease (model loads serialised across the two children — see the
top-level README's "How it runs") is a no-op in the CPU lane: with
`--gpu off` the gates never fire.  It is covered by
`test_gizmo_lifecycle.py::test_gpu_lease_defers_llama_load`
(`gpu_exclusive`), which runs in the GPU lane (`--gpu vulkan`) with the
media sweeper's cadence shortened to 1 s (`CHIMERAFILE_MEDIA_SWEEP_S`) so
the gate is observed inside the whisper model's ~3 s load window.

**Raw child** uses the config key `test_spawn` to spawn a bare
`llama --server` child (the first llama entry's file and args) instead
of the serve layer:

```sh
pytest tests/test_child_kv.py --serve-config tests/configs/tiny-llama-child.json
```

Useful options and environment:

- `--keep-server` — leave a spawned server running afterwards
- `CHIMERAFILE_TEST_MODEL=/path/to.gguf` — try another model file without
  editing the config (single-model configs only)
- `CHIMERAFILE_BIN` — path to the APE (default: the repo build)
- `CHIMERAFILE_LIVE_PORT` — port probed to decide the `gpu_exclusive`
  skip (default 18080)
- `pytest tests/test_compat.py::test_tools_required_is_enforced -v` — one test

A spawned server's log lands in `/tmp/opencode/chimera-pytest-<port>.log`.

## What this is not

- Not an engine test suite (llama.cpp's covers that at the pinned commit).
- Not a model-quality benchmark: the behavioural grading of tool calling
  (right tool, right arguments, forced choice) lives in
  `scripts/serve-tools-behaviour.py`, which stays standalone on purpose.
