# serve tests

A pytest suite for the **serve integration layer** — the part chimerafile
owns: the OpenAI surface, model residency, queueing, and per-model KV
retention.  Engine-level behaviour (sampling, chat templates, PEG,
quantisations) is covered by llama.cpp's own test suite against the pinned
submodule; this suite deliberately does not re-test it.

## Layout

| File | What it checks |
| --- | --- |
| `test_compat.py` | The OpenAI surface against whatever model the config points at: chat, streaming, `max_tokens`, thinking split, tools (auto + forced `required`), the error matrix, overflow-with-tools (the load-crash regression), Bearer auth. Doubles as the model/backend acceptance check. |
| `test_kv_retention.py` | Per-model KV cache via `timings.prompt_processing_s`: identical prompt decodes nothing, shared prefix decodes only the tail, divergent prefix colds the cache. |
| `test_queueing.py` | The pump's serialisation under load: concurrent chats all succeed, `/health` stays responsive in flight, stream + non-stream interleave, recovery after a burst. |
| `test_residency.py` | Multi-model: both models load and serve, per-model caches survive interleaving, TTL unload + on-demand reload (via `GET /v1/status`). |

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
on the GPU and can happen while the live service is up:

```sh
pytest tests/
```

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

`two-models.json` loads the same model file twice, so its entries differ
in `args` (and `ctx`): the server keys a physical model by file + args,
and a distinct args list is what forces two physical loads.

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
