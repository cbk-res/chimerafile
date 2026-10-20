"""Model residency: multi-model loading, per-model KV isolation, TTL unload.

Run with the dedicated two-model config.  The default (tiny, CPU) config
costs nothing, so the live service can stay up; a GPU config needs the
GPU to itself and skips while the live service is running:

    pytest tests/test_residency.py --serve-config tests/configs/two-models.json

    # with a GPU config instead:
    systemctl --user stop chimerafile-serve
    pytest tests/test_residency.py --serve-config my-two-model-gpu.json
    systemctl --user start chimerafile-serve
"""

import time

import pytest

pytestmark = pytest.mark.gpu_exclusive


def need_two_models(server):
    if len(server.models) < 2:
        pytest.skip("residency tests need the two-model config "
                    "(tests/configs/two-models.json)")


def resident(server, model_id):
    s, body = server.client.get("/v1/status")
    assert s == 200, f"/v1/status: status={s} body={str(body)[:200]}"
    for r in body.get("resident", []):
        if r.get("id") == model_id:
            return r
    pytest.fail(f"{model_id} missing from /v1/status: {body}")


def test_models_listing(server):
    need_two_models(server)
    s, body = server.client.models()
    assert s == 200, f"status={s} body={str(body)[:200]}"
    listed = {m["id"]: m.get("engine") for m in body.get("data", [])}
    for mid in server.models:
        assert listed.get(mid) == "llama", f"{mid}: {listed}"


def test_both_models_load_and_serve(server):
    need_two_models(server)
    c = server.client
    for mid in server.models:
        s, r = c.chat({
            "model": mid,
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 32,
        })
        assert s == 200, f"{mid}: status={s} body={str(r)[:200]}"
        msg = ((r.get("choices") or [{}])[0].get("message", {}))
        assert isinstance(msg.get("content"), str), str(r)[:200]
    s, body = c.get("/v1/status")
    states = {r["id"]: r["state"] for r in body.get("resident", [])}
    for mid in server.models:
        assert states.get(mid) == "loaded", f"{mid}: {states}"


def test_interleaved_requests_keep_separate_caches(server):
    # Warm model A's cache, use model B in between, then repeat A's prompt:
    # A's KV must have survived the interleave (per-model caches).
    need_two_models(server)
    a, b = server.models[0], server.models[1]
    c = server.client
    prefix = ("the lighthouse keeper wound the great lamp every evening "
              "at dusk. ") * 80
    p_a = {
        "model": a,
        "messages": [{"role": "user",
                      "content": prefix + " What time is it? One word."}],
        "max_tokens": 32,
    }
    s, _ = c.chat(p_a)
    assert s == 200
    s, _ = c.chat({"model": b,
                   "messages": [{"role": "user", "content": "Say hi."}],
                   "max_tokens": 16})
    assert s == 200
    s, r = c.chat(p_a)
    assert s == 200
    t = r["timings"]["prompt_ms"]
    assert t < 0.05, \
        f"{a}'s cache did not survive the interleave: {t:.4f}s"


def test_ttl_unload_and_reload(server):
    candidates = [(mid, e) for mid, e in server.cfg.get("models", {}).items()
                  if e.get("engine") == "llama" and e.get("ttl", 0) > 0]
    if not candidates:
        pytest.skip("no model with a positive ttl in the config")
    mid, entry = min(candidates, key=lambda kv: kv[1]["ttl"])
    ttl = entry["ttl"]
    c = server.client

    s, _ = c.chat({"model": mid,
                   "messages": [{"role": "user", "content": "Say ok."}],
                   "max_tokens": 16})
    assert s == 200, f"status={s}"
    assert resident(server, mid)["state"] == "loaded"

    # ttl elapses, then the next sweeper cycle unloads.
    time.sleep(ttl + 5)
    state = resident(server, mid)
    assert state["state"] == "unloaded", f"ttl={ttl}s but still {state}"

    s, r = c.chat({"model": mid,
                   "messages": [{"role": "user", "content": "Say ok."}],
                   "max_tokens": 16})
    assert s == 200, f"reload failed: status={s} body={str(r)[:200]}"
    assert resident(server, mid)["state"] == "loaded"
