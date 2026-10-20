"""Protocol surface and the "does this model work" smoke.

Run against whatever model the selected config points at, so it doubles
as the model/backend acceptance check:

    pytest tests/test_compat.py
    pytest tests/test_compat.py --serve-config my-model.json --gpu vulkan

With the tiny test profile (the default config) the weights are random:
the plumbing is real, but only the forced-grammar tool test is meaningful
(a random model will not call a tool of its own volition, so the
auto-mode test skips).
"""

import json

import pytest

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
]

# Prompts tried in order when nudging the model into a tool call.
TOOL_PROMPTS = [
    "Call the get_weather tool for Paris.",
    "You must use the get_weather tool. What is the weather in Paris?",
    "Use your tools to look up the weather in Paris right now.",
]

MARKUP = ("<tool_call", "</tool_call", "to=functions", "<function=", "<<function")

SENT = "the quick brown fox jumps over the lazy dog. "


def big_message(n_bytes):
    text = (SENT * (n_bytes // len(SENT) + 1))[:n_bytes]
    return {"role": "user",
            "content": text + " What is one plus one? Reply in a word."}


def message_of(r):
    return ((r.get("choices") or [{}])[0].get("message", {})
            if isinstance(r, dict) else {})


def test_health(server):
    s, body = server.client.health()
    assert s == 200, f"status={s} body={body}"
    assert body.get("status") == "ok"


def test_models_lists_the_config_models(server):
    s, body = server.client.models()
    assert s == 200, f"status={s} body={body}"
    listed = {m["id"]: m.get("engine") for m in body.get("data", [])}
    for mid in server.models:
        assert mid in listed, f"{mid} missing from /v1/models: {listed}"
        assert listed[mid] == "llama"


def test_chat_nonstreaming(server):
    s, r = server.client.chat({
        "model": server.model,
        "messages": [{"role": "user",
                      "content": "What is the capital of France? One word."}],
        "max_tokens": 128,
    })
    assert s == 200, f"status={s} body={str(r)[:300]}"
    choice = r["choices"][0]
    msg = choice["message"]
    assert msg["role"] == "assistant"
    assert isinstance(msg.get("content"), str)
    assert (msg.get("content") or "").strip(), \
        f"empty reply: content={msg.get('content')!r} " \
        f"reasoning={(msg.get('reasoning_content') or '')[:80]!r}"
    assert choice["finish_reason"] in ("stop", "length", "tool_calls")
    usage = r["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == \
        usage["prompt_tokens"] + usage["completion_tokens"]
    # timings are the child's own (upstream llama-server) shape
    assert r["timings"]["prompt_ms"] >= 0
    assert r["timings"]["predicted_ms"] >= 0


def test_max_tokens_is_respected(server):
    s, r = server.client.chat({
        "model": server.model,
        "messages": [{"role": "user", "content": "Count from one to twenty."}],
        "max_tokens": 8,
    })
    assert s == 200, f"status={s}"
    assert r["usage"]["completion_tokens"] <= 8, \
        f"completion_tokens={r['usage']['completion_tokens']}"


def test_chat_streaming(server):
    chunks, done, raw, err = server.client.stream_chat({
        "model": server.model,
        "messages": [{"role": "user",
                      "content": "Say the word hello, and nothing else."}],
        "max_tokens": 128,
    })
    assert err is None and done, f"err={err} chunks={len(chunks)}"
    first = chunks[0]["choices"][0]["delta"]
    assert first.get("role") == "assistant"
    content = "".join(
        ch.get("delta", {}).get("content") or ""
        for c in chunks for ch in c.get("choices", []))
    reasoning = "".join(
        ch.get("delta", {}).get("reasoning_content") or ""
        for c in chunks for ch in c.get("choices", []))
    assert content.strip() or reasoning.strip(), \
        f"empty stream: content={content!r}"
    assert not any(m in content for m in MARKUP), content[:200]
    finishes = [ch.get("finish_reason")
                for c in chunks for ch in c.get("choices", [])
                if ch.get("finish_reason")]
    assert finishes and finishes[-1] in ("stop", "length", "tool_calls"), \
        f"finishes={finishes}"


def test_thinking_is_split_out(server):
    s, r = server.client.chat({
        "model": server.model,
        "messages": [{"role": "user",
                      "content": "Work out 17 * 24 and show your steps."}],
        "max_tokens": 512,
    })
    assert s == 200, f"status={s}"
    msg = message_of(r)
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    assert content.strip() or reasoning.strip(), \
        f"nothing came back: {msg!r}"
    # Whatever side the text landed on, it must be clean of markup.
    for m in (content, reasoning):
        assert not any(tag in m for tag in MARKUP), m[:200]


def test_tools_auto_structured(server):
    if server.profile == "tiny":
        pytest.skip("random-weight model: no spontaneous tool calls")
    c = server.client
    s, r = None, None
    for prompt in TOOL_PROMPTS:
        s, r = c.chat({
            "model": server.model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 512,
        })
        assert s == 200, f"status={s} body={str(r)[:200]}"
        tcs = message_of(r).get("tool_calls") or []
        if tcs:
            break
    tc = (message_of(r).get("tool_calls") or [{}])[0]
    assert tc, "no tool call after the nudge prompts"
    assert tc.get("id", "").startswith("call_"), tc.get("id")
    assert tc.get("type") == "function"
    fn = tc.get("function") or {}
    assert fn.get("name") == "get_weather", fn.get("name")
    args = json.loads(fn.get("arguments") or "{}")
    assert isinstance(args, dict) and args, args


def test_tools_required_is_enforced(server):
    # A math question: the model would answer in plain text, so the
    # grammar's job is to stop it ending the turn without a call.
    # On the tiny profile this is a mechanism check only (the grammar
    # must build and the request must complete); a random model rambles
    # to max_tokens rather than emitting a call.
    s, r = server.client.chat({
        "model": server.model,
        "messages": [{"role": "user",
                      "content": "What is 2+2? Answer in one word."}],
        "tools": TOOLS,
        "tool_choice": "required",
        "max_tokens": 512,
    })
    assert s == 200, f"status={s} body={str(r)[:200]}"
    msg = message_of(r)
    tcs = msg.get("tool_calls") or []
    finish = ((r.get("choices") or [{}])[0].get("finish_reason")
              if isinstance(r, dict) else None)
    if tcs:
        args = json.loads((tcs[0].get("function") or {})
                          .get("arguments") or "{}")
        assert isinstance(args, dict), args
    else:
        assert finish != "stop", \
            f"turn ended without a call: finish={finish} " \
            f"content={(msg.get('content') or '')[:120]!r}"


def test_error_matrix(server):
    c = server.client

    s, _ = c.raw("/v1/chat/completions", b"{not json")
    assert s == 400, f"bad JSON: status={s}"

    s, r = c.chat({"messages": [{"role": "user", "content": "hi"}]})
    assert s == 400, f"missing model: status={s} body={str(r)[:160]}"

    s, r = c.chat({"model": "no-such-model",
                   "messages": [{"role": "user", "content": "hi"}]})
    assert s == 404, f"unknown model: status={s}"
    assert isinstance(r, dict) \
        and r.get("error", {}).get("code") == "model_not_found", str(r)[:160]

    s, r = c.chat({"model": server.model,
                   "messages": [{"role": "user", "content": "hi"}],
                   "tools": TOOLS, "tool_choice": 5, "max_tokens": 16})
    assert s == 400, f"numeric tool_choice: status={s}"

    s, r = c.chat({"model": server.model,
                   "messages": [{"role": "user", "content": "hi"}],
                   "tools": {}, "max_tokens": 16})
    assert s == 400, f"non-array tools: status={s}"

    s, _ = c.get("/v1/definitely-not-real")
    assert s == 404, f"unknown path: status={s}"


def test_overflow_with_tools_400(server):
    """The load-crash regression: an oversized prompt with tools must be
    a clean 400, before the first chunk in streaming mode.  The serve
    layer now pre-rejects provably oversized prompts (4*ctx byte bound),
    so the child never sees this one."""
    c = server.client

    s, r = c.chat({"model": server.model,
                   "messages": [big_message(600_000)],
                   "tools": TOOLS, "tool_choice": "auto", "max_tokens": 64})
    assert s == 400, f"status={s} body={str(r)[:200]}"
    err = r.get("error", {}) if isinstance(r, dict) else {}
    assert err.get("code") == "context_length_exceeded", str(r)[:200]

    chunks, done, raw, err_s = c.stream_chat({
        "model": server.model,
        "messages": [big_message(600_000)],
        "tools": TOOLS, "tool_choice": "auto", "max_tokens": 64,
    })
    assert err_s == "HTTP 400", f"err={err_s} raw={raw[:120]!r}"
    assert "data:" not in raw, raw[:120]


def test_bearer_auth(server):
    key = server.cfg.get("key")
    if not key:
        pytest.skip("config has no 'key'; auth is not enabled")
    c = server.client
    s, _ = c.get("/v1/models")
    assert s == 401, f"no key: status={s}"
    s, _ = c.get("/v1/models",
                 headers={"Authorization": "Bearer wrong"})
    assert s == 401, f"wrong key: status={s}"
    s, _ = c.get("/v1/models",
                 headers={"Authorization": f"Bearer {key}"})
    assert s == 200, f"good key: status={s}"
    s, _ = c.health()
    assert s == 200, "health must stay open without the key"
