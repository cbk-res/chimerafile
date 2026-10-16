#!/usr/bin/env python3
"""Tool-calling test for `chimerafile serve` — the OpenAI tools surface.

Starts a real server from a config file (or reuses one already listening)
and exercises the tool path end to end:

    A  tool_choice="none"           -> no tool calls
    B  tool_choice="auto" + tools   -> structured tool_calls reply
    C  tool-result round trip       -> assistant tool_call + role="tool" message
    D  streaming                    -> stable ids, well-formed deltas, no
                                       raw tool markup leaking into content
    E  request shapes               -> object tool_choice accepted, bad shape
                                       400, parallel_tool_calls accepted

Standard library only.  Example (from the repo root):

    python3 scripts/serve-tools-test.py \
        --config /tmp/opencode/serve-tools-config.json

Exit status 0 = every check passed.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

PASS = 0
FAIL = 0

# The chat template's tool-call markup must never reach the client as
# content: the server parses it into tool_calls.  Generic special-token
# fragments (e.g. "<|") are not listed: when a reply goes off-grammar the
# server deliberately streams the remainder as raw text.
MARKUP_LEAKS = ("<tool_call", "</tool_call", "to=functions", "<function=",
                "<<function")

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
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
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


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}".rstrip())


def http(url, body=None, headers=None, timeout=600):
    """GET/POST; returns (status, bytes)."""
    req = urllib.request.Request(url, data=body, method="POST" if body else "GET")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def http_json(url, obj=None, body=None, headers=None, timeout=600):
    """POST a JSON object (obj) or a raw body with explicit headers;
    parses the response as JSON, or returns the text if it is not JSON."""
    if obj is not None:
        body = json.dumps(obj).encode()
        headers = {"Content-Type": "application/json"}
    status, data = http(url, body, headers or {}, timeout)
    try:
        return status, json.loads(data)
    except Exception:
        return status, data.decode("utf8", "replace")


def stream_chat(base, payload, timeout=600):
    """POST a streaming chat request.

    Returns (chunks, done, error); chunks are the parsed SSE payloads,
    done is True when [DONE] arrived, error is None or a string."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    chunks, done = [], False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_s = line[5:].strip()
                if payload_s == "[DONE]":
                    done = True
                    break
                chunks.append(json.loads(payload_s))
        return chunks, done, None
    except urllib.error.HTTPError as e:
        return [], False, f"HTTP {e.code}: {e.read().decode('utf8', 'replace')[:300]}"
    except Exception as e:
        return [], False, str(e)


def content_of(chunks):
    """Concatenate the content deltas of a streamed reply."""
    out = []
    for c in chunks:
        for ch in c.get("choices", []):
            piece = ch.get("delta", {}).get("content")
            if isinstance(piece, str):
                out.append(piece)
    return "".join(out)


def finish_of(chunks):
    """The last non-null finish_reason seen in a streamed reply."""
    last = None
    for c in chunks:
        for ch in c.get("choices", []):
            if ch.get("finish_reason"):
                last = ch["finish_reason"]
    return last


def assembled_tool_calls(chunks):
    """Reassemble tool_calls from streaming deltas: {index: {...}}.

    Returns (calls, problems) — problems lists structural violations
    (missing index, id changing between deltas, ...)."""
    calls, problems = {}, []
    for c in chunks:
        for ch in c.get("choices", []):
            for tc in ch.get("delta", {}).get("tool_calls", []) or []:
                if "index" not in tc:
                    problems.append("tool_call delta without index")
                    continue
                slot = calls.setdefault(
                    tc["index"],
                    {"id": None, "name": None, "arguments": "", "ids": set()},
                )
                if "id" in tc:
                    slot["ids"].add(tc["id"])
                    if slot["id"] is None:
                        slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    if slot["name"] is None:
                        slot["name"] = fn["name"]
                    elif fn["name"] != slot["name"]:
                        problems.append(f"name changed: {fn['name']}")
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    for idx, slot in calls.items():
        if len(slot["ids"]) > 1:
            problems.append(f"index {idx}: id changed across deltas {slot['ids']}")
        if not slot["id"]:
            problems.append(f"index {idx}: no id")
        if not slot["name"]:
            problems.append(f"index {idx}: no function name")
    return calls, problems


def well_formed_tool_call(message, expected_name):
    """Validate a non-streaming tool_calls entry.  Returns (ok, detail)."""
    tcs = message.get("tool_calls")
    if not isinstance(tcs, list) or not tcs:
        return False, "no tool_calls"
    tc = tcs[0]
    if not isinstance(tc.get("id"), str) or not tc["id"].startswith("call_"):
        return False, f"bad id {tc.get('id')!r}"
    if tc.get("type") != "function":
        return False, f"bad type {tc.get('type')!r}"
    fn = tc.get("function") or {}
    if fn.get("name") != expected_name:
        return False, f"name {fn.get('name')!r}"
    try:
        args = json.loads(fn.get("arguments") or "")
    except Exception as e:
        return False, f"arguments not JSON: {e}"
    if not isinstance(args, dict):
        return False, "arguments is not an object"
    return True, json.dumps(args)


def no_markup(text):
    """True when no raw tool-call markup leaked into the content."""
    return not any(m in text for m in MARKUP_LEAKS)


DUMP_DIR = None


def dump(name, obj):
    """Save a failing exchange for post-mortem; returns the path."""
    global DUMP_DIR
    if DUMP_DIR is None:
        return "(no dump dir)"
    os.makedirs(DUMP_DIR, exist_ok=True)
    path = os.path.join(DUMP_DIR, name + ".json")
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=1)
    except Exception as e:
        return f"(dump failed: {e})"
    return path


def chat(base, payload, timeout=600):
    payload = dict(payload)
    payload["stream"] = False
    return http_json(base + "/v1/chat/completions", payload, timeout=timeout)


def find_tool_call(base, model, max_attempts=3, timeout=600):
    """Ask for a tool call until one comes back well formed.

    Returns (status, response) of the last attempt; callers inspect the
    response themselves."""
    r = None
    for prompt in TOOL_PROMPTS[:max_attempts]:
        s, r = chat(base, {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 512,
        }, timeout=timeout)
        if s != 200:
            return s, r
        msg = (r.get("choices") or [{}])[0].get("message", {})
        ok, _ = well_formed_tool_call(msg, "get_weather")
        if ok:
            return s, r
    return s, r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", default="llamafile/o/chimerafile/chimerafile",
                    help="path to the chimerafile APE")
    ap.add_argument("--config", default="/tmp/opencode/serve-tools-config.json",
                    help="serve config with one llama model")
    ap.add_argument("--port", type=int, default=0,
                    help="port (default: the config's listen port)")
    ap.add_argument("--keep", action="store_true",
                    help="do not shut the server down when finished")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    listen = cfg.get("listen", "127.0.0.1:18080")
    port = args.port or int(listen.rsplit(":", 1)[1])
    models = cfg.get("models", {})
    llama_ids = [k for k, v in models.items() if v.get("engine") == "llama"]
    if not llama_ids:
        print(f"no llama model in {args.config}")
        return 2
    model = llama_ids[0]
    base = f"http://127.0.0.1:{port}"
    global DUMP_DIR
    DUMP_DIR = os.path.join(os.path.dirname(os.path.abspath(args.config)),
                            "serve-tools-dump")
    print(f"model={model} port={port}")

    # Reuse an already-running server if it answers; otherwise spawn ours.
    proc = None
    log = None
    log_path = None
    try:
        s, _ = http(base + "/health", timeout=2)
        running = s == 200
    except Exception:
        running = False
    if running:
        print(f"reusing the server already listening on {port}")
    else:
        import shlex
        log_path = os.path.join(os.path.dirname(os.path.abspath(args.config)),
                                "serve-tools-test.log")
        log = open(log_path, "wb")
        proc = subprocess.Popen(
            ["sh", "-c",
             f"{shlex.quote(os.path.abspath(args.bin))} serve "
             f"--config {shlex.quote(os.path.abspath(args.config))}"],
            stdout=log, stderr=subprocess.STDOUT,
        )
        print("waiting for server ...")
        up = False
        for _ in range(240):
            if proc.poll() is not None:
                break
            try:
                s, _ = http(base + "/health", timeout=2)
                if s == 200:
                    up = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        check("server starts", up, f"(exit={proc.poll()})")
        if not up:
            print(f"server never came up; log at {log_path}")
            log.close()
            return 1

    def shutdown():
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        # ── A: tool_choice="none" ------------------------------------
        print("A: tool_choice=none ...")
        s, r = chat(base, {
            "model": model,
            "messages": [{"role": "user",
                          "content": "Call get_weather for Paris now."}],
            "tools": TOOLS,
            "tool_choice": "none",
            "max_tokens": 64,
        })
        msg = ((r.get("choices") or [{}])[0].get("message", {})
               if isinstance(r, dict) else {})
        finish = ((r.get("choices") or [{}])[0].get("finish_reason")
                  if isinstance(r, dict) else None)
        check("A: 200", s == 200, f"(status={s} body={str(r)[:200]})")
        check("A: no tool_calls", not msg.get("tool_calls"),
              str(msg.get("tool_calls"))[:120])
        check("A: finish_reason != tool_calls", finish != "tool_calls",
              str(finish))
        check("A: content is a string", isinstance(msg.get("content"), str),
              repr(msg.get("content"))[:120])

        # ── B: auto + tools -> structured call -----------------------
        print("B: auto + tools ...")
        s, r = find_tool_call(base, model)
        msg = ((r.get("choices") or [{}])[0].get("message", {})
               if isinstance(r, dict) else {})
        finish = ((r.get("choices") or [{}])[0].get("finish_reason")
                  if isinstance(r, dict) else None)
        ok, detail = well_formed_tool_call(msg, "get_weather")
        check("B: 200", s == 200, f"(status={s})")
        check("B: structured tool_call (id/name/args)", ok, detail)
        if not ok:
            print(f"       dumped: {dump('b_response', r)}")
        check("B: finish_reason=tool_calls", finish == "tool_calls",
              str(finish))
        content = msg.get("content")
        check("B: no markup leak in content",
              content is None or (isinstance(content, str)
                                  and no_markup(content)),
              repr(content)[:200])
        good_call = ok

        # ── C: tool-result round trip --------------------------------
        print("C: tool-result round trip ...")
        tool_id = ("call_" + "0" * 24) if not good_call else \
            msg["tool_calls"][0]["id"]
        s, r = chat(base, {
            "model": model,
            "messages": [
                {"role": "user", "content": "Check the weather in Paris."},
                {"role": "assistant", "tool_calls": [{
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": json.dumps({"city": "Paris"})},
                }]},
                {"role": "tool", "tool_call_id": tool_id,
                 "content": "22 C, sunny with a light breeze."},
                {"role": "user",
                 "content": "Thanks. Given that, what should I wear? "
                            "Answer in one short sentence."},
            ],
            # Generous budget: the thinking model can spend ~500 tokens
            # on reasoning, and the answer must fit after them.
            "max_tokens": 800,
        })
        msg = ((r.get("choices") or [{}])[0].get("message", {})
               if isinstance(r, dict) else {})
        reply = msg.get("content")
        check("C: 200 (tool message accepted)", s == 200,
              f"(status={s} body={str(r)[:200]})")
        check("C: non-empty string reply",
              isinstance(reply, str) and bool(reply.strip()),
              repr(reply)[:200])
        # A reply whose content is empty must at least carry the
        # reasoning the model spent its tokens on.
        check("C: empty content implies reasoning_content",
              bool((reply or "").strip()) or
              bool((msg.get("reasoning_content") or "").strip()),
              f"content={repr(reply)[:80]} "
              f"reasoning={repr(msg.get('reasoning_content'))[:80]}")
        check("C: no markup leak in reply",
              isinstance(reply, str) and no_markup(reply),
              repr(reply)[:200])

        # ── D: streaming deltas + markup-leak check ------------------
        print("D: streaming ...")
        chunks, done, err = None, None, None
        calls, problems = [], ["no tool_calls streamed"]
        # auto first (the interesting path), then "required" as a
        # deterministic fallback so the delta checks always run
        attempts = [(p, "auto") for p in TOOL_PROMPTS]
        attempts.append((TOOL_PROMPTS[0], "required"))
        for prompt, choice in attempts:
            chunks, done, err = stream_chat(base, {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": TOOLS,
                "tool_choice": choice,
                "temperature": 0.7,
                "max_tokens": 512,
                "stream": True,
            })
            if err is None and done:
                calls, problems = assembled_tool_calls(chunks)
                if calls:
                    break
        first_role = bool(chunks) and chunks[0]["choices"][0]["delta"].get(
            "role") == "assistant"
        check("D: stream + [DONE]", err is None and done,
              err or f"(chunks={len(chunks or [])})")
        check("D: first delta role=assistant", first_role,
              str(chunks[0] if chunks else None)[:160])
        check("D: stable ids, well-formed deltas", not problems,
              "; ".join(problems))
        check("D: streamed tool_call assembled", bool(calls),
              "none")
        stream_content = content_of(chunks or [])
        check("D: no markup leak in streamed content", no_markup(stream_content),
              repr(stream_content[:200]))
        fin = finish_of(chunks or [])
        if err is not None or not calls or fin != "tool_calls":
            print(f"       dumped: {dump('d_stream', {'err': err, 'done': done,
                                                      'chunks': chunks})}")
        if calls:
            check("D: finish_reason=tool_calls", fin == "tool_calls", str(fin))
            # the assembled arguments must parse
            args_ok = True
            detail = ""
            for slot in calls.values():
                try:
                    a = json.loads(slot["arguments"])
                    if not isinstance(a, dict):
                        args_ok, detail = False, "not an object"
                except Exception as e:
                    args_ok, detail = False, str(e)
            check("D: streamed arguments parse as JSON", args_ok, detail)

        # ── E: request shapes ----------------------------------------
        print("E: request shapes ...")
        base_msg = [{"role": "user", "content": "Say hi."}]

        s, r = chat(base, {"model": model, "messages": base_msg,
                           "tools": TOOLS,
                           "tool_choice": {"type": "none"},
                           "max_tokens": 32})
        check("E: object tool_choice {type:none} -> 200", s == 200,
              f"(status={s} {str(r)[:160]})")

        s, r = chat(base, {"model": model, "messages": base_msg,
                           "tools": TOOLS,
                           "tool_choice": {"type": "function",
                                           "function": {"name": "get_weather"}},
                           "max_tokens": 32})
        check("E: object tool_choice {type:function} -> 200", s == 200,
              f"(status={s} {str(r)[:160]})")

        s, r = chat(base, {"model": model, "messages": base_msg,
                           "tools": TOOLS,
                           "tool_choice": {"type": "banana"},
                           "max_tokens": 32})
        check("E: object tool_choice bad type -> 400",
              s == 400 and isinstance(r, dict)
              and r.get("error", {}).get("type") == "invalid_request_error",
              f"(status={s} {str(r)[:160]})")

        s, r = chat(base, {"model": model, "messages": base_msg,
                           "tools": TOOLS, "tool_choice": 5,
                           "max_tokens": 32})
        check("E: numeric tool_choice -> 400", s == 400,
              f"(status={s} {str(r)[:160]})")

        s, r = chat(base, {"model": model, "messages": base_msg,
                           "tools": TOOLS, "tool_choice": "auto",
                           "parallel_tool_calls": True,
                           "max_tokens": 32})
        check("E: parallel_tool_calls=true -> 200", s == 200,
              f"(status={s} {str(r)[:160]})")

        s, r = chat(base, {"model": model, "messages": base_msg,
                           "tools": TOOLS, "tool_choice": "auto",
                           "parallel_tool_calls": False,
                           "max_tokens": 32})
        check("E: parallel_tool_calls=false -> 200", s == 200,
              f"(status={s} {str(r)[:160]})")

        s, r = chat(base, {"model": model, "messages": base_msg,
                           "tools": {}, "max_tokens": 32})
        check("E: non-array tools -> 400", s == 400,
              f"(status={s} {str(r)[:160]})")

    finally:
        if not args.keep:
            shutdown()

    if proc is not None:
        check("clean shutdown (SIGTERM -> exit 0)", proc.returncode == 0,
              f"(exit={proc.returncode})")
    if log is not None:
        log.close()

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL and log_path:
        print(f"server log: {log_path}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
