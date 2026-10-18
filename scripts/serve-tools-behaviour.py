#!/usr/bin/env python3
"""Tool-calling behaviour tests for `chimerafile serve` — behavioural scenarios.

Where serve-tools-test.py checks the protocol surface, this script grades
behaviour: the right tool with the right arguments, forced tool_choice
enforced by the grammar, streaming assembly, schema conformance, and the
overflow-with-tools regression (the load-crash trigger).

Standard library only.  Example (from the repo root):

    python3 scripts/serve-tools-behaviour.py \
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
# content: the server parses it into tool_calls.
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
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone, e.g. Europe/London",
                    },
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                    "op": {
                        "type": "string",
                        "enum": ["add", "sub", "mul", "div"],
                        "description": "Operation",
                    },
                },
                "required": ["a", "b", "op"],
            },
        },
    },
]

SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


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


def chat(base, payload, timeout=600):
    payload = dict(payload)
    payload["stream"] = False
    return http_json(base + "/v1/chat/completions", payload, timeout=timeout)


def stream_chat(base, payload, timeout=600):
    """POST a streaming chat request.

    Returns (chunks, done, raw, error); chunks are the parsed SSE payloads,
    done is True when [DONE] arrived, raw is the full response body text
    (for the no-chunks-before-400 check)."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    chunks, done, raw = [], False, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                text = line.decode("utf8", "replace")
                raw.append(text)
                line = text.strip()
                if not line.startswith("data:"):
                    continue
                payload_s = line[5:].strip()
                if payload_s == "[DONE]":
                    done = True
                    break
                chunks.append(json.loads(payload_s))
        return chunks, done, "".join(raw), None
    except urllib.error.HTTPError as e:
        return [], False, e.read().decode("utf8", "replace"), f"HTTP {e.code}"
    except Exception as e:
        return [], False, "".join(raw), str(e)


def stream_content(chunks):
    out = []
    for c in chunks:
        for ch in c.get("choices", []):
            piece = ch.get("delta", {}).get("content")
            if isinstance(piece, str):
                out.append(piece)
    return "".join(out)


def stream_finish(chunks):
    last = None
    for c in chunks:
        for ch in c.get("choices", []):
            if ch.get("finish_reason"):
                last = ch["finish_reason"]
    return last


def stream_tool_calls(chunks):
    """Reassemble tool_calls from streaming deltas.

    Returns (calls, problems); calls is {index: {id, name, arguments}}."""
    calls, problems = {}, []
    for c in chunks:
        for ch in c.get("choices", []):
            for tc in ch.get("delta", {}).get("tool_calls", []) or []:
                if "index" not in tc:
                    problems.append("tool_call delta without index")
                    continue
                slot = calls.setdefault(
                    tc["index"], {"id": None, "name": None, "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    for idx, slot in calls.items():
        if not slot["id"]:
            problems.append(f"index {idx}: no id")
        if not slot["name"]:
            problems.append(f"index {idx}: no name")
    return calls, problems


def validate_args(name, args, problems):
    """Validate a tool-call's arguments against its schema (JSON-schema
    subset: type, required, properties, enum).  Appends to problems."""
    schema = SCHEMAS.get(name)
    if schema is None:
        problems.append(f"unknown tool {name!r}")
        return
    if not isinstance(args, dict):
        problems.append(f"{name}: arguments not an object")
        return
    for key in schema.get("required", []):
        if key not in args:
            problems.append(f"{name}: missing required {key!r}")
    for key, val in args.items():
        prop = schema.get("properties", {}).get(key)
        if prop is None:
            problems.append(f"{name}: unexpected arg {key!r}")
            continue
        t = prop.get("type")
        if t == "string" and not isinstance(val, str):
            problems.append(f"{name}.{key}: not a string ({val!r})")
        elif t == "number" and (isinstance(val, bool)
                                or not isinstance(val, (int, float))):
            problems.append(f"{name}.{key}: not a number ({val!r})")
        elif t == "integer" and (isinstance(val, bool) or not isinstance(val, int)):
            problems.append(f"{name}.{key}: not an integer ({val!r})")
        if "enum" in prop and val not in prop["enum"]:
            problems.append(f"{name}.{key}: {val!r} not in {prop['enum']}")


def message_of(r):
    return ((r.get("choices") or [{}])[0].get("message", {})
            if isinstance(r, dict) else {})


def finish_of(r):
    return ((r.get("choices") or [{}])[0].get("finish_reason")
            if isinstance(r, dict) else None)


def calls_of(msg):
    """Non-streaming tool_calls as {name: args_dict-or-None} plus problems."""
    calls, problems = {}, []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name")
        try:
            args = json.loads(fn.get("arguments") or "null")
        except Exception:
            args = None
            problems.append(f"{name}: arguments not JSON")
        validate_args(name, args, problems)
        calls[name] = args
    return calls, problems


DUMP_DIR = None


def dump(name, obj):
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


def no_markup(text):
    return not any(m in text for m in MARKUP_LEAKS)


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
    ap.add_argument("--only", default="",
                    help="run only the scenarios whose id contains this "
                         "(e.g. S6, or S6,S8)")
    args = ap.parse_args()
    only = {t.strip() for t in args.only.split(",") if t.strip()}

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
                            "serve-tools-behaviour-dump")
    print(f"model={model} port={port}")

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
                                "serve-tools-behaviour.log")
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
        # ── S1: basic auto call --------------------------------------
        if not only or "S1" in only:
            print("S1: basic auto call ...")
            s, r = chat(base, {
                "model": model,
                "messages": [{"role": "user",
                              "content": "What is the weather in Paris right now?"}],
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 512,
            })
            msg = message_of(r)
            calls, problems = calls_of(msg)
            call_args = calls.get("get_weather")
            check("S1: get_weather called", call_args is not None,
                  f"calls={list(calls)} {str(problems)[:120]}")
            if call_args:
                check("S1: city is Paris",
                      isinstance(call_args.get("city"), str)
                      and "paris" in call_args["city"].lower(),
                  repr(call_args))
            check("S1: no markup leak",
                  msg.get("content") in (None, "") or
                  no_markup(msg.get("content") or ""),
                  repr(msg.get("content"))[:160])

        # ── S2: forced required (grammar enforcement) ----------------
        if not only or "S2" in only:
            print("S2: forced required ...")
            # A math question: the model would answer in plain text, so the
            # grammar's job is to stop it ending the turn without a call.
            s, r = chat(base, {
                "model": model,
                "messages": [{"role": "user",
                              "content": "What is 2+2? Answer in one word."}],
                "tools": TOOLS,
                "tool_choice": "required",
                "max_tokens": 512,
            })
            msg = message_of(r)
            finish = finish_of(r)
            calls, problems = calls_of(msg)
            if calls:
                check("S2: forced call is well formed", not problems,
                      "; ".join(problems))
            else:
                check("S2: no clean turn-end without a call",
                      finish != "stop",
                      f"finish={finish} content={repr(msg.get('content'))[:120]}")

        # ── S3: forced named function --------------------------------
        if not only or "S3" in only:
            print("S3: forced named function ...")
            s, r = chat(base, {
                "model": model,
                "messages": [{"role": "user",
                              "content": "What time is it in London?"}],
                "tools": TOOLS,
                "tool_choice": {"type": "function",
                                "function": {"name": "get_time"}},
                "max_tokens": 512,
            })
            msg = message_of(r)
            calls, problems = calls_of(msg)
            check("S3: a well-formed call comes back",
                  bool(calls) and not problems,
                  f"calls={list(calls)} {str(problems)[:120]}")

        # ── S4: multiple tools in one reply --------------------------
        if not only or "S4" in only:
            print("S4: multiple tools ...")
            s, r = chat(base, {
                "model": model,
                "messages": [{"role": "user",
                              "content": "Give me the weather in Paris and the "
                                         "time in Tokyo."}],
                "tools": TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "max_tokens": 1024,
            })
            msg = message_of(r)
            calls, problems = calls_of(msg)
            names = set(calls)
            check("S4: all calls schema-conformant", not problems,
                  "; ".join(problems[:4]))
            check("S4: weather and time both covered",
                  {"get_weather", "get_time"} <= names,
                  f"tools={names}")

        # ── S5: tool-result round trip --------------------------------
        if not only or "S5" in only:
            print("S5: tool-result round trip ...")
            s, r = chat(base, {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Weather in Paris, please."},
                    {"role": "assistant", "tool_calls": [{
                        "id": "call_behaviour0001",
                        "type": "function",
                        "function": {"name": "get_weather",
                                     "arguments": json.dumps({"city": "Paris"})},
                    }]},
                    {"role": "tool", "tool_call_id": "call_behaviour0001",
                     "content": "22 C, sunny, light breeze."},
                    {"role": "user",
                     "content": "So should I take an umbrella? One sentence."},
                ],
                "max_tokens": 800,
            })
            msg = message_of(r)
            reply = msg.get("content")
            check("S5: answer uses the result (non-empty)",
                  isinstance(reply, str) and bool(reply.strip()),
                  repr(reply)[:160])
            check("S5: no markup leak",
                  isinstance(reply, str) and no_markup(reply),
                  repr(reply)[:160])

        # ── S6: streaming with a forced call --------------------------
        if not only or "S6" in only:
            print("S6: streaming forced call ...")
            # A small model can occasionally emit a call whose arguments
            # do not match its name; one retry keeps that from flaking
            # the suite.
            chunks, done, raw, err = None, False, "", None
            calls, structural, schema_problems = {}, [], []
            for attempt in range(2):
                chunks, done, raw, err = stream_chat(base, {
                    "model": model,
                    "messages": [{"role": "user",
                                  "content": "What is the weather in Berlin?"}],
                    "tools": TOOLS,
                    "tool_choice": "required",
                    "max_tokens": 512,
                    "stream": True,
                })
                if err is not None or not done:
                    break
                calls, structural = stream_tool_calls(chunks)
                schema_problems = []
                for slot in calls.values():
                    try:
                        call_args = json.loads(slot["arguments"] or "null")
                    except Exception:
                        call_args = None
                    else:
                        validate_args(slot["name"], call_args,
                                      schema_problems)
                if not structural and not schema_problems:
                    break
                print(f"  (glitch on attempt {attempt + 1}; retrying)")
            check("S6: stream completed", err is None and done,
                  err or f"(chunks={len(chunks or [])})")
            if err is None and done:
                check("S6: assembled call is well formed",
                      bool(calls) and not structural,
                      f"calls={list(calls)} {str(structural)[:120]}")
                check("S6: streamed args conform to the schema",
                      not schema_problems, "; ".join(schema_problems[:4]))
                if structural or schema_problems:
                    print(f"       dumped: {dump('s6_stream',
                                                 {'err': err, 'done': done,
                                                  'chunks': chunks})}")
            check("S6: no markup in streamed content",
                  no_markup(stream_content(chunks or [])),
                  repr(stream_content(chunks or []))[:160])
            check("S6: finish_reason=tool_calls",
                  bool(calls) and stream_finish(chunks) == "tool_calls",
                  str(stream_finish(chunks)))

        # ── S7: schema conformance (types, enums, required) ----------
        if not only or "S7" in only:
            print("S7: schema conformance ...")
            s, r = chat(base, {
                "model": model,
                "messages": [{"role": "user",
                              "content": "Use the calculate tool to work out "
                                         "what 150 times 4 is."}],
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 512,
            })
            msg = message_of(r)
            calls, problems = calls_of(msg)
            check("S7: a call came back", bool(calls),
                  f"content={repr(msg.get('content'))[:120]}")
            call_args = calls.get("calculate")
            check("S7: calculate args typed correctly",
                  call_args is not None
                  and isinstance(call_args.get("a"), (int, float))
                  and isinstance(call_args.get("b"), (int, float))
                  and call_args.get("op") in ("add", "sub", "mul", "div"),
                  repr(call_args))
            check("S7: no schema violations", not problems, "; ".join(problems))

        # ── S8: multi-turn history with calls -------------------------
        if not only or "S8" in only:
            print("S8: multi-turn history ...")
            s, r = chat(base, {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Time in New York?"},
                    {"role": "assistant", "tool_calls": [{
                        "id": "call_behaviour0002",
                        "type": "function",
                        "function": {"name": "get_time",
                                     "arguments":
                                         json.dumps({"timezone": "America/New_York"})},
                    }]},
                    {"role": "tool", "tool_call_id": "call_behaviour0002",
                     "content": "14:05 EST."},
                    {"role": "assistant", "content": "It is 14:05 in New York."},
                    {"role": "user",
                     "content": "And the weather there? Then tell me what I "
                                "should wear. Keep it to two sentences."},
                ],
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 1024,
            })
            msg = message_of(r)
            reply = msg.get("content")
            calls, problems = calls_of(msg)
            check("S8: handled (well-formed call or answer)",
                  (bool(calls) and not problems) or
                  (isinstance(reply, str) and bool(reply.strip())),
                  f"calls={list(calls)} {str(problems)[:80]} "
                  f"content={repr(reply)[:120]}")
            check("S8: no markup leak",
                  msg.get("content") in (None, "") or
                  no_markup(msg.get("content") or ""),
                  repr(msg.get("content"))[:160])

        # ── S9: overflow with tools (load-crash regression) ----------
        if not only or "S9" in only:
            print("S9: overflow with tools ...")
            sent = "the quick brown fox jumps over the lazy dog. "
            big = (sent * (600_000 // len(sent) + 1))[:600_000]
            big_msg = {"role": "user",
                       "content": big + " What is one plus one? Reply in a word."}
            s, r = chat(base, {
                "model": model,
                "messages": [big_msg],
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 64,
            })
            err = r.get("error", {}) if isinstance(r, dict) else {}
            check("S9: non-streaming 400 context_length_exceeded",
                  s == 400
                  and err.get("type") == "invalid_request_error"
                  and err.get("code") == "context_length_exceeded",
                  f"status={s} {str(r)[:160]}")
            chunks, done, raw, err2 = stream_chat(base, {
                "model": model,
                "messages": [big_msg],
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 64,
                "stream": True,
            })
            check("S9: streaming 400 before any chunk",
                  err2 == "HTTP 400" and "data:" not in raw,
                  f"err={err2} raw={raw[:120]!r}")

        # ── S10: plain chat control (no tools) ------------------------
        if not only or "S10" in only:
            print("S10: plain chat control ...")
            s, r = chat(base, {
                "model": model,
                "messages": [{"role": "user",
                              "content": "What is the capital of France? "
                                         "One word."}],
                "max_tokens": 128,
            })
            msg = message_of(r)
            reply = msg.get("content")
            check("S10: 200 with content",
                  s == 200 and isinstance(reply, str) and bool(reply.strip()),
                  f"status={s} content={repr(reply)[:120]}")
            check("S10: clean finish", finish_of(r) in ("stop", "length"),
                  str(finish_of(r)))

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
