#!/usr/bin/env python3
"""Smoke test for `chimerafile serve` — all four engines over one port.

Brings up a real server on a scratch port with one model per engine and
exercises the OpenAI-shaped surface end to end: chat (streaming and not),
audio transcription (whisper + transcribe), image generation + edits,
the error paths, and a clean shutdown.

Standard library only.  Example (from the repo root):

    python3 scripts/serve-smoke-test.py \
        --bin llamafile/o/chimerafile/chimerafile \
        --llama      models/gemma-4-E2B-it-Q4_K_M.gguf \
        --whisper    models/ggml-tiny.bin \
        --transcribe models/whisper-tiny-F16.gguf \
        --diffusion  models/sd15.gguf \
        --audio      models/jfk.wav

Exit status 0 = every check passed.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}".rstrip())


def http(url, body=None, headers=None, timeout=600):
    """GET/POST; returns (status, parsed_json_or_bytes)."""
    req = urllib.request.Request(url, data=body, method="POST" if body else "GET")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, data
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def http_json(url, obj=None, body=None, headers=None, timeout=600):
    """POST a JSON object (obj) or a raw body with explicit headers;
    parses a JSON response, or returns the body as text if it is not JSON."""
    if obj is not None:
        body = json.dumps(obj).encode()
        headers = {"Content-Type": "application/json"}
    status, data = http(url, body, headers or {}, timeout)
    try:
        return status, json.loads(data)
    except Exception:
        return status, data.decode("utf8", "replace")


def multipart_urlencoded(fields, files):
    """Build a multipart/form-data body.  files: {name: (filename, bytes)}."""
    boundary = "chimerafile-smoke-boundary"
    out = []
    for k, v in fields.items():
        out.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
    for k, (fname, data) in files.items():
        out.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        )
        out.append(data)
        out.append(b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    # bytes and str mixed above: rebuild properly
    buf = b""
    for part in out:
        buf += part if isinstance(part, bytes) else part.encode()
    ctype = f"multipart/form-data; boundary={boundary}"
    return buf, ctype


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", required=True, help="path to the chimerafile APE")
    ap.add_argument("--llama", required=True, help="GGUF for the llama engine")
    ap.add_argument("--whisper", required=True, help="ggml-*.bin for the whisper engine")
    ap.add_argument("--transcribe", required=True, help="GGUF for the transcribe engine")
    ap.add_argument("--diffusion", required=True, help="GGUF for the diffusion engine")
    ap.add_argument("--audio", required=True, help="WAV/MP3/FLAC/OGG to transcribe")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--steps", type=int, default=6, help="diffusion steps (keep low)")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    audio_bytes = open(args.audio, "rb").read()

    cfg = {
        "listen": f"127.0.0.1:{args.port}",
        "models": {
            "assistant": {"engine": "llama", "file": os.path.abspath(args.llama),
                          "ctx": 2048, "gpu_layers": 0},
            "scribe": {"engine": "whisper", "file": os.path.abspath(args.whisper)},
            "subtitles": {"engine": "transcribe",
                          "file": os.path.abspath(args.transcribe)},
            "artist": {"engine": "diffusion", "file": os.path.abspath(args.diffusion),
                       "ttl": 0},
        },
    }
    tmp = tempfile.mkdtemp(prefix="chimerafile-smoke-")
    cfg_path = os.path.join(tmp, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    log_path = os.path.join(tmp, "server.log")
    log = open(log_path, "wb")
    # The APE is a multi-OS bundle: on Linux the kernel cannot execve it
    # directly, and the shell's script-fallback is what runs it (the
    # bundle's hidden shell logic locates or installs the user-mode
    # loader).  So spawn it through a shell, the same way a user would.
    import shlex
    proc = subprocess.Popen(
        ["sh", "-c", f"{shlex.quote(args.bin)} serve --config {shlex.quote(cfg_path)}"],
        stdout=log, stderr=subprocess.STDOUT,
    )

    def shutdown():
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        # ── boot ────────────────────────────────────────────────────────
        print("waiting for server ...")
        up = False
        for _ in range(120):
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
            return 1

        # ── status ──────────────────────────────────────────────────────
        s, st = http_json(base + "/v1/status")
        st = st if isinstance(st, dict) else {}
        check("status: 200 + listen", s == 200 and st.get("listen") == cfg["listen"])
        check("status: four models registered",
              sorted(x["id"] for x in st.get("resident", [])) ==
              ["artist", "assistant", "scribe", "subtitles"])
        check("status: all unloaded at boot",
              all(x["state"] == "unloaded" for x in st.get("resident", [])))

        # ── chat (llama) ────────────────────────────────────────────────
        print("chat (first request loads the model) ...")
        s, r = http_json(base + "/v1/chat/completions", {
            "model": "assistant",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 16,
        })
        ok = (
            s == 200
            and r.get("object") == "chat.completion"
            and isinstance(r.get("choices"), list) and len(r["choices"]) == 1
            and isinstance(r["choices"][0].get("message", {}).get("content"), str)
            and "usage" in r
        )
        check("chat: non-stream completion", ok, f"(status={s})")

        s, r = http_json(base + "/v1/chat/completions", {
            "model": "nope",
            "messages": [{"role": "user", "content": "x"}],
        })
        check("chat: 404 unknown model", s == 404 and r.get("error", {}).get("code") == "model_not_found")

        s, _ = http(base + "/v1/chat/completions", b"{not json",
                    {"Content-Type": "application/json"})
        check("chat: 400 invalid JSON", s == 400)

        print("chat streaming ...")
        body = json.dumps({
            "model": "assistant",
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            "max_tokens": 32,
            "stream": True,
        }).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        chunks, done, first_role = [], False, False
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    c = json.loads(payload)
                    chunks.append(c)
                    if c["choices"][0]["delta"].get("role") == "assistant":
                        first_role = True
        except Exception as e:
            check("chat: stream", False, str(e))
            chunks = []
        check("chat: stream chunks + [DONE]",
              len(chunks) > 0 and done and first_role,
              f"(chunks={len(chunks)}, done={done})")

        # ── audio (whisper + transcribe) ────────────────────────────────
        print("audio transcription ...")
        for mid, label in [("scribe", "whisper"), ("subtitles", "transcribe")]:
            mp, ctype = multipart_urlencoded(
                {"model": mid, "response_format": "json"},
                {"file": ("audio.wav", audio_bytes)},
            )
            s, r = http_json(base + "/v1/audio/transcriptions",
                             body=mp, headers={"Content-Type": ctype})
            ok = s == 200 and isinstance(r, dict) and len(r.get("text", "")) > 0
            check(f"audio: {label} json", ok, f"(status={s})")

        mp, ctype = multipart_urlencoded(
            {"model": "scribe", "response_format": "srt"},
            {"file": ("audio.wav", audio_bytes)},
        )
        s, r = http_json(base + "/v1/audio/transcriptions",
                         body=mp, headers={"Content-Type": ctype})
        # srt is not JSON; http_json returns the body as text
        check("audio: srt format",
              s == 200 and isinstance(r, str) and r.startswith("1\n00:"), f"(status={s})")

        mp, ctype = multipart_urlencoded(
            {"model": "assistant"},  # a llama model, not a speech engine
            {"file": ("audio.wav", audio_bytes)},
        )
        s, r = http_json(base + "/v1/audio/transcriptions",
                         body=mp, headers={"Content-Type": ctype})
        check("audio: 400 non-speech model",
              s == 400 and isinstance(r, dict)
              and r.get("error", {}).get("code") == "unsupported_model")

        mp, ctype = multipart_urlencoded({"model": "nope"}, {"file": ("a.wav", audio_bytes)})
        s, _ = http_json(base + "/v1/audio/transcriptions",
                         body=mp, headers={"Content-Type": ctype})
        check("audio: 404 unknown model", s == 404)

        # ── images (diffusion) ──────────────────────────────────────────
        print("images (first request loads the model) ...")
        s, r = http_json(base + "/v1/images/generations", {
            "model": "artist", "prompt": "a red fox", "steps": args.steps, "seed": 42,
        }, timeout=900)
        png = None
        ok = s == 200 and isinstance(r, dict) and isinstance(r.get("data"), list) \
            and len(r["data"]) == 1
        if ok:
            import base64
            png = base64.b64decode(r["data"][0]["b64_json"])
            # PNG magic, then the IHDR colour type (byte 25): 2 = RGB, 6 = RGBA
            ok = png[:8] == b"\x89PNG\r\n\x1a\n" and png[25] in (2, 6)
        check("images: generation -> PNG", ok, f"(status={s})")

        if png:
            mp, ctype = multipart_urlencoded(
                {"model": "artist", "prompt": "make it snowy",
                 "size": "256x256", "steps": str(args.steps), "seed": "3"},
                {"image": ("in.png", png)},
            )
            s, r = http_json(base + "/v1/images/edits",
                             body=mp, headers={"Content-Type": ctype}, timeout=900)
            ok = s == 200 and isinstance(r, dict) and isinstance(r.get("data"), list)
            check("images: edits (img2img)", ok, f"(status={s})")

        s, r = http_json(base + "/v1/images/generations", {
            "model": "scribe", "prompt": "x",
        })
        check("images: 400 non-diffusion model",
              s == 400 and isinstance(r, dict)
              and r.get("error", {}).get("code") == "unsupported_model")

        s, r = http_json(base + "/v1/images/generations", {
            "model": "artist", "prompt": "x", "size": "512",
        })
        check("images: 400 bad size", s == 400)

        # ── repeat generation (fresh-context path) ──────────────────────
        s, r = http_json(base + "/v1/images/generations", {
            "model": "artist", "prompt": "a blue whale", "steps": args.steps, "seed": 43,
        }, timeout=900)
        check("images: second generation (repeat request)",
              s == 200 and isinstance(r, dict) and len(r.get("data", [])) == 1)

        # ── status after traffic ────────────────────────────────────────
        s, st = http_json(base + "/v1/status")
        states = {x["id"]: x["state"] for x in st.get("resident", [])} \
            if isinstance(st, dict) else {}
        check("status: models used are loaded, idle",
              states.get("assistant") == "loaded"
              and states.get("scribe") == "loaded"
              and states.get("artist") == "loaded"
              and all(x["in_use"] == 0 for x in st.get("resident", [])))

    finally:
        shutdown()

    check("clean shutdown (SIGTERM -> exit 0)", proc.returncode == 0,
          f"(exit={proc.returncode})")

    log.close()
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        print(f"server log: {log_path}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
