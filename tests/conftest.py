"""Config-driven pytest harness for `chimerafile serve`.

The unit under test is the serve integration layer (the part chimerafile
owns, not the engine): the OpenAI surface, model residency, queueing, and
per-model KV retention.  The engine itself is covered by llama.cpp's own
test suite against the pinned submodule.

The default config uses the repo's tiny test model
(llamafile/models/TinyLLama-v0.1-5M-F16.gguf) on the CPU, so a full run
costs nothing on the GPU and can happen while the live service is up:

    # spawn a fresh server from the default (tiny, CPU) config
    pytest tests/

    # or point at any other model / quant / backend
    pytest tests/ --serve-config my-model.json --gpu vulkan

    # reuse a server already listening on the config's port
    pytest tests/ --reuse

    # the residency module needs its own two-model config
    pytest tests/test_residency.py --serve-config tests/configs/two-models.json

Environment:
    CHIMERAFILE_TEST_MODEL   replace the (single) model file in the config
    CHIMERAFILE_BIN          path to the APE (default: the repo build)
    CHIMERAFILE_LIVE_PORT    port of the live service, for the
                             gpu_exclusive skip check (default 18080)
"""

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "tests" / "configs" / "tiny-llama.json"


def pytest_addoption(parser):
    parser.addoption("--serve-config", default=str(DEFAULT_CONFIG),
                     help="serve config for the server under test")
    parser.addoption("--gpu", default="cpu",
                     help="backend for the spawned server: vulkan, metal, "
                          "rocm, or cpu (default: cpu, matching the "
                          "default tiny config)")
    parser.addoption("--reuse", action="store_true",
                     help="use a server already listening on the config port")
    parser.addoption("--keep-server", action="store_true",
                     help="leave the spawned server running afterwards")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu_exclusive: needs the GPU to itself; skipped while the live "
        "service is running (stop it first, see tests/README.md). "
        "Never applies to CPU backends.")


def port_open(port, host="127.0.0.1", timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def health_ok(base, timeout=2):
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip gpu_exclusive tests while the live service holds the GPU."""
    if config.getoption("--gpu") == "cpu":
        return
    try:
        live_port = int(os.environ.get("CHIMERAFILE_LIVE_PORT", "18080"))
    except ValueError:
        live_port = 18080
    if not health_ok(f"http://127.0.0.1:{live_port}"):
        return
    reason = (f"live service on :{live_port} holds the GPU; run "
              "'systemctl --user stop chimerafile-serve' first")
    for item in items:
        if "gpu_exclusive" in item.keywords:
            item.add_marker(pytest.mark.skip(reason=reason))


class Client:
    """Minimal OpenAI-compatible client (stdlib only)."""

    def __init__(self, base):
        self.base = base

    def _request(self, path, body=None, headers=None, timeout=600):
        data = None
        hdrs = dict(headers or {})
        if body is not None:
            data = body if isinstance(body, (bytes, bytearray)) \
                else json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(
            self.base + path, data=data,
            method="POST" if data is not None else "GET", headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def get(self, path, timeout=15, headers=None):
        s, data = self._request(path, headers=headers, timeout=timeout)
        try:
            return s, json.loads(data)
        except Exception:
            return s, data.decode("utf8", "replace")

    def raw(self, path, body, timeout=600):
        """POST a raw byte body (for the malformed-JSON checks)."""
        return self._request(path, body=body, timeout=timeout)

    def health(self, timeout=5):
        s, body = self.get("/health", timeout=timeout)
        return s, (body if isinstance(body, dict) else {})

    def models(self, timeout=10):
        return self.get("/v1/models", timeout=timeout)

    def chat(self, payload, timeout=600):
        """Non-streaming chat; returns (status, parsed-json-or-text)."""
        payload = dict(payload)
        payload["stream"] = False
        s, data = self._request("/v1/chat/completions", payload,
                                timeout=timeout)
        try:
            return s, json.loads(data)
        except Exception:
            return s, data.decode("utf8", "replace")

    def stream_chat(self, payload, timeout=600):
        """Streaming chat; returns (chunks, done, raw, error).

        chunks are the parsed SSE payloads, done is True when [DONE]
        arrived, raw is the full body text (for the no-chunks-before-400
        check), error is None or a string."""
        body = json.dumps(dict(payload, stream=True)).encode()
        chunks, done, raw = [], False, []
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(
                        self.base + "/v1/chat/completions", data=body,
                        headers={"Content-Type": "application/json"}),
                    timeout=timeout) as r:
                for line in r:
                    text = line.decode("utf8", "replace")
                    raw.append(text)
                    line = text.strip()
                    if not line.startswith("data:"):
                        continue
                    p = line[5:].strip()
                    if p == "[DONE]":
                        done = True
                        break
                    chunks.append(json.loads(p))
            return chunks, done, "".join(raw), None
        except urllib.error.HTTPError as e:
            return [], False, e.read().decode("utf8", "replace"), \
                f"HTTP {e.code}"
        except Exception as e:
            return [], False, "".join(raw), str(e)


def absolutize(cfg):
    """Resolve the config's relative paths against the repo root.

    The server resolves model files relative to the config file's
    directory, and a spawned server gets a rewritten copy under /tmp, so
    relative paths would break; the tests themselves also want to find
    the model (CHIMERAFILE_TEST_MODEL) without knowing the layout.
    """
    for entry in cfg.get("models", {}).values():
        f = entry.get("file")
        if f and not os.path.isabs(f):
            entry["file"] = str(ROOT / f)
        args, i = [], 0
        while i < len(entry.get("args", [])):
            if (entry["args"][i] == "--chat-template-file"
                    and i + 1 < len(entry["args"])):
                p = entry["args"][i + 1]
                if not os.path.isabs(p):
                    p = str(ROOT / p)
                args += [entry["args"][i], p]
                i += 2
            else:
                args.append(entry["args"][i])
                i += 1
        entry["args"] = args
    return cfg


class ServerInfo:
    def __init__(self, base, port, cfg, model, models, profile,
                 spawned, log_path):
        self.base = base
        self.port = port
        self.cfg = cfg
        self.model = model      # the primary llama model id
        self.models = models    # every llama model id in the config
        self.profile = profile  # "tiny": random weights, no model-quality tests
        self.spawned = spawned
        self.log_path = log_path
        self.client = Client(base)


@pytest.fixture(scope="session")
def server(request):
    cfg_path = Path(request.config.getoption("--serve-config"))
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text())
    absolutize(cfg)

    # Convenience override for trying another gguf without editing the config.
    override = os.environ.get("CHIMERAFILE_TEST_MODEL")
    if override:
        entries = cfg.get("models", {})
        if len(entries) == 1:
            for entry in entries.values():
                entry["file"] = override
        else:
            pytest.fail("CHIMERAFILE_TEST_MODEL only applies to single-model "
                        "configs")

    listen = cfg.get("listen", "127.0.0.1:18091")
    port = int(listen.rsplit(":", 1)[1])
    base = f"http://127.0.0.1:{port}"
    models = cfg.get("models", {})
    llama_ids = [k for k, v in models.items() if v.get("engine") == "llama"]
    if not llama_ids:
        pytest.fail(f"no llama model in {cfg_path}")

    reuse = request.config.getoption("--reuse")
    proc = None
    log_path = None
    if not reuse and health_ok(base):
        print(f"\nreusing the server already listening on {port}")
        reuse = True
    if not reuse:
        gpu = request.config.getoption("--gpu")
        # The server resolves relative paths against the config's own
        # directory, so it gets a rewritten copy with absolute paths.
        cfg_path = Path("/tmp/opencode") / f"chimera-pytest-{port}-cfg.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        cmd = [os.environ.get("CHIMERAFILE_BIN",
                              str(ROOT / "llamafile" / "o" / "chimerafile"
                                  / "chimerafile"))]
        if gpu != "cpu":
            cmd += ["--gpu", gpu]
        cmd += ["serve", "--config", str(cfg_path)]
        log_path = Path("/tmp/opencode") / f"chimera-pytest-{port}.log"
        log = open(log_path, "wb")
        # The APE is not directly exec-able on Linux; the kernel refuses it
        # and a shell then runs the script embedded in it (which re-execs
        # the APE runtime).  Spawn through a shell for that reason.
        proc = subprocess.Popen(
            ["/bin/sh", "-c", " ".join(shlex.quote(a) for a in cmd)],
            stdout=log, stderr=subprocess.STDOUT)
        print(f"\nspawning test server: {' '.join(cmd)}")
        up = False
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            if health_ok(base):
                up = True
                break
            time.sleep(1)
        if not up:
            proc.kill()
            pytest.fail(f"test server never came up (exit={proc.poll()}); "
                        f"log: {log_path}")
        log.close()

    try:
        yield ServerInfo(base, port, cfg, llama_ids[0], llama_ids,
                         cfg.get("test_profile", "full"),
                         proc is not None, log_path)
    finally:
        if proc is not None and not request.config.getoption("--keep-server"):
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
