"""Child-process lifecycle tests (the "gizmo" layer).

The session-scoped `server` fixture covers the request surface; these
tests drive the process tree itself: spawn failure, crash self-heal,
TTL release, orphan cleanup after a SIGKILL'd serve, generation
cancellation on client disconnect, alias routing, load coalescing,
capacity eviction, and the media child's own lifecycle (crash
self-heal, TTL unload, group kill).  Each test spawns its own serve on
a fresh port and inspects the process table via /proc, scoped to the
spawned tree so the live service is never touched.
"""

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import Client, ROOT, absolutize, health_ok


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Gizmo:
    """A per-test serve process with direct access to its children."""

    def __init__(self, cfg_name, model_file=None):
        cfg = json.loads((ROOT / "tests" / "configs" / cfg_name).read_text())
        absolutize(cfg)
        if model_file:
            for v in cfg["models"].values():
                if v.get("engine") == "llama" and not v.get("alias_of"):
                    v["file"] = model_file
        self.cfg = cfg
        self.port = free_port()
        cfg["listen"] = f"127.0.0.1:{self.port}"
        self.base = f"http://127.0.0.1:{self.port}"
        cfg_path = Path("/tmp/opencode") / f"chimera-gizmo-{self.port}-cfg.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        self.cfg_path = cfg_path
        bin_path = os.environ.get(
            "CHIMERAFILE_BIN",
            str(ROOT / "llamafile" / "o" / "chimerafile" / "chimerafile"))
        self.log_path = Path("/tmp/opencode") / f"chimera-gizmo-{self.port}.log"
        # The APE is not directly exec-able on Linux (see conftest).
        # start_new_session puts serve and its whole tree in their own
        # process group, as a launcher (systemd, job control) would, so
        # the tests can kill the tree the way operators do.
        log = open(self.log_path, "wb")
        cmd = [bin_path, "--gpu", "off", "serve", "--config", str(cfg_path)]
        self.proc = subprocess.Popen(
            ["/bin/sh", "-c", " ".join(shlex.quote(a) for a in cmd)],
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        log.close()
        self.client = Client(self.base)
        deadline = time.time() + 90
        while time.time() < deadline:
            if self.proc.poll() is not None:
                pytest.fail(f"gizmo server died at startup; log: {self.log_path}")
            if health_ok(self.base):
                return
            time.sleep(0.5)
        self.teardown()
        pytest.fail(f"gizmo server never came up; log: {self.log_path}")

    # ── process-tree helpers (scoped to our tree, never the live svc) ──

    def _tree(self):
        """{pid: argv} for our serve, the router, and the model children."""
        argv_of, ppid_of = {}, {}
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = [t for t in
                            f.read().decode(errors="replace").split("\0") if t]
                with open(f"/proc/{pid}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            argv_of[int(pid)] = argv
            ppid_of[int(pid)] = ppid
        tree, stack = {}, [self.proc.pid]
        while stack:
            p = stack.pop()
            if p not in argv_of or p in tree:
                continue
            tree[p] = argv_of[p]
            stack += [q for q, pp in ppid_of.items() if pp == p]
        return tree

    def router_pid(self):
        for pid, argv in self._tree().items():
            if "--models-preset" in argv:
                return pid
        return None

    def child_pid(self, alias):
        for pid, argv in self._tree().items():
            if "--alias" in argv and argv[argv.index("--alias") + 1] == alias:
                return pid
        return None

    def child_pids(self):
        return [pid for pid, argv in self._tree().items()
                if "--alias" in argv and "--models-preset" not in argv]

    def media_pid(self):
        """The media child (chimerafile media), or None when the config has
        no media models."""
        for pid, argv in self._tree().items():
            if "media" in argv:
                return pid
        return None

    def cfg_model_file(self, model_id):
        return self.cfg["models"][model_id].get("file") or ""

    # ── API helpers ───────────────────────────────────────────────────

    def status_rows(self):
        s, body = self.client.get("/v1/status")
        assert s == 200, f"/v1/status: {s} {str(body)[:120]}"
        return {r["id"]: r for r in body.get("resident", [])}

    def chat(self, model, n=16):
        return self.client.chat({
            "model": model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": n,
        })

    def transcribe(self, model, wav_bytes):
        return self.client.multipart(
            "/v1/audio/transcriptions",
            {"model": model, "response_format": "json"},
            "file", "speech.wav", wav_bytes)

    def teardown(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        else:
            self.proc.wait()  # reap (the test may have killed it)


@pytest.fixture
def gserv():
    started = []

    def _start(cfg_name, model_file=None):
        g = Gizmo(cfg_name, model_file)
        started.append(g)
        return g

    yield _start
    for g in started:
        g.teardown()


@pytest.fixture(scope="module")
def speech_wav():
    """A short espeak-ng utterance; one synthesis serves all media tests."""
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not available; cannot synthesise audio")
    out = "/tmp/opencode/chimera-gizmo-speech.wav"
    subprocess.run(
        ["espeak-ng", "-w", out,
         "The quick brown fox jumps over the lazy dog."], check=True)
    with open(out, "rb") as f:
        return f.read()


# ── lifecycle tests ────────────────────────────────────────────────────

def test_spawn_failure_503(gserv, tmp_path):
    # A copy, so the real model file is never disturbed: the config
    # validates the file at startup, so the file must exist to boot and
    # vanish only between requests.
    src = ROOT / "llamafile" / "models" / "TinyLLama-v0.1-5M-F16.gguf"
    dst = tmp_path / "tiny.gguf"
    dst.write_bytes(src.read_bytes())
    g = gserv("tiny-llama.json", model_file=str(dst))
    s, _ = g.chat("tiny")
    assert s == 200, "the model must load while its file exists"
    s, _ = g.client.post("/v1/residency/tiny/unload", {})
    assert s == 200
    dst.unlink()
    s, r = g.chat("tiny")
    assert s == 503, f"status={s} body={str(r)[:200]}"
    assert r["error"]["code"] == "model_load_failed", str(r)[:200]
    row = g.status_rows()["tiny"]
    assert row["state"] == "unloaded" and row["port"] is None, f"row={row}"


def test_crash_self_heal(gserv):
    g = gserv("tiny-llama.json")
    s, _ = g.chat("tiny")
    assert s == 200
    old = g.child_pid("tiny")
    assert old is not None, "no model child after a successful chat"
    os.kill(old, signal.SIGKILL)
    deadline = time.time() + 15
    while time.time() < deadline and g.child_pid("tiny") == old:
        time.sleep(0.2)
    s, r = g.chat("tiny")
    assert s == 200, f"the router must respawn the killed child: {str(r)[:160]}"
    new = g.child_pid("tiny")
    assert new is not None and new != old, "the child was not respawned"


def test_ttl_releases_port(gserv):
    g = gserv("gizmo-ttl.json")  # ttl: 5
    s, _ = g.chat("tiny")
    assert s == 200
    row = g.status_rows()["tiny"]
    assert row["state"] == "loaded" and row["port"], f"row={row}"
    time.sleep(12)  # ttl 5s + the 1s sweeper cycle + the child's exit
    row = g.status_rows()["tiny"]
    assert row["state"] == "unloaded", f"row={row}"
    assert row["port"] is None
    assert g.child_pid("tiny") is None, "the child outlived its TTL"


def _proc_gone(pid, marker):
    """True when pid is dead, or alive but no longer carrying `marker`
    (a reused pid must not count as the original survivor)."""
    if not os.path.isdir(f"/proc/{pid}"):
        return True
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().decode(errors="replace")
    except OSError:
        return True
    return marker not in cmd


def test_no_orphans_after_group_kill(gserv):
    # The APE runtime swallows PR_SET_PDEATHSIG (verified against
    # .ape-1.10), so a bare SIGKILL of the serve process orphans its
    # tree; the contract is that the launcher kills the whole group on a
    # hard stop, as systemd (KillMode=control-group) and job control do.
    g = gserv("tiny-llama.json")
    s, _ = g.chat("tiny")
    assert s == 200
    router, child = g.router_pid(), g.child_pid("tiny")
    assert router is not None and child is not None
    preset_marker = f"/tmp/chimerafile-serve-{g.proc.pid}/"
    model_marker = g.cfg_model_file("tiny")
    os.killpg(g.proc.pid, signal.SIGKILL)
    deadline = time.time() + 15
    while time.time() < deadline:
        if _proc_gone(router, preset_marker) and _proc_gone(child, model_marker):
            break
        time.sleep(0.5)
    assert _proc_gone(router, preset_marker), \
        "the router outlived the group kill"
    assert _proc_gone(child, model_marker), \
        "the model child outlived the group kill"


def test_graceful_stop_reaps_the_tree(gserv):
    # SIGTERM is serve's own guarantee: it stops the router, and the
    # router takes its model children down with it.
    g = gserv("tiny-llama.json")
    s, _ = g.chat("tiny")
    assert s == 200
    router, child = g.router_pid(), g.child_pid("tiny")
    assert router is not None and child is not None
    preset_marker = f"/tmp/chimerafile-serve-{g.proc.pid}/"
    model_marker = g.cfg_model_file("tiny")
    g.proc.send_signal(signal.SIGTERM)
    g.proc.wait(timeout=30)
    deadline = time.time() + 15
    while time.time() < deadline:
        if _proc_gone(router, preset_marker) and _proc_gone(child, model_marker):
            break
        time.sleep(0.5)
    assert _proc_gone(router, preset_marker), \
        "the router outlived the graceful stop"
    assert _proc_gone(child, model_marker), \
        "the model child outlived the graceful stop"


def test_client_disconnect_stops_generation(gserv):
    g = gserv("tiny-llama.json")
    s, _ = g.chat("tiny")
    assert s == 200
    port = g.status_rows()["tiny"]["port"]
    assert port, "no child port to inspect"

    # Open a long stream, let a few frames out, then walk out.
    sock = socket.create_connection(("127.0.0.1", g.port), timeout=60)
    payload = json.dumps({"model": "tiny",
                          "messages": [{"role": "user", "content":
                                        "Count slowly and carefully."}],
                          "max_tokens": 4096, "stream": True}).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\n"
           f"Host: 127.0.0.1:{g.port}\r\n"
           f"Connection: close\r\n"
           f"Content-Type: application/json\r\n"
           f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload
    sock.sendall(req)
    buf = b""
    while buf.count(b'"finish_reason":null') < 3 and len(buf) < (1 << 20):
        b = sock.recv(4096)
        if not b:
            break
        buf += b
    sock.close()

    # The slot must go idle (either way), then prove the generation did
    # not run to completion: the child's own release log says how many
    # tokens it decoded.  A run-to-completion fills the 2048-token slot
    # context; a cancelled one stops within a few dozen.
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots",
                                        timeout=5) as r:
                slots = json.load(r)
            if all(sl.get("state", "idle") == "idle" for sl in slots):
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        pytest.fail("the child's slot never went idle after the disconnect")
    router_log = Path(f"/tmp/chimerafile-serve-{g.proc.pid}/router.log")
    text = router_log.read_text(errors="replace")
    stops = [line for line in text.splitlines() if "stop processing" in line]
    assert stops, "no slot release recorded in the child log"
    n_tokens = int(stops[-1].split("n_tokens = ")[1].split(",")[0])
    assert n_tokens < 500, \
        f"generation ran to completion after the client left ({n_tokens} tokens)"


def test_alias_routing_single_child(gserv):
    g = gserv("gizmo-alias.json")
    s, r1 = g.chat("tiny")
    assert s == 200
    assert r1.get("model") == "tiny", str(r1)[:160]
    s, r2 = g.chat("petit")
    assert s == 200
    assert r2.get("model") == "petit", \
        f"the alias must echo back as the client asked: {str(r2)[:160]}"
    rows = g.status_rows()
    assert rows["tiny"]["port"] and rows["tiny"]["port"] == rows["petit"]["port"]
    assert len(g.child_pids()) == 1, "two ids must share one child"


def test_load_coalescing(gserv):
    g = gserv("tiny-llama.json")
    results, lock = [], threading.Lock()

    def go():
        s, r = g.chat("tiny")
        with lock:
            results.append((s, str(r)[:120]))

    threads = [threading.Thread(target=go) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert all(s == 200 for s, _ in results), str(results)
    assert g.child_pid("tiny") is not None
    assert len(g.child_pids()) == 1, \
        "concurrent first loads must coalesce into one child"


def test_capacity_eviction(gserv):
    g = gserv("gizmo-evict.json")  # max_loaded_models: 1
    s, _ = g.chat("tiny-a")
    assert s == 200
    rows = g.status_rows()
    assert rows["tiny-a"]["state"] == "loaded", f"row={rows['tiny-a']}"
    s, r = g.chat("tiny-b")
    assert s == 200, f"b must load by evicting a: {str(r)[:200]}"
    rows = g.status_rows()
    assert rows["tiny-b"]["state"] == "loaded", f"row={rows['tiny-b']}"
    assert rows["tiny-a"]["state"] == "unloaded", f"row={rows['tiny-a']}"
    assert rows["tiny-a"]["port"] is None
    # And a comes back in turn, evicting b.
    s, _ = g.chat("tiny-a")
    assert s == 200
    rows = g.status_rows()
    assert rows["tiny-a"]["state"] == "loaded", f"row={rows['tiny-a']}"
    assert rows["tiny-b"]["state"] == "unloaded", f"row={rows['tiny-b']}"


def test_pinned_model_blocks_eviction(gserv):
    # tiny-a is pinned and the fleet holds one model: a request for b can
    # neither evict a nor load alongside it -- it gets a retryable 503.
    g = gserv("gizmo-pinned.json")
    s, _ = g.chat("tiny-a")
    assert s == 200
    # Let the admission snapshot catch up: it is published once a second,
    # and the point of the test is the steady-state decision, not the
    # load window.
    time.sleep(1.5)
    req = urllib.request.Request(
        g.base + "/v1/chat/completions",
        data=json.dumps({"model": "tiny-b",
                         "messages": [{"role": "user", "content": "Say hi."}],
                         "max_tokens": 8}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=60)
        pytest.fail("expected the pinned fleet to reject tiny-b")
    except urllib.error.HTTPError as e:
        assert e.code == 503, f"status={e.code}"
        assert e.headers.get("Retry-After") == "5", dict(e.headers)
        r = json.loads(e.read())
        assert r["error"]["code"] == "model_loading", str(r)[:200]
    rows = g.status_rows()
    assert rows["tiny-a"]["state"] == "loaded", \
        f"the pinned model must not be touched: {rows['tiny-a']}"
    assert rows["tiny-b"]["state"] == "unloaded", f"row={rows['tiny-b']}"


# ── media-child lifecycle tests ────────────────────────────────────────

def test_media_crash_self_heal(gserv, speech_wav):
    g = gserv("tiny-media.json")
    s, r = g.transcribe("whisper-turbo", speech_wav)
    assert s == 200, \
        f"the media model must load while the child is up: {str(r)[:160]}"
    old = g.media_pid()
    assert old is not None, "no media child after a successful transcription"
    os.kill(old, signal.SIGKILL)
    # The request path respawns the child (bounded) and retries once, so a
    # single transcription normally heals it; the loop is the safety net.
    deadline = time.time() + 120
    while time.time() < deadline:
        s, r = g.transcribe("whisper-turbo", speech_wav)
        if s == 200:
            break
        time.sleep(2)
    assert s == 200, f"the media child must respawn and serve: {str(r)[:160]}"
    new = g.media_pid()
    assert new is not None and new != old, "the media child was not respawned"
    # The respawn happened on a request thread; the child must still die
    # with serve on a graceful stop (spawn with a clean signal mask).
    g.proc.send_signal(signal.SIGTERM)
    g.proc.wait(timeout=30)
    deadline = time.time() + 15
    while time.time() < deadline and not _proc_gone(new, str(g.cfg_path)):
        time.sleep(0.5)
    assert _proc_gone(new, str(g.cfg_path)), \
        "the respawned media child outlived the graceful stop"


def test_media_ttl_unloads_and_reloads(gserv, speech_wav):
    g = gserv("gizmo-media-ttl.json")  # wspk: whisper, ttl: 3
    s, r = g.transcribe("wspk", speech_wav)
    assert s == 200, f"status={s} body={str(r)[:160]}"
    row = g.status_rows()["wspk"]
    assert row["state"] == "loaded", f"row={row}"
    time.sleep(12)  # ttl 3s + the child's 5s sweeper cycle + the unload
    row = g.status_rows()["wspk"]
    assert row["state"] == "unloaded", f"row={row}"
    # and it comes back on demand
    s, r = g.transcribe("wspk", speech_wav)
    assert s == 200, f"status={s} body={str(r)[:160]}"
    row = g.status_rows()["wspk"]
    assert row["state"] == "loaded", f"row={row}"


def test_no_orphans_after_group_kill_media(gserv):
    # The media config adds the media child to the tree: serve + router +
    # model child + media child must all die with the group.
    g = gserv("tiny-media.json")
    s, _ = g.chat("tiny")
    assert s == 200
    router, child, media = g.router_pid(), g.child_pid("tiny"), g.media_pid()
    assert router is not None and child is not None and media is not None, \
        f"tree incomplete: router={router} child={child} media={media}"
    preset_marker = f"/tmp/chimerafile-serve-{g.proc.pid}/"
    model_marker = g.cfg_model_file("tiny")
    media_marker = str(g.cfg_path)
    os.killpg(g.proc.pid, signal.SIGKILL)
    deadline = time.time() + 15
    while time.time() < deadline:
        if (_proc_gone(router, preset_marker)
                and _proc_gone(child, model_marker)
                and _proc_gone(media, media_marker)):
            break
        time.sleep(0.5)
    assert _proc_gone(router, preset_marker), \
        "the router outlived the group kill"
    assert _proc_gone(child, model_marker), \
        "the model child outlived the group kill"
    assert _proc_gone(media, media_marker), \
        "the media child outlived the group kill"
