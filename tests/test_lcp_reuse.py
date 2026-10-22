"""LCP slot-reuse crash scenario on a hybrid-recurrent (Qwen3-Next) model.

Regression gate for the kv-cells seq_pos crash: a long multi-turn
conversation reuses the same slot via LCP prefix matching, and because the
assistant reply is truncated before it is sent back, every next prompt
diverges slightly inside the previous completion - the slot's stale tail is
truncated on every turn (f_keep < 1.0).  On the old per-position count map
in llama_kv_cells that path could desync the bookkeeping and corrupt the
cell tree (child SIGSEGV, observed on live and vehicle runs).

The scenario mirrors the live opencode session: a ~9k-token filler system
prompt, the vehicle's 9 flat tools on every request (question absent, as
in live), and a growing "step N" conversation.

Verdicts the test reports:
    works   - every request completes, the child survives, and the log
              shows the LCP reuse path firing with truncation (f_sim > 0.9,
              f_keep < 1.0), proving the scenario went through the path;
    dropped - the serve layer dropped the request (4xx/5xx, child alive);
    crash   - the model child died (recycled or gone) or the server died.

Run it (~15 min, iGPU):
    pytest tests/test_lcp_reuse.py --serve-config tests/configs/lcp-mellum.json \
        --gpu vulkan

Against the live service instead:
    pytest tests/test_lcp_reuse.py --serve-config \
        tests/configs/lcp-mellum-live.json --reuse
"""

import json
import os
import re
from pathlib import Path

import pytest

from conftest import ROOT, health_ok

N = int(os.environ.get("CHIMERAFILE_LCP_N", "25"))

TOOLS_PATH = ROOT / "tests" / "fixtures" / "tools-flat.json"

para = ("This is deterministic filler text used to approximate the size of the "
        "opencode system prompt. It carries no meaning. " * 6)
SYSTEM = para * 150  # ~9000 tokens

LCP_RE = re.compile(r"selected slot by LCP similarity, "
                    rf"f_sim_best = (\d+\.\d+) .* f_keep = (\d+\.\d+)")


def _mellum_child(model):
    """pid of the model child for this config, or None."""
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{d}/cmdline").read_bytes() \
                .decode("utf8", "replace")
        except OSError:
            continue
        if "--alias" in cmd and f" {model} " in f" {cmd} ":
            return int(d)
    return None


def _pid_listening_on(port):
    """PID of the process listening on the TCP port (Linux /proc)."""
    try:
        tcp = Path("/proc/net/tcp").read_text()
    except OSError:
        return None
    inodes = set()
    for line in tcp.splitlines()[1:]:
        parts = line.split()
        # state 0A = LISTEN; local_address is hex IP:PORT
        if len(parts) >= 10 and parts[3] == "0A" \
                and parts[1].rsplit(":", 1)[1] == f"{port:04X}":
            inodes.add(parts[9])
    if not inodes:
        return None
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            fds = os.listdir(f"/proc/{d}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(f"/proc/{d}/fd/{fd}")
            except OSError:
                continue
            m = re.match(r"socket:\[(\d+)\]", link)
            if m and m.group(1) in inodes:
                return int(d)
    return None


def _router_log(server):
    # A spawned server gives us the pid directly; a reused one (e.g. the
    # live service) is found via the port it listens on.
    pid = server.proc_pid or _pid_listening_on(server.port)
    if not pid:
        return None
    log = Path("/tmp") / f"chimerafile-serve-{pid}" / "router.log"
    if not log.is_file():
        return None
    return log.read_text(errors="replace")


def _verdict_fail(server, child_before, status, body, label):
    """Classify a failed request and fail the test with the verdict."""
    child_after = _mellum_child(server.model)
    health = health_ok(server.base)
    detail = f"body={str(body)[:300]}"
    if child_after != child_before:
        pytest.fail(f"CRASH: {label} failed (status={status}, {detail}); "
                    f"model child {child_before} -> {child_after}, "
                    f"server health={health}")
    if not health:
        pytest.fail(f"CRASH: {label} failed (status={status}, {detail}); "
                    f"the server is down")
    pytest.fail(f"DROPPED: {label} failed (status={status}, {detail}); "
                f"child {child_before} alive, server up - the serve layer "
                f"dropped the request")


def _run_scenario(server, n):
    """Drive the growing-context conversation; return (n_ok, child_before)."""
    c = server.client
    tools = json.loads(TOOLS_PATH.read_text())
    child = _mellum_child(server.model)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "hey buddy, just say hi"}]
    n_ok = 0
    for i in range(1, n + 1):
        label = f"request {i}/{n}"
        payload = {"model": server.model,
                   "messages": messages,
                   "tools": tools,
                   "temperature": 0.8,
                   "max_tokens": 600}
        try:
            s, r = c.chat(payload)
        except Exception as e:
            pytest.fail(f"CRASH: {label} failed at the transport level "
                        f"(child died?): {e}")
        if s != 200 or not isinstance(r, dict) or "error" in r \
                or not r.get("choices"):
            _verdict_fail(server, child, s, r, label)
        n_ok += 1
        msg = r["choices"][0]["message"]
        content = msg.get("content") or ""
        # Truncate before sending back: this is what makes the next prompt
        # diverge inside the previous completion (f_keep < 1.0).
        messages.append({"role": "assistant", "content": content[:2000]})
        messages.append({"role": "user", "content":
                         f"step {i}: append one short sentence noting "
                         f"step {i} is done."})
        print(f"  req {i:2d}: OK {len(content):4d} chars child={child}",
              flush=True)
    return n_ok, child


def test_growing_context_lcp_reuse(server):
    """The whole scenario: every request completes and the child survives."""
    if not server.cfg.get("test_lcp"):
        pytest.skip("needs the LCP config (mellum-class hybrid model), "
                    "see tests/configs/lcp-mellum.json")
    n_ok, child = _run_scenario(server, N)
    assert n_ok == N
    assert _mellum_child(server.model) == child, \
        "the model child was recycled during the run"
    s, _ = server.client.health()
    assert s == 200, "the server died after the run"


def test_lcp_path_fired(server):
    """The scenario must actually have exercised the LCP truncation path,
    or the crash test above passes vacuously."""
    if not server.cfg.get("test_lcp"):
        pytest.skip("needs the LCP config")
    log = _router_log(server)
    if log is None:
        pytest.skip("cannot locate the serve work dir "
                    "(no spawned pid, no process on the port)")
    sims, keeps = [], []
    for sim, keep in LCP_RE.findall(log):
        sims.append(float(sim))
        keeps.append(float(keep))
    assert sims, "no LCP slot selection in the log - the scenario did not " \
                 "exercise the LCP path"
    assert max(sims) > 0.9, f"LCP similarity too low to match the crash " \
                            f"signature: {sims[:5]}..."
    truncated = [k for k in keeps if k < 1.0]
    assert truncated, f"LCP reuse never truncated a tail: f_keep values " \
                      f"{keeps[:5]}..."
    print(f"\nLCP path: {len(sims)} reuses, max f_sim={max(sims)}, "
          f"{len(truncated)} truncations (min f_keep={min(truncated)})")
