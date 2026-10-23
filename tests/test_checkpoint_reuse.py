"""Context-checkpoint reuse on LCP slot switch (the 2026-10-23 20:35 abort).

A slot that switched to an unrelated prompt used to restore checkpoints of
the previous prompt: the restore's byte count disagreed with the live state
and load_tgt aborted ("checkpoint size mismatch"), killing the model child.
The invalidation now also drops checkpoints covering tokens the new prompt
does not share (upstream f174553, ci-patches/ckpt-invalidate.patch), and
each KV cache applies PARTIAL_ONLY itself so checkpoints carry only the
state that cannot be rolled back (upstream e467e5b,
ci-patches/kv-partial-only.patch).

The scenario mirrors the live crash: a long shared system prompt, a growing
multi-turn conversation whose replies are truncated before being sent back
(every next prompt diverges inside the previous completion, f_keep < 1.0),
and a short divergent "probe" turn mid-conversation - a new prompt on a
slot that carries another conversation's checkpoints.  On the iswa vehicle
(lcp-mellum) the checkpoints carry real SWA-window data, so this is the
exact crash shape; on the CPU tiny config it pins the flow end to end.

Verdicts follow test_lcp_reuse.py:
    works   - every request completes, the child survives, and the log
              shows the LCP reuse path firing with truncation;
    dropped - the serve layer dropped the request (4xx/5xx, child alive);
    crash   - the model child died (recycled or gone) or the server died.

Run it (CPU):
    pytest tests/test_checkpoint_reuse.py
Run it on the iswa vehicle (~1 min extra GPU time, part of the GPU lane):
    pytest tests/test_checkpoint_reuse.py \
        --serve-config tests/configs/lcp-mellum.json --gpu vulkan
"""

import os
import re
import time
from pathlib import Path

import pytest

from conftest import health_ok

N_TURNS = int(os.environ.get("CHIMERAFILE_CKPT_TURNS", "6"))

para = ("This is deterministic filler text used to approximate the size of the "
        "opencode system prompt. It carries no meaning. " * 6)
SYSTEM = para * 8  # ~800 tokens: small enough for the tiny vehicle's slot
                  # context, big enough for the LCP prefix to dominate


def _model_child(model):
    """pid of the model child for this config, or None."""
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            # argv is NUL-separated in /proc; make it space-separated
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ") \
                .decode("utf8", "replace")
        except OSError:
            continue
        if "--alias" in cmd and f" {model} " in f" {cmd} ":
            return int(d.name)
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
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            fds = os.listdir(d / "fd")
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(d / "fd" / fd)
            except OSError:
                continue
            m = re.match(r"socket:\[(\d+)\]", link)
            if m and m.group(1) in inodes:
                return int(d.name)
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
    child_after = _model_child(server.model)
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


def _probe(server, child, label):
    """A short, divergent turn sharing the same system prompt - the shape
    that hit a slot carrying another conversation's checkpoints in the live
    crash (partial LCP against the slot's growing prompt)."""
    c = server.client
    payload = {"model": server.model,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user",
                             "content": "What is the weather in Paris "
                                        "right now? Use the tool."}],
               "temperature": 0.0,
               "max_tokens": 16}
    s, r = c.chat(payload)
    if s != 200 or not isinstance(r, dict) or "error" in r \
            or not r.get("choices"):
        _verdict_fail(server, child, s, r, label)
    assert _model_child(server.model) == child, \
        f"the model child died on the probe turn ({label})"
    return r["choices"][0]["message"].get("content") or ""


def _wait_child(server, timeout=300):
    """The model is loaded on the first request, so wait for the child to
    appear before capturing its pid."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        child = _model_child(server.model)
        if child:
            return child
        time.sleep(0.5)
    pytest.fail("the model child never appeared")


def _run_scenario(server, n):
    """Growing conversation with a divergent probe mid-way; every turn
    must complete and the child must survive."""
    c = server.client
    # warm up: the first request also loads the model
    s, r = c.chat({"model": server.model,
                   "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 1})
    if s != 200:
        pytest.fail(f"CRASH: warm-up failed (status={s}, {str(r)[:300]})")
    child = _wait_child(server)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "hey buddy, just say hi"}]
    n_ok = 0
    for i in range(1, n + 1):
        label = f"turn {i}/{n}"
        if i == 3:
            # the probe: no conversation context, partial LCP against the
            # slot's growing prompt
            content = _probe(server, child, label)
        else:
            payload = {"model": server.model,
                       "messages": messages,
                       "temperature": 0.0,
                       "max_tokens": 64}
            try:
                s, r = c.chat(payload)
            except Exception as e:
                pytest.fail(f"CRASH: {label} failed at the transport level "
                            f"(child died?): {e}")
            if s != 200 or not isinstance(r, dict) or "error" in r \
                    or not r.get("choices"):
                _verdict_fail(server, child, s, r, label)
            assert _model_child(server.model) == child, \
                f"the model child died on {label}"
            msg = r["choices"][0]["message"]
            content = msg.get("content") or ""
            # Truncate before sending back: this is what makes the next
            # prompt diverge inside the previous completion (f_keep < 1.0).
            messages.append({"role": "assistant", "content": content[:256]})
            messages.append({"role": "user",
                             "content": f"step {i}: append one short "
                                        f"sentence noting step {i} is "
                                        f"done."})
        n_ok += 1
        print(f"  turn {i:2d}: OK ({label}) child={child}", flush=True)
    return n_ok, child


def test_checkpoint_reuse_survives(server):
    """Every turn completes, including the divergent probe, and the child
    survives - the pre-fix build aborted in load_tgt on this shape."""
    if not server.cfg.get("test_ckpt"):
        pytest.skip("needs the test_ckpt config key, see "
                    "tests/configs/tiny-llama.json")
    n_ok, child = _run_scenario(server, N_TURNS)
    assert n_ok == N_TURNS
    assert _model_child(server.model) == child, \
        "the model child was recycled during the run"
    s, _ = server.client.health()
    assert s == 200, "the server died after the run"


LCP_RE = re.compile(r"selected slot by LCP similarity, "
                    r"f_sim_best = (\d+\.\d+) .* f_keep = (\d+\.\d+)")


def test_checkpoint_lcp_path_fired(server):
    """The scenario must actually have exercised the LCP reuse path, or
    the crash test above passes vacuously."""
    if not server.cfg.get("test_ckpt"):
        pytest.skip("needs the test_ckpt config key")
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
    assert max(sims) > 0.5, f"LCP similarity too low to be the reuse under " \
                            f"test: {sims[:5]}..."
    truncated = [k for k in keeps if k < 1.0]
    assert truncated, f"LCP reuse never truncated a tail: f_keep values " \
                      f"{keeps[:5]}..."
    print(f"\ncheckpoint scenario: {len(sims)} LCP reuses, "
          f"max f_sim={max(sims)}, {len(truncated)} truncations "
          f"(min f_keep={min(truncated)})")
