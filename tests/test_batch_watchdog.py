"""Batch-allocator seq_pos watchdog (ci-patches/batch-watch.patch).

A production core showed seq_pos[118] destroyed with a corrupted tree header
for a sequence id no running slot ever used: a wild write elsewhere in the
process, noticed only at destruction. The watchdog keeps a ring of recent
batch operations and verifies the seq_pos tree headers on every init() and
destruction; on a violation it dumps the ring and aborts so the core is taken
at detection time.

In normal operation it must be invisible: arm on the first batch in the
llama child, pass the std::set layout self-test, and never abort.
"""

import time


def test_watchdog_armed_and_quiet(server):
    c = server.client
    # The first request loads the model in the llama child; the first
    # llama_batch_allocr construction arms the watchdog there.
    s, _ = c.chat({"model": server.model,
                   "messages": [{"role": "user", "content": "Say hi."}],
                   "max_tokens": 8})
    # A misfiring header check aborts the child and this request would
    # come back as a 502, so the 200 is the no-false-positive check.
    assert s == 200

    if not server.spawned or server.proc_pid is None:
        # A reused server has no known work dir; the 200 above still proves
        # the armed child survives real traffic.
        return

    log = f"/tmp/chimerafile-serve-{server.proc_pid}/router.log"
    text = ""
    for _ in range(20):  # the child's log worker may lag the response
        text = open(log).read()
        if "seq_pos watchdog" in text:
            break
        time.sleep(0.5)
    assert "seq_pos watchdog armed" in text, text
    # the layout self-test gates the checks; FAILED means they are inert
    assert "layout check ok" in text, text
