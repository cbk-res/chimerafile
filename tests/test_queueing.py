"""Queueing: the pump serialises engine work on the main thread while the
HTTP workers stay responsive.  These tests hammer that from several
threads at once."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor


def test_concurrent_chats_all_succeed(server):
    c = server.client
    if server.profile == "tiny":
        # The tiny model decodes at ~1800 tok/s and ends its turns on its
        # own end-of-turn token (it is a 5M model trained on short
        # stories, so turns die at a few hundred tokens regardless of
        # max_tokens or temperature), so a 4-chat burst would finish
        # between health polls.  Widen the burst instead.
        n, max_tokens = 32, 256
        questions = [f"Say hello number {i}." for i in range(n)]
    else:
        n, max_tokens = 4, 48
        questions = ["Say ok.",
                     "What is 1+1? Answer in one word.",
                     "Name a colour.",
                     "Say goodbye."]

    def ask(i):
        return c.chat({
            "model": server.model,
            "messages": [{"role": "user", "content": questions[i]}],
            "max_tokens": max_tokens,
        })

    healths = []
    stop = threading.Event()

    def probe():
        while not stop.is_set():
            s, _ = c.health(timeout=2)
            healths.append(s)
            time.sleep(0.25)

    hp = threading.Thread(target=probe, daemon=True)
    hp.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(ask, range(len(questions))))
    finally:
        stop.set()
        hp.join()

    assert all(s == 200 for s, _ in results), \
        f"statuses={[s for s, _ in results]} bodies={str(results)[:300]}"
    for s, r in results:
        msg = ((r.get("choices") or [{}])[0].get("message", {}))
        assert isinstance(msg.get("content"), str), str(r)[:200]
    # The health probes ran while generations were in flight.
    assert len(healths) >= 3, f"too few probes: {len(healths)}"
    assert all(h == 200 for h in healths), f"health dropped: {healths[:10]}"


def test_stream_and_nonstream_do_not_interfere(server):
    c = server.client
    done = []

    def stream():
        chunks, ok, raw, err = c.stream_chat({
            "model": server.model,
            "messages": [{"role": "user",
                          "content": "Count from one to ten."}],
            "max_tokens": 128,
        })
        done.append(("stream", err is None and ok, len(chunks)))

    def plain():
        s, r = c.chat({
            "model": server.model,
            "messages": [{"role": "user",
                          "content": "What is 2+2? One word."}],
            "max_tokens": 48,
        })
        done.append(("plain", s == 200, r))

    ths = [threading.Thread(target=stream), threading.Thread(target=plain)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout=600)

    assert len(done) == 2, f"lost a request: {done}"
    for name, ok, detail in done:
        assert ok, f"{name} failed: {str(detail)[:200]}"
    # The plain reply must be intact despite the interleaved stream.
    plain = [d for d in done if d[0] == "plain"][0][2]
    msg = ((plain.get("choices") or [{}])[0].get("message", {}))
    assert isinstance(msg.get("content"), str)


def test_server_recovers_after_burst(server):
    c = server.client
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(
            lambda i: c.chat({
                "model": server.model,
                "messages": [{"role": "user", "content": f"Say word {i}."}],
                "max_tokens": 16,
            }), range(6)))
    assert all(s == 200 for s, _ in results), \
        f"statuses={[s for s, _ in results]}"
    s, r = c.chat({
        "model": server.model,
        "messages": [{"role": "user",
                      "content": "Are you still there? Say yes."}],
        "max_tokens": 32,
    })
    assert s == 200, f"post-burst: status={s} body={str(r)[:200]}"
