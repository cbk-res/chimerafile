"""KV retention on a raw upstream llama-server child.

The same scenarios as test_kv_retention.py, but against a bare
``llama --server`` child with no serve layer in between, and with the
direct upstream signal: ``timings.cache_n``, the prompt tokens served
from the KV cache.  This is the empirical check that warm continuation
is inherited from upstream, not something the serve layer adds:

    pytest tests/test_child_kv.py \
        --serve-config tests/configs/tiny-llama-child.json

The child runs with upstream's default parallelism and LCP slot
routing, so the assertions are on cache_n (exact) rather than on
timings: a "cold" request may be routed to a slot that already holds
the prefix.  Divergent prompts still share the chat-template wrapper,
so the cold bound is relative, not zero.
"""

import pytest

SENT_A = "the lighthouse keeper wound the great lamp every evening at dusk. "
SENT_B = "the harbour master counted the fishing boats returning through the gap. "

PREFIX_A = SENT_A * 100
PREFIX_B = SENT_B * 90
TAIL_1 = " How many sentences did you just read? Reply with the number only."
TAIL_2 = " Who appears in those sentences? Reply with one word."


def need_child(server):
    if server.cfg.get("test_spawn") != "llama-server":
        pytest.skip("needs the raw-child config "
                    "(tests/configs/tiny-llama-child.json)")


def prompt(server, text, tail=""):
    return {
        "model": server.model,
        "messages": [{"role": "user", "content": text + tail}],
        "max_tokens": 32,
    }


def cache_n(r):
    return r["timings"]["cache_n"]


def test_identical_prompt_fully_cached(server):
    need_child(server)
    c = server.client
    # A small throwaway request first: it absorbs the one-time setup so
    # the first measured run is a clean cold prefill.
    s0, _ = c.chat(prompt(server, "Say hi."))
    assert s0 == 200

    p = prompt(server, PREFIX_A, TAIL_1)
    s1, r1 = c.chat(p)
    s2, r2 = c.chat(p)
    assert s1 == s2 == 200
    n = r2["usage"]["prompt_tokens"]
    # At most the first token is not served from the cache.
    assert cache_n(r2) >= n - 1, \
        f"expected ~{n} cached tokens, cache_n={cache_n(r2)}"
    assert r1["usage"]["prompt_tokens"] == n


def test_shared_prefix_caches_the_prefix(server):
    need_child(server)
    c = server.client
    s0, _ = c.chat(prompt(server, "Say hi."))
    assert s0 == 200

    s1, r1 = c.chat(prompt(server, PREFIX_A, TAIL_1))
    s2, r2 = c.chat(prompt(server, PREFIX_A, TAIL_2))
    assert s1 == s2 == 200
    n = r2["usage"]["prompt_tokens"]
    # The tail is a dozen tokens on a ~2000-token prefix: nearly
    # everything should come from the cache.
    assert cache_n(r2) > 0.9 * n, \
        f"shared prefix not cached: cache_n={cache_n(r2)} of {n}"


def test_divergent_prefix_colds_the_cache(server):
    need_child(server)
    c = server.client
    s0, _ = c.chat(prompt(server, "Say hi."))
    assert s0 == 200

    s1, r1 = c.chat(prompt(server, PREFIX_A, TAIL_1))
    s2, r2 = c.chat(prompt(server, PREFIX_B, TAIL_1))
    s3, r3 = c.chat(prompt(server, PREFIX_B, TAIL_1))
    assert s1 == s2 == s3 == 200
    # The sentence bodies diverge at the first word, but the
    # chat-template wrapper is shared: cold means only the wrapper was
    # served from the cache.
    n = r2["usage"]["prompt_tokens"]
    assert cache_n(r2) < 0.1 * n, \
        f"divergent prefix should be (nearly) cold, cache_n={cache_n(r2)} of {n}"
    # And repeating it is warm again.
    n3 = r3["usage"]["prompt_tokens"]
    assert cache_n(r3) >= n3 - 1, \
        f"expected ~{n3} cached tokens, cache_n={cache_n(r3)}"
    msg = ((r3.get("choices") or [{}])[0].get("message", {}))
    assert isinstance(msg.get("content"), str), \
        "the cold path must not corrupt the reply"
