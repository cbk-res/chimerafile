"""Per-model KV retention, measured through the child's timings.prompt_ms.

The child keeps prompt tokens in its KV cache between requests (unified
cache, matched by content): an identical prompt decodes nothing, a shared
prefix decodes only the new tail, and a divergent prefix pays a full
prefill.  The session server is shared across tests, and the cache never
clears on its own, so every test uses sentences that no other test uses:
a prompt that has never been seen is guaranteed to be a cold prefill.
"""

SENT_A = "the lighthouse keeper wound the great lamp every evening at dusk. "
SENT_B = "the harbour master counted the fishing boats returning through the gap. "
SENT_C = "the night watchman polished the brass bells until they rang like church. "
SENT_D = "the tide table was redrawn each spring when the ice left the bay. "

PREFIX_A = SENT_A * 100
PREFIX_B = SENT_B * 100
PREFIX_C = SENT_C * 90
PREFIX_D = SENT_D * 90
TAIL_1 = " How many sentences did you just read? Reply with the number only."
TAIL_2 = " Who appears in those sentences? Reply with one word."


def prompt(server, text, tail=""):
    return {
        "model": server.model,
        "messages": [{"role": "user", "content": text + tail}],
        "max_tokens": 32,
    }


def prompt_s(r):
    return r["timings"]["prompt_ms"]


def test_identical_prompt_decodes_nothing(server):
    c = server.client
    # A small throwaway request first: it absorbs the one-time setup so
    # the first measured run is a clean cold prefill.
    s0, _ = c.chat(prompt(server, "Say hi."))
    assert s0 == 200

    p = prompt(server, PREFIX_A, TAIL_1)
    s1, r1 = c.chat(p)
    s2, r2 = c.chat(p)
    assert s1 == s2 == 200
    t_cold, t_warm = prompt_s(r1), prompt_s(r2)
    assert t_cold > 0.01, f"cold prefill suspiciously fast: {t_cold:.4f}s"
    assert t_warm < 0.25 * t_cold, \
        f"identical prompt not warm: cold={t_cold:.4f}s warm={t_warm:.4f}s"
    assert r1["usage"]["prompt_tokens"] == r2["usage"]["prompt_tokens"]


def test_shared_prefix_decodes_only_the_tail(server):
    c = server.client
    s0, _ = c.chat(prompt(server, "Say hi. Again."))
    assert s0 == 200

    s1, r1 = c.chat(prompt(server, PREFIX_B, TAIL_1))
    s2, r2 = c.chat(prompt(server, PREFIX_B, TAIL_2))
    assert s1 == s2 == 200
    t_cold, t_tail = prompt_s(r1), prompt_s(r2)
    assert t_tail < 0.5 * t_cold, \
        f"shared prefix not warm: cold={t_cold:.4f}s tail={t_tail:.4f}s"


def test_divergent_prefix_colds_the_cache(server):
    c = server.client
    s0, _ = c.chat(prompt(server, "Say hi. Once more."))
    assert s0 == 200

    s1, r1 = c.chat(prompt(server, PREFIX_C, TAIL_1))
    s2, r2 = c.chat(prompt(server, PREFIX_D, TAIL_1))
    s3, r3 = c.chat(prompt(server, PREFIX_D, TAIL_1))
    assert s1 == s2 == s3 == 200
    t_first, t_divergent, t_warm = prompt_s(r1), prompt_s(r2), prompt_s(r3)
    # The divergent prefix must pay for a full prefill of a similar
    # length, and the repeated one must then be cheap again.
    assert t_divergent > 0.5 * t_first, \
        f"divergent prefix was not a cold prefill: " \
        f"first={t_first:.4f}s divergent={t_divergent:.4f}s"
    assert t_warm < 0.25 * t_divergent, \
        f"repeated divergent prompt not warm: " \
        f"divergent={t_divergent:.4f}s warm={t_warm:.4f}s"
    msg = ((r2.get("choices") or [{}])[0].get("message", {}))
    assert isinstance(msg.get("content"), str), \
        "the clear path must not corrupt the reply"
