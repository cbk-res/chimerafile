"""Tool schemas with a doubly-nested array of objects (cosmo crash regression).

The cosmo build corrupts the model child (SIGSEGV, the request dies and
the child is recycled) when the schema-to-grammar path handles a tool
schema that carries a doubly-nested array of objects - the `question`
tool shape: an array of objects whose items themselves carry an array of
objects.  Flat arrays (of scalars or of objects) are unaffected.

This test pins the behavior whatever fix mode the build is in:

- "works":  the nested schema passes the grammar path, the request
  completes;
- "dropped": the child logs one warning per dropped tool ("tools:
  dropped tool schema with nested array of objects (tool=...)") and
  serves the request with the remaining tools.

A crash fails the completion tests hard.  The last test prints which
mode is in effect.
"""

import re
from pathlib import Path

import pytest

# The known crash shape (the `question` tool from opencode).
NESTED_TOOL = {
    "type": "function",
    "function": {
        "name": "question",
        "description": "Ask the user questions during execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string",
                                          "description": "Complete question"},
                            "header": {"type": "string",
                                        "description": "Very short label"},
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string",
                                                   "description": "Display text"},
                                        "description": {"type": "string",
                                                         "description": "Explanation"},
                                    },
                                    "required": ["label", "description"],
                                },
                            },
                        },
                        "required": ["question", "header", "options"],
                    },
                }
            },
            "required": ["questions"],
        },
    },
}

# Flat array of scalars: not the crash shape, must survive any filtering.
FLAT_ARRAY_TOOL = {
    "type": "function",
    "function": {
        "name": "list_things",
        "description": "List things",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["items"],
        },
    },
}

DROP_WARNING_RE = re.compile(r"nested array of objects \(tool=([\w-]+)\)")


def _chat(client, model, text, tools=None, max_tokens=64):
    payload = {"model": model,
               "messages": [{"role": "user", "content": text}],
               "tool_choice": "auto",
               "max_tokens": max_tokens}
    if tools is not None:
        payload["tools"] = tools
    try:
        return client.chat(payload)
    except Exception as e:
        pytest.fail(f"request failed at the transport level (child died?): {e}")


def _assert_complete(s, r, label):
    assert s == 200, f"{label}: status={s} body={str(r)[:300]}"
    assert isinstance(r, dict), f"{label}: non-JSON body: {str(r)[:200]}"
    assert "error" not in r, f"{label}: error in response: {r['error']}"
    assert r.get("choices"), f"{label}: no choices: {str(r)[:300]}"


def _router_log(server):
    if not server.proc_pid:
        pytest.skip("server was reused: cannot locate the serve work dir")
    log = Path("/tmp") / f"chimerafile-serve-{server.proc_pid}" / "router.log"
    if not log.is_file():
        pytest.skip(f"no router log at {log}")
    return log.read_text(errors="replace")


def test_nested_tool_requests_complete(server):
    """The crash signature: a request carrying the nested schema used to
    kill the model child; it must complete, repeatedly, and the child
    must survive a plain follow-up."""
    c = server.client
    for i in range(3):
        s, r = _chat(c, server.model, "What is 1+1? Answer briefly.",
                     tools=[NESTED_TOOL, FLAT_ARRAY_TOOL])
        _assert_complete(s, r, f"nested request {i + 1}")
    s, r = _chat(c, server.model, "What is 1+1? Answer briefly.", max_tokens=16)
    _assert_complete(s, r, "follow-up without tools")


def test_flat_array_tool_completes(server):
    c = server.client
    s, r = _chat(c, server.model, "What is 1+1? Answer briefly.",
                 tools=[FLAT_ARRAY_TOOL])
    _assert_complete(s, r, "flat array request")


def test_flat_array_tool_not_dropped(server):
    """The drop must be precise: a flat array of scalars is not the crash
    shape and must not be dropped with it."""
    dropped = DROP_WARNING_RE.findall(_router_log(server))
    assert FLAT_ARRAY_TOOL["function"]["name"] not in dropped, \
        f"flat array tool was dropped: {dropped}"


def test_nested_mode_printed(server):
    """Informational: which fix mode is this build in?"""
    dropped = DROP_WARNING_RE.findall(_router_log(server))
    if NESTED_TOOL["function"]["name"] in dropped:
        mode = "dropped (schema stripped before the grammar path)"
    else:
        mode = "works (schema passes the grammar path)"
    print(f"\nnested-array tool schema mode: {mode}")
