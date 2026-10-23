"""Nested tool schemas (the `question` tool shape) must survive the model child.

A tool schema with a doubly-nested array of objects (an array of objects
whose items carry an array of objects - the `question` tool) used to kill
the model child on the first request carrying it.  Root cause: the child
serializes its PEG parser arena with common_peg_arena::to_json() on the
80 KiB cosmo httplib worker stack; the initializer-list construction
deep-copies the whole parser tree, and the embedded tool schemas are
nested that many levels, so the recursive copy (about 10 KiB of stack per
level in this build) overflows into the guard page (SIGSEGV).  The fix
(peg-tomove.patch) builds the arena JSON with moves; the serialized
output is unchanged.

This test requires the "works" mode: the nested schema reaches the child,
the requests complete, and the child stays alive (a death-then-restart
would pass a pure completion check, so the router log is checked for the
child-exit signature too).
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

# Flat array of scalars: never was the crash shape, must keep working.
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

# server-models.cpp logs this when a model child exits (any exit code).
CHILD_EXIT_RE = re.compile(r"exited with status")


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


def test_child_survived_nested_requests(server):
    """A child that dies and gets respawned would still complete the
    follow-up request above; the exit itself must not have happened."""
    log = _router_log(server)
    exits = CHILD_EXIT_RE.findall(log)
    assert not exits, f"model child exited during nested tool requests: {exits}"


def test_flat_array_tool_completes(server):
    c = server.client
    s, r = _chat(c, server.model, "What is 1+1? Answer briefly.",
                 tools=[FLAT_ARRAY_TOOL])
    _assert_complete(s, r, "flat array request")
