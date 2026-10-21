#!/bin/bash
# Patch llama.cpp so the model children tolerate tool schemas with unanchored
# regex patterns (see ci-patches/tool-schema-sanitize.patch).
#
# The chat path applies the model's chat template with the request's tools,
# and the schema-to-grammar step that builds the template's output parser
# rejects any `pattern` that is not anchored with ^...$.  Tool schemas from
# editors and MCP servers carry unanchored patterns regularly, so a single
# bad schema fails the whole request.  The patch strips those keys before
# the template is applied.
#
# 'make setup' calls this after llamafile's own setup patches are in (see the
# root Makefile). Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/tool-schema-sanitize.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

if grep -q "sanitize_tool_schemas" "$TARGET/tools/server/server-common.cpp"; then
    echo "llama.cpp already patched for tool schema sanitizing, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: tool schemas with unanchored patterns are stripped before the template is applied."
