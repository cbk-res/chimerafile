#!/bin/bash
# Patch llama.cpp so the model child survives tool schemas with a
# doubly-nested array of objects (see ci-patches/drop-nested-array.patch).
#
# The cosmo build corrupts the model child (the request dies with SIGSEGV)
# when the schema-to-grammar path handles such a schema; the `question`
# tool carried by some clients is the known shape.  Until the root cause
# is fixed, the patch drops the offending tools and logs one warning per
# tool name.  Set LLAMAFILE_DROP_NESTED_ARRAY_TOOLS=0 in the child's
# environment to re-enable them.
#
# 'make setup' calls this after llamafile's own setup patches are in (see
# the root Makefile). Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/drop-nested-array.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

if grep -q "drop_nested_array_tools" "$TARGET/tools/server/server-common.cpp"; then
    echo "llama.cpp already patched for nested-array tool dropping, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: tool schemas with doubly-nested arrays of objects are dropped"
