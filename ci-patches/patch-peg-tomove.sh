#!/bin/bash
# Patch llama.cpp so the peg arena save() path (to_json + dump) fits on the
# 80 KiB cosmo httplib worker stack (see ci-patches/peg-tomove.patch).
#
# A request carrying a tool schema with a doubly-nested array of objects
# (the `question` tool shape) used to kill the model child (SIGSEGV) in
# common_peg_arena::save().  Two stack overflow sites, both driven by the
# schema's nesting depth (7-8 levels) on top of the ~17 KiB request chain:
#
# - to_json() built its result from an initializer list, deep-copying the
#   whole parsers tree; the recursive copy costs ~10 KiB of stack per
#   nesting level in this build.  Now built with move-assignment.
# - the schema JSON was embedded in the serialized format, so dump()'s
#   recursive serializer (5 KiB frame per level here) recursed that deep.
#   The schema is now omitted from the format: parsing follows the child
#   parsers only, grammars are built from the live arena before save(),
#   and load() treats a missing "schema" field as null.
#
# The serialized output stays a full round-trip for everything the loaded
# arena is used for (output parsing); the dropped schema field is the only
# difference.
#
# 'make setup' calls this after llamafile's own setup patches are in (see
# the root Makefile). Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/peg-tomove.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

if grep -q 'result\["parsers"\] = std::move(parsers);' "$TARGET/common/peg-parser.cpp"; then
    echo "llama.cpp already patched for move-based arena serialization, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: peg arena to_json no longer deep-copies the parser tree."
