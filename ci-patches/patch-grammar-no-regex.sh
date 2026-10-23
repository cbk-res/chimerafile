#!/bin/bash
# Patch llama.cpp so the schema-to-grammar path never touches std::regex
# (see ci-patches/grammar-no-regex.patch).
#
# common/json-schema-to-grammar.cpp used std::regex for rule-name
# sanitizing, GBNF literal escaping and the uuid "format" check.  The
# patch replaces all three uses with hand-rolled character scans that
# are byte-for-byte equivalent to the regex versions (verified by
# differential fuzzing) and drops the <regex> include.
#
# Note: this is NOT the fix for the nested-array tool crash.  That crash
# was a stack overflow in common_peg_arena::to_json() (see
# peg-tomove.patch); the model child died serializing the parser arena
# on the 80 KiB cosmo worker stack, before or independent of the grammar
# path.  This patch is kept because it removes per-request std::regex
# construction from the grammar path (and any std::regex risk with it).
#
# 'make setup' calls this after llamafile's own setup patches are in (see
# the root Makefile). Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/grammar-no-regex.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

if grep -q "sanitize_rule_name" "$TARGET/common/json-schema-to-grammar.cpp"; then
    echo "llama.cpp already patched for regex-free grammar building, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: schema-to-grammar no longer uses std::regex."
