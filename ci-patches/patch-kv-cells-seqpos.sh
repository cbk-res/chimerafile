#!/bin/bash
# Patch llama.cpp to index KV-cell sequence positions per cell instead of per
# position (see ci-patches/kv-cells-seqpos.patch).
#
# The bookkeeping was a map of position to occurrence count; seq_rm decremented
# the count found at the position, and any desync between the count and the
# cells that actually carry the position landed on std::map::end() and
# corrupted the tree (SIGSEGV in the KV cache during LCP slot-reuse
# truncation on hybrid recurrent+attention models).  Keying the set by
# (position, cell index) makes each update exact, so a desync can no longer
# erase the wrong node.  Ported from upstream llama.cpp 970b51c.
#
# 'make setup' calls this after llamafile's own setup patches are in (see the
# root Makefile). Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/kv-cells-seqpos.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

if grep -q "std::set<std::pair<llama_pos, uint32_t>> seq_pos" "$TARGET/src/llama-kv-cells.h"; then
    echo "llama.cpp already patched for per-cell seq_pos, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: KV-cell seq_pos is indexed per cell, closing the seq_rm desync UB."
