#!/bin/bash
# Patch llama.cpp with a seq_pos corruption watchdog for llama_batch_allocr
# (see ci-patches/batch-watch.patch).
#
# A production core showed seq_pos[118] destroyed with a corrupted tree header
# (root = 0x9) for a sequence id no running slot ever uses: a wild write
# elsewhere in the process smashed the set's inline header, and the corruption
# was only noticed when the sets were destroyed, long after it happened. The
# watchdog keeps a 4096-record ring of the recent batch operations (ctor,
# init, clear, ubatch_add, dtor, with token/seq/pos ranges) and verifies the
# seq_pos tree headers on every init() and destruction; on a violation it
# dumps the ring and aborts, so the core is taken at detection time with the
# preceding operations on record. Pure observation -- no behavior change
# unless a violation fires. Disable with LLAMA_BATCH_WATCH=0.
#
# 'make setup' calls this after llamafile's own setup patches are in (see the
# root Makefile). Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/batch-watch.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

if grep -q "seq_pos corruption watchdog" "$TARGET/src/llama-batch.cpp"; then
    echo "llama.cpp already patched with the batch-allocr watchdog, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: batch-allocr seq_pos watchdog armed (LLAMA_BATCH_WATCH=0 disables)."
