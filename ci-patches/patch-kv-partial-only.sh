#!/bin/bash
# Patch llama.cpp so each KV cache applies PARTIAL_ONLY itself (see
# ci-patches/kv-partial-only.patch).
#
# A non-SWA cache rolls back with seq_rm, so it holds no partial state; the old
# code still serialized every cell into server checkpoints, and the iSWA
# wrapper gated its base cache inconsistently. Each cache now writes and reads
# nothing for PARTIAL_ONLY unless it is SWA, and the iSWA wrapper only
# delegates. Ported from upstream llama.cpp e467e5b (the dsa-iswa and
# memory-hybrid parts of that commit are not ported: those files do not exist
# at our pin, and no pinned model uses them).
#
# 'make setup' calls this after llamafile's own setup patches are in (see the
# root Makefile). Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/kv-partial-only.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

if grep -q "PARTIAL_ONLY) && swa_type == LLAMA_SWA_TYPE_NONE" "$TARGET/src/llama-kv-cache.cpp"; then
    echo "llama.cpp already patched for PARTIAL_ONLY in the plain KV cache, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: PARTIAL_ONLY is applied by each KV cache, iSWA only delegates."
