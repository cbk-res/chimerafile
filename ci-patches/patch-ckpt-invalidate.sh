#!/bin/bash
# Patch llama.cpp to invalidate context checkpoints that cover tokens the new
# prompt does not share (see ci-patches/ckpt-invalidate.patch).
#
# When a slot switched to an unrelated prompt, the LCP reuse path kept and
# restored checkpoints of the previous prompt: the restore's cell count then
# disagreed with the live state and load_tgt aborted ("checkpoint size
# mismatch"). The invalidation now also drops checkpoints with n_tokens beyond
# the prefix shared with the new prompt (n_prefix, captured before any
# checkpoint is restored). Ported from upstream llama.cpp f174553, which adds
# the matching regression tests to tools/server/tests/unit/test_completion.py.
#
# 'make setup' calls this after llamafile's own setup patches are in (see the
# root Makefile). Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/ckpt-invalidate.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

if grep -q "cur.n_tokens > n_prefix" "$TARGET/tools/server/server-context.cpp"; then
    echo "llama.cpp already patched for SWA checkpoint invalidation, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: stale SWA checkpoints are invalidated on LCP reuse, closing the checkpoint size-mismatch abort."
