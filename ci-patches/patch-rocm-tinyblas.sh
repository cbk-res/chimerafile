#!/bin/bash
# Patch llama.cpp's CUDA/HIP sources for a TinyBLAS-only ROCm build.
#
# llamafile's rocm.sh always builds the ROCm backend for TinyBLAS (it copies
# the tinyblas headers into the build dir and passes -DGGML_USE_TINYBLAS=1),
# but the pinned llama.cpp still includes <hipblas/hipblas.h> and maps every
# cublas* name onto hipBLAS unconditionally, and its error-string helper is
# only compiled for CUDA 12+.  Without this patch the build needs a hipBLAS
# installation and the resulting ggml-rocm.so depends on libhipblas at
# runtime — the extracted ROCm runtime used here has neither.
#
# 'make setup' calls this after llamafile's own setup patches are in (see the
# root Makefile), and the ROCm CI job calls it on a tree that never ran setup
# — hence the two idempotent fix-ups below.  Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/rocm-tinyblas.patch"
SETUP_PATCH="$ROOT/llamafile/llama.cpp.patches/patches/ggml_src_ggml-cuda_common.cuh.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

# 1. llamafile's setup patch compiles the BF16-MMA probe out under
#    GGML_USE_TINYBLAS (TinyBLAS has no BF16; the raw bits would be read as
#    FP16).  'make setup' has applied it already; a tree that skipped setup
#    has not, so apply it here when it is missing.
if ! grep -q "TinyBLAS does not support BF16" "$TARGET/ggml/src/ggml-cuda/common.cuh"; then
    ( cd "$ROOT/llamafile" && patch -p1 --quiet < "$SETUP_PATCH" )
    echo "llama.cpp patched: BF16 MMA probe disabled under TinyBLAS."
fi

# 2. The TinyBLAS delta beyond setup: drop the hipBLAS include/type mappings
#    and make the CUDA error-string helper available to the TinyBLAS path.
if grep -q "GGML_USE_TINYBLAS" "$TARGET/ggml/src/ggml-cuda/vendors/hip.h"; then
    echo "llama.cpp already patched for TinyBLAS, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: TinyBLAS-only ROCm backend (no hipBLAS dependency)."
