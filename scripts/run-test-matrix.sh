#!/bin/sh
# Run the full test matrix for the serve layer (see tests/README.md).
#
#   scripts/run-test-matrix.sh              # the CPU lane (safe while the
#                                           # live service is up)
#   GPU=vulkan scripts/run-test-matrix.sh   # + the GPU lane (the gpu_exclusive
#                                           # tests skip while the live
#                                           # service holds the GPU)
#
# Every suite spawns its own server on a free port; nothing here touches the
# live service.  Exits non-zero when any suite fails.  PYTEST overrides the
# interpreter (default: pytest on the PATH).

set -u

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if [ -z "${PYTEST:-}" ]; then
    PYTEST=$(command -v pytest || true)
fi
if [ -z "$PYTEST" ] || [ ! -x "$PYTEST" ]; then
    echo "run-test-matrix: no pytest found; set PYTEST=/path/to/pytest" >&2
    exit 2
fi

cd "$root"

fail=0

run() {
    label=$1
    shift
    echo ""
    echo "==> $label"
    if "$PYTEST" "$@" -q; then
        echo "==> $label: OK"
    else
        echo "==> $label: FAILED" >&2
        fail=1
    fi
}

# The CPU lane: everything runs --gpu off (the default), so it costs nothing
# on the GPU and can run while the live service is up.
run "compat + kv + queueing + gizmo (default config)" tests/
run "media (audio + images)" tests/test_media_audio.py \
    tests/test_media_images.py --serve-config tests/configs/tiny-media.json
run "media (sdcpp + transcribe)" tests/test_media_sdcpp.py \
    tests/test_media_transcribe.py \
    --serve-config tests/configs/tiny-media-sdcpp.json
run "residency (two-model config)" tests/test_residency.py \
    --serve-config tests/configs/two-models.json
run "child kv (raw llama --server child)" tests/test_child_kv.py \
    --serve-config tests/configs/tiny-llama-child.json

# The GPU lane: the full suite on the real backend; the gpu_exclusive tests
# (the GPU lease) run when the live service has released the GPU.
if [ -n "${GPU:-}" ]; then
    run "full suite --gpu $GPU" tests/ --gpu "$GPU"
    run "LCP slot-reuse (mellum config, ~15 min)" tests/test_lcp_reuse.py \
        --serve-config tests/configs/lcp-mellum.json --gpu "$GPU"
fi

echo ""
if [ "$fail" = 0 ]; then
    echo "test matrix: all suites passed"
else
    echo "test matrix: one or more suites failed" >&2
fi
exit "$fail"
