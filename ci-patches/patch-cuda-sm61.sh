#!/bin/bash
# Patch cuda.sh: add Pascal (sm_61) support for older NVIDIA GPUs like
# the GeForce MX150 and GTX 10-series.  Without this the CUDA backend
# has no compatible kernel code and will crash during initialisation
# on these devices.
#
# We insert sm_61 as the FIRST architecture so the driver picks it for
# older GPUs while still having PTX/SASS for newer ones.
set -e

TARGET="llamafile/llamafile/cuda.sh"
[ -f "$TARGET" ] || { echo "ERROR: $TARGET not found"; exit 1; }

if grep -q "CHIMERAFILE_SM61" "$TARGET"; then
    echo "cuda.sh already patched, skipping."
    exit 0
fi

# Insert marker comment BEFORE the ARCH_FLAGS block so it stays outside
# the double-quoted string (putting it inside would leak comment text
# as nvcc arguments when $arch_flags is expanded unquoted).
sed -i '/^# NVIDIA GPU architecture targets$/i\# CHIMERAFILE_SM61: Pascal (GTX 10-series, MX150)' "$TARGET"

# Insert sm_61 before the first sm_75 line in the non-minimal arch block.
# No inline comment here — the trailing \\ must be the last thing before
# the newline for bash line continuation to work inside the "..." string.
sed -i '/gencode arch=compute_75,code=sm_75 \\/i\  -gencode arch=compute_61,code=sm_61 \\' "$TARGET"

echo "cuda.sh patched: added sm_61 (Pascal) architecture support."
