#!/bin/bash
# Patch vulkan.c: add device-count guard before backend registration.
# Without this, Vulkan registers even with 0 devices, then crashes
# when ggml tries to initialise a backend for hardware that doesn't
# exist or is incompatible.
#
# We insert the guard right before the "Register the Vulkan backend"
# comment block.
set -e

TARGET="llamafile/llamafile/vulkan.c"
[ -f "$TARGET" ] || { echo "ERROR: $TARGET not found"; exit 1; }

# Check if already patched
if grep -q "CHIMERAFILE_DEVICE_COUNT_GUARD" "$TARGET"; then
    echo "vulkan.c already patched, skipping."
    exit 0
fi

GUARD='
    // CHIMERAFILE_DEVICE_COUNT_GUARD: verify at least one device before
    // registering, matching the CUDA/ROCm behaviour in cuda.c.
    if (g_vulkan.get_device_count.default_abi || g_vulkan.get_device_count.windows_abi) {
        int count;
        if (IsWindows())
            count = g_vulkan.get_device_count.windows_abi();
        else
            count = g_vulkan.get_device_count.default_abi();
        if (count <= 0) {
            llamafile_info("vulkan", "Vulkan library loaded but no devices detected; skipping");
            if (g_vulkan.lib_handle) {
                cosmo_dlclose(g_vulkan.lib_handle);
                g_vulkan.lib_handle = NULL;
            }
            memset(&g_vulkan.backend_init, 0, sizeof(g_vulkan.backend_init));
            memset(&g_vulkan.backend_reg, 0, sizeof(g_vulkan.backend_reg));
            memset(&g_vulkan.get_device_count, 0, sizeof(g_vulkan.get_device_count));
            memset(&g_vulkan.get_device_description, 0, sizeof(g_vulkan.get_device_description));
            memset(&g_vulkan.log_set, 0, sizeof(g_vulkan.log_set));
            return false;
        }
    }
'

# Insert the guard before "// Register the Vulkan backend with GGML"
awk -v guard="$GUARD" '
    /Register the Vulkan backend with GGML/ {
        print guard;
    }
    { print }
' "$TARGET" > "$TARGET.tmp" && mv "$TARGET.tmp" "$TARGET"

echo "vulkan.c patched: added device-count guard before Vulkan registration."
