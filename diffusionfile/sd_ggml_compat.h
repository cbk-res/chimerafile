// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// sd_ggml_compat.h — GGML API compatibility shim for stable-diffusion.cpp
//
// stable-diffusion.cpp was written against its own GGML fork (based on an
// older ggml SHA) which exposes certain functions and struct layouts that
// llama.cpp's GGML has either renamed, moved to internal headers, or
// changed in signature.
//
// This header is injected via -include before any stable-diffusion.cpp
// source file.  It provides:
//
//   1. ggml-cpu.h        — ggml_graph_compute_with_ctx(),
//                          ggml_backend_is_cpu(),
//                          ggml_backend_cpu_set_n_threads(),
//                          ggml_set_f32(), ggml_set_f32_1d(),
//                          ggml_backend_cpu_init()
//   2. struct ggml_cgraph — opaque in public ggml.h; we pull in the
//                          layout via ggml-impl.h (with a rename trick
//                          to avoid a static/extern conflict)
//   3. ggml_upscale()    — 3-arg → 4-arg (added enum ggml_scale_mode)
//
// No modifications to the llamafile submodule are required.

#ifndef SD_GGML_COMPAT_H_
#define SD_GGML_COMPAT_H_

// ── Include headers that expose the APIs stable-diffusion.cpp expects ──────

// ggml-cpu.h provides the CPU-backend-specific functions that
// stable-diffusion.cpp relies on:
//   - ggml_graph_compute_with_ctx()
//   - ggml_backend_is_cpu()
//   - ggml_backend_cpu_set_n_threads()
//   - ggml_set_f32()
//   - ggml_set_f32_1d()
// These live here (not in plain ggml.h) because upstream moved them
// when the backend API was refactored.
//
#include "ggml-cpu.h"

// ggml-impl.h provides the full definition of struct ggml_cgraph.
// In llama.cpp's public ggml.h it is only forward-declared, but
// stable-diffusion.cpp accesses members directly (gf->n_nodes, etc.).
//
// ggml-impl.h also declares ggml_log_callback_default with external
// linkage, while stable-diffusion.cpp's ggml_extend.hpp defines it
// as static inline.  We rename the declaration in ggml-impl.h to
// avoid a linkage conflict.
#define ggml_log_callback_default ggml_log_callback_default_compat_shim
#include "ggml-impl.h"
#undef ggml_log_callback_default

// __STATIC_INLINE__ is defined by stable-diffusion.cpp's ggml_extend.hpp
// (which #define's it to `static inline`), but for our own shim functions
// we need it defined before they are used below.
#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

// ── ggml_backend_cpu_init (GPU-preferring override) ──────────────────────
// stable-diffusion.cpp hard-codes a CPU backend.  The linker --wrap flag
// redirects calls to ggml_backend_cpu_init into __wrap_ggml_backend_cpu_init
// below, which tries a GPU backend first.  __real_ggml_backend_cpu_init is
// provided by the linker and points to the original function in ggml-cpu.o.
//
// This is defined in the compat header so it's available to every
// stable-diffusion.cpp translation unit that calls ggml_backend_cpu_init.
#ifdef __cplusplus
extern "C"
#endif
ggml_backend_t __real_ggml_backend_cpu_init(void);
#ifdef __cplusplus
extern "C"
#endif
ggml_backend_t __wrap_ggml_backend_cpu_init(void) {
    // Try a discrete GPU first (Vulkan, CUDA, etc.)
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu) {
        ggml_backend_t b = ggml_backend_dev_init(gpu, NULL);
        if (b) return b;
    }
    // Try an integrated GPU (e.g. Apple Silicon, AMD APU)
    ggml_backend_dev_t igpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (igpu) {
        ggml_backend_t b = ggml_backend_dev_init(igpu, NULL);
        if (b) return b;
    }
    // Fall back to CPU
    return __real_ggml_backend_cpu_init();
}

// ── ggml_cpu_has_* stubs ──────────────────────────────────────────────────
// These CPU feature query functions existed in older GGML but were removed
// from the public API.  They are only used by sd_get_system_info() for
// display; we stub them to return 0.

// ggml_cpu_has_blas was removed from the public API; the rest are in
// ggml-cpu.h (already included above) with external linkage.
__STATIC_INLINE__ int ggml_cpu_has_blas(void) { return 0; }

// ── ggml_internal_get_type_traits → ggml_get_type_traits ──────────────
// The old API returned the struct by value; the new API returns a pointer.
// We wrap it to keep struct syntax (.to_float) working.
__STATIC_INLINE__
struct ggml_type_traits ggml_internal_get_type_traits(enum ggml_type type) {
    return *ggml_get_type_traits(type);
}

// ── ggml_upscale 3-arg → 4-arg ────────────────────────────────────────────
// The upstream GGML added an enum ggml_scale_mode parameter.  stable-
// diffusion.cpp always passes 3 arguments.  This fixed-arity macro
// appends GGML_SCALE_MODE_NEAREST, preserving the original behaviour.
//
// After expansion the macro no longer matches (4 args != 3), so there
// is no infinite recursion.

#define ggml_upscale(ctx, a, s) ggml_upscale(ctx, a, s, GGML_SCALE_MODE_NEAREST)

#endif /* SD_GGML_COMPAT_H_ */
