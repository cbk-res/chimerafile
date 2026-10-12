#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# BUILD.mk for chimerafile — unified llamafile + whisperfile binary
#
# Parsed AFTER the primary Makefile (via -f), so all llamafile variables
# (LLAMAFILE_OBJS, LLAMAFILE_DEPS, GGML_OBJS, etc.) are already defined.
#
# We do NOT link whisper.cpp.a because it contains whisper's GGML fork
# (symbol collision with llama.cpp's GGML). Instead we recompile the
# whisper core AND common sources against llama.cpp's GGML headers.

PKGS += CHIMERAFILE

# Version — single source of truth is the VERSION file at repo root
CHIMERAFILE_VERSION_STRING := $(shell cat $(CHIMERAFILE_DIR)/VERSION 2>/dev/null || echo 0.0.0)

# WHISPER_VERSION must match the whisper.cpp submodule.
WHISPER_VERSION := 1.8.3

# ── Build-time engine toggles ────────────────────────────────────────
CHIMERAFILE_WITH_WHISPER    ?= 1
CHIMERAFILE_WITH_DIFFUSION  ?= 1
CHIMERAFILE_WITH_TRANSCRIBE ?= 1

# ==============================================================================
# Include paths
# ==============================================================================

CHIMERAFILE_INCLUDES := \
	-iquote . \
	$(LLAMAFILE_INCLUDES) \
	-iquote whisper.cpp/include \
	-iquote whisper.cpp/src \
	-iquote whisper.cpp/examples \
	-iquote whisperfile

CHIMERAFILE_WHISPER_INCLUDES := \
	-iquote . \
	-iquote whisperfile \
	-iquote llama.cpp/ggml/include \
	-iquote llama.cpp/ggml/src \
	-iquote whisper.cpp/include \
	-iquote whisper.cpp/src \
	-iquote whisper.cpp/examples

# ==============================================================================
# Compiler flags
# ==============================================================================

CXXFLAGS += -DGGML_MAX_NAME=128
CFLAGS   += -DGGML_MAX_NAME=128

# LLAMAFILE_VERSION_STRING comes from llamafile/version.h at compile time.

CHIMERAFILE_CPPFLAGS := \
	$(CHIMERAFILE_INCLUDES) \
	-DLLAMAFILE_TUI \
	-DCOSMOCC=1 \
	-DCHIMERAFILE=1 \
	-DWHISPERFILE \
	-DCHIMERAFILE_VERSION_STRING=\"$(CHIMERAFILE_VERSION_STRING)\"

ifeq ($(CHIMERAFILE_WITH_WHISPER),0)
CHIMERAFILE_CPPFLAGS += -DCHIMERAFILE_NO_WHISPER
endif
ifeq ($(CHIMERAFILE_WITH_DIFFUSION),0)
CHIMERAFILE_CPPFLAGS += -DCHIMERAFILE_NO_DIFFUSION
endif
ifeq ($(CHIMERAFILE_WITH_TRANSCRIBE),0)
CHIMERAFILE_CPPFLAGS += -DCHIMERAFILE_NO_TRANSCRIBE
endif

CHIMERAFILE_WHISPER_CPPFLAGS := \
	$(CHIMERAFILE_WHISPER_INCLUDES) \
	-D_XOPEN_SOURCE=600 \
	-DWHISPERFILE \
	-DCHIMERAFILE=1 \
	-DWHISPER_VERSION=\"$(WHISPER_VERSION)\"

# ==============================================================================
# Entry points
# ==============================================================================

o/$(MODE)/chimerafile/chimerafile.o: $(CHIMERAFILE_DIR)/chimerafile.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_CPPFLAGS) -c -o $@ $<

o/$(MODE)/chimerafile/llama_dispatch.o: $(CHIMERAFILE_DIR)/llama_dispatch.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_CPPFLAGS) -c -o $@ $<

# check_cpu.c is not in LLAMAFILE_SRCS_C (the standalone packages pull it in
# via the pattern rule below), so we build it explicitly.
CHIMERAFILE_CHECK_CPU_OBJ = o/$(MODE)/llamafile/check_cpu.o
$(CHIMERAFILE_CHECK_CPU_OBJ): llamafile/check_cpu.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LLAMAFILE_CPPFLAGS) -c -o $@ $<

# ==============================================================================
# Whisperfile support + entry point
# ==============================================================================
ifeq ($(CHIMERAFILE_WITH_WHISPER),1)

o/$(MODE)/chimerafile/whisperfile_main.o: whisperfile/whisperfile.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) \
		-Dmain=whisperfile_main \
		-c -o $@ $<

o/$(MODE)/chimerafile/slurp.o: whisperfile/slurp.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) -frtti -c -o $@ $<

o/$(MODE)/chimerafile/color.o: whisperfile/color.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) -frtti -c -o $@ $<

o/$(MODE)/chimerafile/common.cpp.o: whisper.cpp/examples/common.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) -frtti -c -o $@ $<

o/$(MODE)/chimerafile/common-ggml.cpp.o: whisper.cpp/examples/common-ggml.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) -frtti -c -o $@ $<

o/$(MODE)/chimerafile/common-whisper.cpp.o: whisper.cpp/examples/common-whisper.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) -frtti -c -o $@ $<

o/$(MODE)/chimerafile/grammar-parser.cpp.o: \
		whisper.cpp/examples/grammar-parser.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) -frtti -c -o $@ $<

o/$(MODE)/chimerafile/cli.chimerafile.cpp.o: \
		whisper.cpp/examples/cli/cli.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) \
		-DGGML_MULTIPLATFORM \
		-frtti -c -o $@ $<

o/$(MODE)/chimerafile/whisper.core.o: whisper.cpp/src/whisper.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) \
		-DGGML_MULTIPLATFORM \
		-frtti -c -o $@ $<

endif  # CHIMERAFILE_WITH_WHISPER

# ==============================================================================
# Diffusion (stable-diffusion.cpp) backend — upstream diffusionfile pattern
# ==============================================================================
# We compile llamafile's own diffusionfile wrapper (diffusionfile/diffusionfile.cpp)
# with -Dmain=diffusionfile_main so the chimerafile dispatcher owns main();
# the wrapper then calls diffusion_cli_main(), which is stable-diffusion.cpp's
# examples/cli/main.cpp compiled with -DDIFFUSIONFILE by the upstream package.
# stable-diffusion.cpp.a, its thirdparty zip and the sd_math_shim are built by
# the upstream patched BUILD.mk against llama.cpp's GGML (single-GGML model).

ifeq ($(CHIMERAFILE_WITH_DIFFUSION),1)

o/$(MODE)/chimerafile/diffusionfile_main.o: diffusionfile/diffusionfile.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) \
		-iquote . \
		-DCOSMOCC=1 \
		-Dmain=diffusionfile_main \
		-c -o $@ $<

endif  # CHIMERAFILE_WITH_DIFFUSION

# ==============================================================================
# Transcribe (transcribe.cpp) backend
# ==============================================================================
# transcribe.cpp ships its OWN vendored ggml, a different version from
# llama.cpp's shared GGML.  To keep a single shared GGML — and therefore one
# set of bundled GPU backends — for every engine, we build transcribe's
# library sources here against llama.cpp's GGML instead of using upstream's
# transcribe.cpp.a (which bundles the private, CPU-only copy).  The llamafile
# submodule stays untouched: its transcribe.cpp.a is never built or linked.

ifeq ($(CHIMERAFILE_WITH_TRANSCRIBE),1)

CHIMERA_TRANSCRIBE_VERSION := $(shell cd transcribe.cpp 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo unknown)

# llamafile's transcribefile wrapper, with main renamed for our dispatcher
o/$(MODE)/chimerafile/transcribefile_main.o: transcribefile/main.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) \
		-iquote . \
		-iquote transcribe.cpp/include \
		-iquote transcribe.cpp/examples/common \
		-DCOSMOCC=1 \
		-DTRANSCRIBEFILE_VERSION_STRING=\"$(CHIMERA_TRANSCRIBE_VERSION)\" \
		-Dmain=transcribefile_main \
		-c -o $@ $<

# transcribe.cpp's example CLI; -DTRANSCRIBEFILE renames its main() to
# transcribe_cli_main() (see transcribe.cpp.patches)
o/$(MODE)/chimerafile/cli.transcribe.cpp.o: transcribe.cpp/examples/cli/main.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) \
		-iquote transcribe.cpp/include \
		-iquote transcribe.cpp/examples/common \
		-DCOSMOCC=1 \
		-DTRANSCRIBEFILE \
		-Wno-sign-compare -Wno-unused-function \
		-c -o $@ $<

o/$(MODE)/chimerafile/wav.cpp.o: transcribe.cpp/examples/common/wav.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) \
		-iquote transcribe.cpp/include \
		-iquote transcribe.cpp/examples/common \
		-DCOSMOCC=1 \
		-c -o $@ $<

o/$(MODE)/chimerafile/cosmo_compat.o: transcribefile/cosmo_compat.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

# transcribe.cpp's library sources, built against the SHARED llama.cpp GGML
# (llama.cpp/ggml) rather than via upstream's transcribe.cpp.a, which bundles
# a private vendored ggml that cannot use the bundled GPU backends.  The source
# set is exactly upstream's TRANSCRIBE_LIB_SRCS (all of transcribe.cpp/src,
# including the vendored miniz).  The ggml_* / ggml_backend_* references resolve
# against the shared GGML already in the link, so transcribe can use the same
# bundled backends (Vulkan et al.) as the other engines.
CHIMERA_TRANSCRIBE_SRCS := $(shell find transcribe.cpp/src -name '*.cpp' -o -name '*.c' | sort)
CHIMERA_TRANSCRIBE_SRCS_CPP := $(filter %.cpp, $(CHIMERA_TRANSCRIBE_SRCS))
CHIMERA_TRANSCRIBE_SRCS_C := $(filter %.c, $(CHIMERA_TRANSCRIBE_SRCS))
CHIMERA_TRANSCRIBE_OBJS := \
	$(CHIMERA_TRANSCRIBE_SRCS_CPP:transcribe.cpp/%.cpp=o/$(MODE)/chimerafile/transcribe/%.cpp.o) \
	$(CHIMERA_TRANSCRIBE_SRCS_C:transcribe.cpp/%.c=o/$(MODE)/chimerafile/transcribe/%.c.o)

CHIMERA_TRANSCRIBE_INCS := \
	-iquote transcribe.cpp/include \
	-iquote transcribe.cpp/src \
	-iquote llama.cpp/ggml/include \
	-iquote llama.cpp/ggml/src \
	-isystem $(COSMOCC)/include/third_party/zlib

# Mirrors upstream's TRANSCRIBE_LIB_CPPFLAGS (see
# transcribe.cpp.patches/llamafile-files/BUILD.mk).  -DGGML_MULTIPLATFORM keeps
# the GGML_CALL backend-interface ABI identical to the shared GGML and the GPU
# dylibs; MINIZ_NO_* trims the vendored miniz to the compress-only path.
CHIMERA_TRANSCRIBE_CPPFLAGS := \
	-DCOSMOCC=1 \
	-DGGML_MULTIPLATFORM \
	-DTRANSCRIBE_BUILD \
	-DGGML_SCHED_MAX_COPIES=4 \
	-DMINIZ_NO_INFLATE_APIS \
	-DMINIZ_NO_STDIO \
	-DMINIZ_NO_TIME \
	-DMINIZ_NO_ZLIB_COMPATIBLE_NAMES

# Static pattern rules (same idiom as the submodule BUILD.mk): the explicit
# per-target rules win over the global o/$(MODE)/%.cpp.o pattern in
# build/rules.mk, which would compile these with the default include paths.
$(CHIMERA_TRANSCRIBE_SRCS_CPP:transcribe.cpp/%.cpp=o/$(MODE)/chimerafile/transcribe/%.cpp.o): \
		o/$(MODE)/chimerafile/transcribe/%.cpp.o: transcribe.cpp/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) \
		$(CHIMERA_TRANSCRIBE_INCS) \
		$(CHIMERA_TRANSCRIBE_CPPFLAGS) \
		-c -o $@ $<

# Vendored miniz.c uses third-party style that trips strict warnings;
# silence them for this one file (upstream does the same via -w).
$(CHIMERA_TRANSCRIBE_SRCS_C:transcribe.cpp/%.c=o/$(MODE)/chimerafile/transcribe/%.c.o): \
		o/$(MODE)/chimerafile/transcribe/%.c.o: transcribe.cpp/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) \
		$(CHIMERA_TRANSCRIBE_INCS) \
		$(CHIMERA_TRANSCRIBE_CPPFLAGS) \
		-w -c -o $@ $<

endif  # CHIMERAFILE_WITH_TRANSCRIBE

# ==============================================================================
# Aggregate object list
# ==============================================================================

CHIMERAFILE_WHISPER_OBJS :=
ifeq ($(CHIMERAFILE_WITH_WHISPER),1)
CHIMERAFILE_WHISPER_OBJS := \
	o/$(MODE)/chimerafile/whisperfile_main.o \
	o/$(MODE)/chimerafile/slurp.o \
	o/$(MODE)/chimerafile/color.o \
	o/$(MODE)/chimerafile/common.cpp.o \
	o/$(MODE)/chimerafile/common-ggml.cpp.o \
	o/$(MODE)/chimerafile/common-whisper.cpp.o \
	o/$(MODE)/chimerafile/grammar-parser.cpp.o \
	o/$(MODE)/chimerafile/cli.chimerafile.cpp.o \
	o/$(MODE)/chimerafile/whisper.core.o
endif

CHIMERAFILE_OWN_OBJS := \
	o/$(MODE)/chimerafile/chimerafile.o \
	o/$(MODE)/chimerafile/llama_dispatch.o \
	$(CHIMERAFILE_CHECK_CPU_OBJ) \
	$(CHIMERAFILE_WHISPER_OBJS)

CHIMERAFILE_DIFFUSION_OBJS :=
ifeq ($(CHIMERAFILE_WITH_DIFFUSION),1)
# The stable-diffusion.cpp package targets (main.diffusionfile.cpp.o,
# stable-diffusion.cpp.a, zip.c.o, sd_math_shim.o) are defined by the
# upstream patched BUILD.mk.  stb.a is listed before the .a so llamafile's
# stb copy wins the duplicate-symbol resolution.
CHIMERAFILE_DIFFUSION_OBJS := \
	o/$(MODE)/chimerafile/diffusionfile_main.o \
	o/$(MODE)/third_party/stb/stb.a \
	o/$(MODE)/stable-diffusion.cpp/examples/cli/main.diffusionfile.cpp.o \
	o/$(MODE)/stable-diffusion.cpp/stable-diffusion.cpp.a \
	o/$(MODE)/stable-diffusion.cpp/thirdparty/zip.c.o \
	o/$(MODE)/stable-diffusion.cpp.patches/sd_math_shim.o
endif

CHIMERAFILE_TRANSCRIBE_OBJS :=
ifeq ($(CHIMERAFILE_WITH_TRANSCRIBE),1)
# The lib objects are built here against the shared GGML (rules above) rather
# than via upstream's transcribe.cpp.a (which bundles a private ggml).
CHIMERAFILE_TRANSCRIBE_OBJS := \
	o/$(MODE)/chimerafile/transcribefile_main.o \
	o/$(MODE)/chimerafile/cli.transcribe.cpp.o \
	o/$(MODE)/chimerafile/wav.cpp.o \
	o/$(MODE)/chimerafile/cosmo_compat.o \
	$(CHIMERA_TRANSCRIBE_OBJS)
endif

CHIMERAFILE_ALL_OBJS := \
	$(CHIMERAFILE_OWN_OBJS) \
	o/$(MODE)/llamafile/server.cpp.o \
	$(LLAMAFILE_OBJS) \
	$(LLAMAFILE_DEPS) \
	$(CHIMERAFILE_DIFFUSION_OBJS) \
	$(CHIMERAFILE_TRANSCRIBE_OBJS)


# ==============================================================================
# GPU backend shared libraries
# ==============================================================================

CHIMERAFILE_GPU_OUTDIR = o/$(MODE)

o/$(MODE)/ggml-cuda.so:
	@echo "==> Building CUDA backend..."
	OUTPUT="$(CHIMERAFILE_GPU_OUTDIR)/ggml-cuda.so" \
	GGML_VERSION="$(GGML_VERSION)" \
	GGML_COMMIT="$(GGML_COMMIT)" \
	llamafile/cuda.sh 2>&1 | while read line; do echo "   [cuda] $$line"; done; \
	touch $@

o/$(MODE)/ggml-vulkan.so:
	@echo "==> Building Vulkan backend..."
	OUTPUT="$(CHIMERAFILE_GPU_OUTDIR)/ggml-vulkan.so" \
	GGML_VERSION="$(GGML_VERSION)" \
	GGML_COMMIT="$(GGML_COMMIT)" \
	llamafile/vulkan.sh 2>&1 | while read line; do echo "   [vulkan] $$line"; done; \
	touch $@

CHIMERAFILE_GPU_SOS := $(wildcard o/$(MODE)/ggml-cuda.so o/$(MODE)/ggml-vulkan.so o/$(MODE)/ggml-rocm.so)

# ==============================================================================
# Final binary
# ==============================================================================

# --allow-multiple-definition dedupes stb (llamafile's third_party/stb vs
# stable-diffusion.cpp's bundled copy) and other benign overlaps.
o/$(MODE)/chimerafile/chimerafile: $(CHIMERAFILE_ALL_OBJS) $(CHIMERAFILE_GPU_SOS) \
	o/$(MODE)/third_party/zipalign/zipalign
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $(filter %.o %.a,$^) $(LDLIBS) \
			-Wl,--allow-multiple-definition
	@echo "==> Built: $@"
	@ZIPALIGN="$(CURDIR)/o/$(MODE)/third_party/zipalign/zipalign"; \
	GPU_DIR=o/$(MODE); \
	HAVE_GPU=0; \
	for so in $$GPU_DIR/ggml-cuda.so $$GPU_DIR/ggml-vulkan.so $$GPU_DIR/ggml-rocm.so; do \
		if [ -f "$$so" ]; then HAVE_GPU=1; fi; \
	done; \
	if [ -x "$$ZIPALIGN" ] && [ "$$HAVE_GPU" = "1" ]; then \
		BUNDLE_DIR=o/$(MODE)/chimerafile/.gpu_bundle; \
		rm -rf "$$BUNDLE_DIR"; mkdir -p "$$BUNDLE_DIR"; \
		for so in $$GPU_DIR/ggml-cuda.so $$GPU_DIR/ggml-vulkan.so $$GPU_DIR/ggml-rocm.so; do \
			if [ -f "$$so" ]; then \
				cp "$$so" "$$BUNDLE_DIR/"; \
				echo "   Bundling: $$(basename $$so)"; \
			fi; \
		done; \
		cd "$$BUNDLE_DIR" && \
		"$$ZIPALIGN" -j0 "$(CURDIR)/$@" *.so && \
		rm -rf "$$BUNDLE_DIR" && \
		echo "==> GPU backends embedded — binary has GPU support"; \
	elif [ "$$HAVE_GPU" = "0" ]; then \
		echo "==> No GPU backends found — binary runs CPU-only."; \
		echo "    Optional: build with 'make cuda' and/or 'make vulkan' then rebuild"; \
	fi

.PHONY: o/$(MODE)/chimerafile
o/$(MODE)/chimerafile: o/$(MODE)/chimerafile/chimerafile
