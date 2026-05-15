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

CHIMERAFILE_VERSION_STRING := 0.1.0

# WHISPER_VERSION must match the whisper.cpp submodule.
# Normally defined in whisper.cpp/BUILD.mk; redefined here since
# we don't use that file's compiler flags.
WHISPER_VERSION := 1.8.3

# ==============================================================================
# Include paths
# ==============================================================================
# CWD is the llamafile root (due to -C), so -iquote . resolves there.

CHIMERAFILE_INCLUDES := \
	-iquote . \
	$(LLAMAFILE_INCLUDES) \
	-iquote whisper.cpp/include \
	-iquote whisper.cpp/src \
	-iquote whisper.cpp/examples \
	-iquote whisperfile

# Whisper sources: llama.cpp GGML headers shadow whisper.cpp's own GGML.
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

# Increase GGML_MAX_NAME so SD1.5 GGUF models (which use longer tensor names
# like "model.diffusion_model.input_blocks.0.0.op.weight" at 76 chars) can
# be loaded.  The default is 64 in llama.cpp's ggml.h.  This must be set
# globally so every translation unit sees the same ggml_tensor layout.
CXXFLAGS += -DGGML_MAX_NAME=128
CFLAGS   += -DGGML_MAX_NAME=128

CHIMERAFILE_CPPFLAGS := \
	$(CHIMERAFILE_INCLUDES) \
	-DLLAMAFILE_TUI \
	-DCOSMOCC=1 \
	-DCHIMERAFILE=1 \
	-DWHISPERFILE \
	-DCHIMERAFILE_VERSION_STRING=\"$(CHIMERAFILE_VERSION_STRING)\" \
	-DLLAMAFILE_VERSION_STRING=\"$(LLAMAFILE_VERSION_STRING)\"

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

# check_cpu.c is NOT in LLAMAFILE_SRCS_C (only compiled by whisperfile build)
# We need it for llamafile_check_cpu() used by both chimerafile.cpp and whisperfile
CHIMERAFILE_CHECK_CPU_OBJ = o/$(MODE)/llamafile/check_cpu.o
$(CHIMERAFILE_CHECK_CPU_OBJ): llamafile/check_cpu.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LLAMAFILE_CPPFLAGS) -c -o $@ $<

# ==============================================================================
# Whisperfile support + entry point
# ==============================================================================

o/$(MODE)/chimerafile/whisperfile_main.o: whisperfile/whisperfile.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) \
		-DLLAMAFILE_VERSION_STRING="$(LLAMAFILE_VERSION_STRING)" \
		-Dmain=whisperfile_main \
		-c -o $@ $<

o/$(MODE)/chimerafile/slurp.o: whisperfile/slurp.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) -frtti -c -o $@ $<

o/$(MODE)/chimerafile/color.o: whisperfile/color.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) -frtti -c -o $@ $<

# ==============================================================================
# Whisper.cpp common library — normally inside whisper.cpp.a
# These provide to_timestamp(), grammar_parser, stb_vorbis, etc.
# ==============================================================================

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

# ==============================================================================
# Whisper.cpp CLI — main() becomes whisper_cli_main() via -DWHISPERFILE
# ==============================================================================

o/$(MODE)/chimerafile/cli.chimerafile.cpp.o: \
		whisper.cpp/examples/cli/cli.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) \
		-DGGML_MULTIPLATFORM \
		-frtti -c -o $@ $<

# ==============================================================================
# Whisper.cpp core — recompiled against llama.cpp GGML
# ==============================================================================

o/$(MODE)/chimerafile/whisper.core.o: whisper.cpp/src/whisper.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_WHISPER_CPPFLAGS) \
		-DGGML_MULTIPLATFORM \
		-frtti -c -o $@ $<

# ==============================================================================
# Diffusion (stable-diffusion.cpp) backend
# ==============================================================================
# Single-GGML approach: compile stable-diffusion.cpp sources against
# llama.cpp's GGML headers, NOT stable-diffusion.cpp/ggml/ (which is a
# separate GGML fork that would cause duplicate symbols).
#
# stb_image and stb_image_write implementations come from
# llamafile's third_party/stb/stb.a, so our wrapper files pre-include
# those headers to absorb the STB_IMAGE_IMPLEMENTATION defines.

CHIMERAFILE_DIFFUSION_INCLUDES := \
	-iquote . \
	-iquote llama.cpp/ggml/include \
	-iquote llama.cpp/ggml/src \
	-iquote third_party/stb \
	-iquote stable-diffusion.cpp \
	-iquote stable-diffusion.cpp/thirdparty

# sd_ggml_compat.h is injected via -include to provide APIs that vanished
# or changed between stable-diffusion.cpp's GGML fork and llama.cpp's GGML.
CHIMERAFILE_DIFFUSION_CPPFLAGS := \
	$(CHIMERAFILE_DIFFUSION_INCLUDES) \
	-DCOSMOCC=1 \
	-D_XOPEN_SOURCE=600 \
	-DCHIMERAFILE=1 \
	-DLLAMAFILE_VERSION_STRING="$(LLAMAFILE_VERSION_STRING)" \
	-include $(CHIMERAFILE_DIR)/diffusionfile/sd_ggml_compat.h

# Entry point — compiled using CHIMERAFILE_DIFFUSION_INCLUDES but WITHOUT
# sd_ggml_compat.h (which would interfere with cosmo.h's ShowCrashReports).
o/$(MODE)/chimerafile/diffusionfile_main.o: $(CHIMERAFILE_DIR)/diffusionfile/diffusionfile.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_DIFFUSION_INCLUDES) \
		-DCOSMOCC=1 \
		-D_XOPEN_SOURCE=600 \
		-DCHIMERAFILE=1 \
		-DLLAMAFILE_VERSION_STRING="$(LLAMAFILE_VERSION_STRING)" \
		-Dmain=diffusionfile_main \
		-frtti \
		-c -o $@ $<

# Wrapper for stable-diffusion.cpp core (handles stb pre-inclusion)
# Uses CHIMERAFILE_DIFFUSION_CPPFLAGS which includes sd_ggml_compat.h
o/$(MODE)/chimerafile/sd_core.o: $(CHIMERAFILE_DIR)/diffusionfile/sd_core.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_DIFFUSION_CPPFLAGS) \
		-DGGML_MULTIPLATFORM \
		-frtti \
		-c -o $@ $<

# util.cpp — uses the OLD stb_image_resize.h API (different symbols from
# llamafile's stb_image_resize2.h), so no wrapper needed — no conflict.
o/$(MODE)/chimerafile/sd_util.o: stable-diffusion.cpp/util.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_DIFFUSION_CPPFLAGS) \
		-DGGML_MULTIPLATFORM \
		-frtti \
		-c -o $@ $<

# model.cpp — no stb issues (json.hpp comes from stable-diffusion.cpp/thirdparty)
o/$(MODE)/chimerafile/sd_model.o: stable-diffusion.cpp/model.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_DIFFUSION_CPPFLAGS) \
		-DGGML_MULTIPLATFORM \
		-frtti \
		-c -o $@ $<

# upscaler.cpp — no stb issues, compile directly
o/$(MODE)/chimerafile/sd_upscaler.o: stable-diffusion.cpp/upscaler.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CHIMERAFILE_DIFFUSION_CPPFLAGS) \
		-DGGML_MULTIPLATFORM \
		-frtti \
		-c -o $@ $<

# zip.c — needed by model.cpp for .ckpt (PyTorch checkpoint) files
o/$(MODE)/chimerafile/sd_zip.o: stable-diffusion.cpp/thirdparty/zip.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CHIMERAFILE_DIFFUSION_CPPFLAGS) \
		-DGGML_MULTIPLATFORM \
		-c -o $@ $<

# ==============================================================================
# Aggregate object list
# ==============================================================================

CHIMERAFILE_DIFFUSION_OBJS := \
	o/$(MODE)/chimerafile/diffusionfile_main.o \
	o/$(MODE)/chimerafile/sd_core.o \
	o/$(MODE)/chimerafile/sd_util.o \
	o/$(MODE)/chimerafile/sd_model.o \
	o/$(MODE)/chimerafile/sd_upscaler.o \
	o/$(MODE)/chimerafile/sd_zip.o

CHIMERAFILE_OWN_OBJS := \
	o/$(MODE)/chimerafile/chimerafile.o \
	o/$(MODE)/chimerafile/llama_dispatch.o \
	$(CHIMERAFILE_CHECK_CPU_OBJ) \
	o/$(MODE)/chimerafile/whisperfile_main.o \
	o/$(MODE)/chimerafile/slurp.o \
	o/$(MODE)/chimerafile/color.o \
	o/$(MODE)/chimerafile/common.cpp.o \
	o/$(MODE)/chimerafile/common-ggml.cpp.o \
	o/$(MODE)/chimerafile/common-whisper.cpp.o \
	o/$(MODE)/chimerafile/grammar-parser.cpp.o \
	o/$(MODE)/chimerafile/cli.chimerafile.cpp.o \
	o/$(MODE)/chimerafile/whisper.core.o

# All objects for the final link:
#   our objects + llamafile's objects (includes GGML, llama, server, TUI, TinyBLAS)
# whisper.cpp.a is NOT included — its GGML symbols conflict with GGML_OBJS.
CHIMERAFILE_ALL_OBJS := \
	$(CHIMERAFILE_OWN_OBJS) \
	$(CHIMERAFILE_DIFFUSION_OBJS) \
	o/$(MODE)/llamafile/server.cpp.o \
	$(LLAMAFILE_OBJS) \
	$(LLAMAFILE_DEPS) \
	$(SERVER_ASSETS) \
	o/$(MODE)/third_party/stb/stb.a

# ==============================================================================
# GPU backend shared libraries
# ==============================================================================
# These call the shell scripts from the llamafile GPU build system.
# The OUTPUT path is overridden so the .so lands in the build tree.

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

# GPU .so files — only added as prerequisites if they exist (via wildcard).
# This ensures the binary is re-bundled when backends are rebuilt.
CHIMERAFILE_GPU_SOS := $(wildcard o/$(MODE)/ggml-cuda.so o/$(MODE)/ggml-vulkan.so o/$(MODE)/ggml-rocm.so)

# ==============================================================================
# Final binary — with automatic GPU backend bundling when available
# ==============================================================================

o/$(MODE)/chimerafile/chimerafile: $(CHIMERAFILE_ALL_OBJS) $(CHIMERAFILE_GPU_SOS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $(filter %.o %.a,$^) $(LDLIBS) \
			-Wl,--allow-multiple-definition \
			-Wl,--wrap=ggml_backend_cpu_init
	@echo "==> Built: $@"
	@ZIPALIGN=o/$(MODE)/third_party/zipalign/zipalign; \
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
		"$$ZIPALIGN" -j0 "$(LLAMAFILE)/o/$(MODE)/chimerafile/chimerafile" *.so && \
		rm -rf "$$BUNDLE_DIR" && \
		echo "==> GPU backends embedded — binary has GPU support"; \
	elif [ "$$HAVE_GPU" = "0" ]; then \
		echo "==> No GPU backends found — binary runs CPU-only."; \
		echo "    Optional: build with 'make cuda' and/or 'make vulkan' then rebuild"; \
	fi

.PHONY: o/$(MODE)/chimerafile
o/$(MODE)/chimerafile: o/$(MODE)/chimerafile/chimerafile
