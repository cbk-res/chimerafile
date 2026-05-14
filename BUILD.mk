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
# Aggregate object list
# ==============================================================================

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
	o/$(MODE)/llamafile/server.cpp.o \
	$(LLAMAFILE_OBJS) \
	$(LLAMAFILE_DEPS) \
	$(SERVER_ASSETS)

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
	$(CXX) $(LDFLAGS) -o $@ $(filter %.o %.a,$^) $(LDLIBS)
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
		cd "$$BUNDLE_DIR" && "$$ZIPALIGN" -j0 "$$(cd $(@D) && pwd)/$$(@F)" *.so; \
		rm -rf "$$BUNDLE_DIR"; \
		echo "==> GPU backends embedded — binary has GPU support"; \
	elif [ "$$HAVE_GPU" = "0" ]; then \
		echo "==> No GPU backends found — binary runs CPU-only."; \
		echo "    Optional: build with 'make cuda' and/or 'make vulkan' then rebuild"; \
	fi

.PHONY: o/$(MODE)/chimerafile
o/$(MODE)/chimerafile: o/$(MODE)/chimerafile/chimerafile
