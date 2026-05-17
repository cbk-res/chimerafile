#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# Chimerafile root Makefile
#
# All real build work runs inside the llamafile/ submodule directory.
# We inject our BUILD.mk using -f rather than MAKEFILES because
# MAKEFILES is read BEFORE the primary makefile — which means
# llamafile's variables (LLAMAFILE_INCLUDES, LLAMAFILE_OBJS, etc.)
# haven't been set yet.  -f reads the additional makefile AFTER.
#

SHELL = /bin/sh

# Absolute path to this repo's root
CHIMERAFILE_REPO := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Absolute path to llamafile submodule
LLAMAFILE := $(CHIMERAFILE_REPO)/llamafile

# Sanity check
ifeq ($(wildcard $(LLAMAFILE)/Makefile),)
$(error llamafile submodule not found at $(LLAMAFILE). Run 'make setup' first.)
endif

MODE ?=

# ── Build-time toggles (forwarded to submake) ───────────────────────
CHIMERAFILE_WITH_WHISPER   ?= 1
CHIMERAFILE_WITH_DIFFUSION ?= 1
CHIMERAFILE_WITH_CUDA      ?= 1
CHIMERAFILE_WITH_VULKAN    ?= 1
CHIMERAFILE_WITH_ROCM      ?= 1

# Invoke make inside llamafile/, reading both Makefile and our BUILD.mk.
SUBMAKE = $(MAKE) -C $(LLAMAFILE) \
	-f $(LLAMAFILE)/Makefile \
	-f $(CHIMERAFILE_REPO)/BUILD.mk \
	CHIMERAFILE_DIR=$(CHIMERAFILE_REPO) \
	MODE=$(MODE) \
	CHIMERAFILE_WITH_WHISPER=$(CHIMERAFILE_WITH_WHISPER) \
	CHIMERAFILE_WITH_DIFFUSION=$(CHIMERAFILE_WITH_DIFFUSION)

.PHONY: all
all:
	@echo "==> Attempting GPU backend builds (optional)..."
	@if [ "$(CHIMERAFILE_WITH_CUDA)" = "1" ]; then \
		cd $(LLAMAFILE) && OUTPUT="$(LLAMAFILE)/o/$(MODE)/ggml-cuda.so" \
			bash llamafile/cuda.sh >/dev/null 2>&1 || true; \
		if [ -f "$(LLAMAFILE)/o/$(MODE)/ggml-cuda.so" ]; then \
			echo "   CUDA backend built"; \
		fi; \
	fi
	@if [ "$(CHIMERAFILE_WITH_VULKAN)" = "1" ]; then \
		cd $(LLAMAFILE) && OUTPUT="$(LLAMAFILE)/o/$(MODE)/ggml-vulkan.so" \
			bash llamafile/vulkan.sh >/dev/null 2>&1 || true; \
		if [ -f "$(LLAMAFILE)/o/$(MODE)/ggml-vulkan.so" ]; then \
			echo "   Vulkan backend built"; \
		fi; \
	fi
	@if [ "$(CHIMERAFILE_WITH_ROCM)" = "1" ]; then \
		cd $(LLAMAFILE) && OUTPUT="$(LLAMAFILE)/o/$(MODE)/ggml-rocm.so" \
			bash llamafile/rocm.sh >/dev/null 2>&1 || true; \
		if [ -f "$(LLAMAFILE)/o/$(MODE)/ggml-rocm.so" ]; then \
			echo "   ROCm backend built"; \
		fi; \
	fi
	@echo "==> Building chimerafile (bundling any GPU backends found)..."
	$(SUBMAKE) o/$(MODE)/chimerafile/chimerafile

.PHONY: setup
setup:
	@echo "==> Initialising llamafile submodule..."
	@cd $(CHIMERAFILE_REPO) && \
		if git submodule status llamafile >/dev/null 2>&1; then \
			git submodule update --init --recursive llamafile; \
		else \
			echo "   (submodule already cloned, skipping)"; \
		fi
	@echo "==> Running llamafile setup (cosmocc, patches, nested submodules)..."
	$(MAKE) -C $(LLAMAFILE) setup
	@echo ""
	@echo "==> Setup complete. Run: make"

PREFIX ?= /usr/local

.PHONY: install
install: all
	@mkdir -p $(PREFIX)/bin
	install $(LLAMAFILE)/o/$(MODE)/chimerafile/chimerafile $(PREFIX)/bin/chimerafile
	@echo "Installed chimerafile -> $(PREFIX)/bin/chimerafile"

.PHONY: bundle
bundle:
	@echo "==> Bundle build: GPU backends + chimerafile"
	@if [ "$(CHIMERAFILE_WITH_CUDA)" = "1" ]; then \
		cd $(LLAMAFILE) && OUTPUT="$(LLAMAFILE)/o/$(MODE)/ggml-cuda.so" \
			bash llamafile/cuda.sh 2>&1 | sed 's/^/   /' || true; \
	fi
	@if [ "$(CHIMERAFILE_WITH_VULKAN)" = "1" ]; then \
		cd $(LLAMAFILE) && OUTPUT="$(LLAMAFILE)/o/$(MODE)/ggml-vulkan.so" \
			bash llamafile/vulkan.sh 2>&1 | sed 's/^/   /' || true; \
	fi
	$(SUBMAKE) o/$(MODE)/chimerafile/chimerafile
	@echo "==> Bundle complete. GPU backends are embedded in the binary."

.PHONY: clean
clean:
	$(MAKE) -C $(LLAMAFILE) clean

.PHONY: distclean
distclean:
	$(MAKE) -C $(LLAMAFILE) distclean

%:
	$(SUBMAKE) $@
