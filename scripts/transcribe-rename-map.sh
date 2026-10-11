#!/bin/sh
# transcribe-rename-map.sh OUT_MAP TRANSCRIBE_A [REST...]
#
# Emit an objcopy --redefine-syms map renaming every globally defined symbol
# in TRANSCRIBE_A that is ALSO defined by one of the REST link inputs to
# trc_<name>.
#
# transcribe.cpp vendors its own ggml (0.15.2), which overlaps heavily with
# llama.cpp's GGML (0.20): both C symbols (ggml_*, gguf_*) and mangled C++
# symbols (ggml_quantize_mat_t<...>, gguf_read_emplace_helper<...>, ...).
# Renaming only the C prefixes leaves the C++ symbols to
# --allow-multiple-definition, which silently binds transcribe's code to
# llama.cpp's differently-laid-out helpers and corrupts memory at runtime.
# So we rename the full collision set, keeping the vendored ggml fully
# isolated.  objcopy --redefine-syms rewrites definitions AND references in
# every member, so internal consistency is preserved automatically.
set -eu

map="$1"; shift
trc_a="$1"; shift

tmp_mine="$(mktemp)"
tmp_rest="$(mktemp)"
trap 'rm -f "$tmp_mine" "$tmp_rest"' EXIT

nm -g --defined-only "$trc_a" \
    | awk '$2 ~ /^[TDBWV]/ && $3 != "" {print $3}' \
    | sort -u > "$tmp_mine"

for f in "$@"; do
    [ -e "$f" ] || continue
    nm -g --defined-only "$f" 2>/dev/null || true
done | awk '$2 ~ /^[TDBWV]/ && $3 != "" {print $3}' | sort -u > "$tmp_rest"

comm -12 "$tmp_mine" "$tmp_rest" | while IFS= read -r s; do
    printf '%s\ttrc_%s\n' "$s" "$s"
done > "$map"
