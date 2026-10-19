#!/bin/bash
# Patch llama.cpp's router so model children can spawn under the APE runtime
# (see ci-patches/router-child.patch).
#
# The router spawns children as '<bin_path> <args>' where bin_path is
# /proc/self/exe — which for an APE is the small shared runtime (~/.ape-1.10),
# not a self-contained binary. The runtime resolves the payload from the APE
# file passed as its first argument, and the 'llama' subcommand dispatches to
# combined chat+server mode unless --server is given. The patch reads the APE
# path from LLAMA_APP_PATH (exported by the front router) and inserts the path
# plus --server into the child command:
#   <runtime> <ape> llama --server --host ... --port ... -m ...
#
# 'make setup' calls this after llamafile's own setup patches are in (see the
# root Makefile). Recorded in setup-baseline/.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/llamafile/llama.cpp"
PATCH="$ROOT/ci-patches/router-child.patch"

[ -d "$TARGET" ] || { echo "ERROR: $TARGET not found (run 'make setup' first)"; exit 1; }
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }

if grep -q "LLAMA_APP_PATH" "$TARGET/tools/server/server-models.cpp"; then
    echo "llama.cpp already patched for router child spawn, skipping."
    exit 0
fi

git -C "$TARGET" apply "$PATCH"

echo "llama.cpp patched: router child spawn for APE runtimes (LLAMA_APP_PATH + --server)."
