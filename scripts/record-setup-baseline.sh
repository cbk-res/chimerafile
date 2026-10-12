#!/bin/bash
# record-setup-baseline.sh
#
# Records the current state of the llamafile submodule trees as the expected
# post-setup baselines under setup-baseline/.
#
# 'make setup' patches the pinned submodule trees in place, so a correctly
# set up tree is dirty *by design*; these baselines capture exactly what that
# dirt looks like (content-hashed), so verify-setup.sh can detect drift.
#
# Runs automatically at the end of 'make setup' and 'make fresh'.  Run it
# standalone to bless a deliberate state change, then commit setup-baseline/.

set -eu

REPO=$(git rev-parse --show-toplevel)
L="$REPO/llamafile"
B="$REPO/setup-baseline"

[ -f "$L/Makefile" ] || {
	echo "error: llamafile submodule not found — run 'make setup' first" >&2
	exit 1
}

. "$REPO/scripts/setup-fingerprint.sh"

mkdir -p "$B"

for sub in llama.cpp whisper.cpp stable-diffusion.cpp transcribe.cpp; do
	if [ ! -d "$L/$sub" ]; then
		echo "error: $sub not initialised — run 'make setup' first" >&2
		exit 1
	fi
	cf_fingerprint "$L/$sub" > "$B/$sub.status"
done

# The llamafile submodule itself: no modified tracked files, no stray
# untracked files, nested submodules at their recorded pins.
cf_fingerprint "$L" > "$B/llamafile.status"

echo "Baselines recorded in $B/ :"
wc -l "$B"/*.status | sed 's/^/  /'
