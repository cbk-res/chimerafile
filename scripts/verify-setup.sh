#!/bin/bash
# verify-setup.sh
#
# Verifies that the llamafile submodule trees match the expected pinned +
# setup state recorded in setup-baseline/.  Non-destructive: it only reads
# git state and reports drift (stray edits, stray files, moved checkouts —
# including edits to files that setup already patches).
#
# Runs automatically before every build (see the root Makefile).
# Standalone:  make verify-setup
#
# If this fails:
#   - accidental drift:        make fresh    (reset to pins, re-apply setup, rebuild)
#   - deliberate, permanent
#     change (new pin, new     make setup    then commit setup-baseline/
#     patches):

set -u

REPO=$(git rev-parse --show-toplevel)
L="$REPO/llamafile"
B="$REPO/setup-baseline"
rc=0

. "$REPO/scripts/setup-fingerprint.sh"

fail() {
	printf '%s\n' "$*"
	rc=1
}

if [ ! -f "$L/Makefile" ]; then
	fail "ERROR: llamafile submodule not initialised — run: make setup"
	exit 1
fi

check_tree() {
	# $1 = dir, $2 = baseline file, $3 = label
	local dir=$1 base=$2 label=$3 cur
	if [ ! -f "$base" ]; then
		fail "ERROR: missing $base — run 'make setup' (or 'make fresh') and commit setup-baseline/"
		return
	fi
	if [ ! -d "$dir" ]; then
		fail "ERROR: $label is missing — run: make fresh (or: make setup)"
		return
	fi
	cur=$(cf_fingerprint "$dir")
	if [ "$cur" != "$(cat "$base")" ]; then
		fail "ERROR: $label differs from its baseline:"
		diff -u "$base" <(printf '%s\n' "$cur") | sed 's/^/    /'
	fi
}

# 1. The llamafile submodule itself (tracked files + stray untracked files +
#    nested submodule HEADs via the gitlink lines).
check_tree "$L" "$B/llamafile.status" "llamafile submodule"

# 2. Each nested submodule's working tree (the setup-patched dirt).
for sub in llama.cpp whisper.cpp stable-diffusion.cpp transcribe.cpp; do
	check_tree "$L/$sub" "$B/$sub.status" "$sub"

	# Explicit HEAD check: a moved checkout is the clearest kind of drift.
	recorded=$(git -C "$L" ls-tree HEAD "$sub" | awk '{print $3}')
	head=$(git -C "$L/$sub" rev-parse HEAD 2>/dev/null || echo missing)
	if [ "$recorded" != "$head" ]; then
		fail "ERROR: $sub is at ${head:0:12} but the llamafile pin records ${recorded:0:12}"
	fi
done

if [ "$rc" -ne 0 ]; then
	echo
	echo "The llamafile trees do not match the expected pinned + setup state."
	echo "  accidental drift?   restore the expected state:   make fresh"
	echo "  deliberate change?  re-setup first:               make setup"
	echo "                    (then: git add setup-baseline && git commit)"
	exit 1
fi
echo "setup state OK — llamafile trees match setup-baseline/"
