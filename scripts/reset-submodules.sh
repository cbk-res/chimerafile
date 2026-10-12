#!/bin/bash
# reset-submodules.sh — reset the llamafile submodule trees to their pinned state.
#
# Resets the llamafile submodule to the commit recorded by the chimerafile
# repo, resets each nested submodule to its recorded commit, and removes
# untracked/ignored files from all submodule working trees.  Run 'make setup'
# (or simply 'make fresh', which does all of this plus a rebuild) afterwards
# to re-apply the setup patches.
#
# Every git command is retried on transient index.lock collisions — the
# editor/IDE periodically runs git status in the tree concurrently, and a
# millisecond-wide lock is otherwise a moving target.

set -u

REPO=$(git rev-parse --show-toplevel)
L="$REPO/llamafile"
ERR=$(mktemp)
trap 'rm -f "$ERR"' EXIT

git_retry() {
	# git_retry LABEL CMD...
	local label=$1; shift
	local i=0
	while :; do
		if "$@" 2>"$ERR"; then
			return 0
		fi
		if grep -q "index.lock" "$ERR" && [ $i -lt 30 ]; then
			i=$((i+1))
			echo "   $label: index.lock busy — retrying ($i/30)..."
			sleep 0.5
			continue
		fi
		echo "error: $label failed:" >&2
		cat "$ERR" >&2
		return 1
	done
}

git_retry "reset llamafile to its pinned commit" \
	git -C "$REPO" submodule update --init --force llamafile || exit 1

git_retry "reset nested submodules to their pinned commits" \
	git -C "$L" submodule update --init --force --recursive || exit 1

git_retry "remove untracked/ignored files from the submodule trees" \
	git -C "$L" submodule foreach --recursive 'git clean -fdx' || exit 1

echo "Submodule trees reset to the pinned state."
