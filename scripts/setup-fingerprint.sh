# setup-fingerprint.sh
#
# Sourced by record-setup-baseline.sh and verify-setup.sh — do not run directly.
#
# cf_fingerprint DIR
#   Prints a content-sensitive fingerprint of the git working tree at DIR:
#   one line per path whose on-disk state differs from DIR's HEAD —
#     <blob-hash>  <path>   file exists on disk (tracked-modified or untracked)
#     deleted      <path>   tracked file removed from disk
#     gitlink <sha> <path>  submodule (sha = the submodule's current HEAD)
#   Sorted with LC_ALL=C.  Empty output means the tree is clean.
#
# Content hashes (git hash-object) rather than plain status lines, so a stray
# edit to an *already-modified* file is detected too.

# Read-only git call with retry on transient index.lock collisions (the
# editor/IDE periodically runs git status in the tree concurrently).
_git_ro() {
	local i=0
	local err
	err=$(mktemp)
	while :; do
		if git "$@" 2>"$err"; then
			rm -f "$err"
			return 0
		fi
		if grep -q "index.lock" "$err" && [ $i -lt 20 ]; then
			i=$((i+1))
			sleep 0.5
			continue
		fi
		cat "$err" >&2
		rm -f "$err"
		return 1
	done
}

cf_fingerprint() {
	local dir=$1 p mode
	{
		_git_ro -C "$dir" diff HEAD --name-only
		_git_ro -C "$dir" ls-files --others --exclude-standard
	} | LC_ALL=C sort -u | while IFS= read -r p; do
		[ -n "$p" ] || continue
		mode=$(_git_ro -C "$dir" ls-files --stage -- "$p" 2>/dev/null | awk '{print $1}')
		if [ "$mode" = "160000" ]; then
			printf 'gitlink %s\t%s\n' \
				"$(_git_ro -C "$dir/$p" rev-parse HEAD 2>/dev/null || echo missing)" "$p"
		elif [ -f "$dir/$p" ]; then
			printf '%s\t%s\n' "$(_git_ro -C "$dir" hash-object -- "$p")" "$p"
		else
			printf 'deleted\t%s\n' "$p"
		fi
	done | LC_ALL=C sort
}
