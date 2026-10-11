#!/bin/sh
# Workaround for cosmo_dlopen on SELinux-enforcing Fedora.
#
# cosmopolitan's cosmo_dlopen (Linux) compiles a tiny "dlopen-helper" PIE,
# maps it together with the host ld-linux into the process, and runs the
# host dynamic loader in-process. The helper is linked against
# /lib64/ld-linux-x86-64.so.2 (SELinux type ld_so_t).
#
# cosmopolitan's elf_map() writes (bzero) the file-backed VMA before
# mprotect(PROT_EXEC); the write COW-forks the tail page, attaching
# an anon_vma. SELinux's file_mprotect hook then requires FILE__EXECMOD
# (not plain FILE__EXEC) on the backing file type. Fedora's policy does
# not grant unconfined_t execmod on ld_so_t -> mprotect returns EACCES
# -> cosmo_dlopen reports "dlopen() isn't supported on this platform".
#
# Fix: relocate ld-linux into $HOME (type user_home_t, where unconfined_t
# DOES have execmod) and build the helper with PT_INTERP pointing there.
# cosmo reuses an existing helper that is newer than the APE, so we stamp
# the helper with a far-future mtime.
#
# Run this after (re)building the APE if you ever see the fatal error:
#   fatal error: ... Vulkan is not usable on this system
#
# Requirements: cc (any system C compiler). No root needed.
set -e

H="$HOME"
DIR="$H/.cosmolk"
LD_DEST="$DIR/ld-linux-x86-64.so.2"
mkdir -p "$DIR"
cp -f /lib64/ld-linux-x86-64.so.2 "$LD_DEST"

TMPD=$(mktemp -d)
cat > "$TMPD/helper.c" <<'EOS'
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv, char **envp) {
  char *ep;
  long addr;
  if (argc != 2) {
    fprintf(stderr, "%s: not intended to be run directly\n", argv[0]);
    return 1;
  }
  addr = strtol(argv[1], &ep, 10);
  if (*ep) {
    fprintf(stderr, "%s: invalid function address\n", argv[0]);
    return 2;
  }
  return ((int (*)(void *))addr)((void *[]){
      dlopen,
      dlsym,
      dlclose,
      dlerror,
  });
}
EOS

cc -pie -fPIC "$TMPD/helper.c" -o "$TMPD/helper" \
   -Wl,-z,execstack -Wl,--dynamic-linker="$LD_DEST" -ldl

mkdir -p "$H/.cosmo"
cp -f "$TMPD/helper" "$H/.cosmo/dlopen-helper"
chmod 711 "$H/.cosmo/dlopen-helper"
touch -d "2027-01-01 00:00:00" "$H/.cosmo/dlopen-helper"
rm -rf "$TMPD"
echo "Installed relocated dlopen-helper (PT_INTERP -> $LD_DEST)"
