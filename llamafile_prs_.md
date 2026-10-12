# Upstream PRs / issues to file (chimerafile → llamafile / cosmopolitan)

Working notes for issues discovered while building Chimerafile (llama + whisper +
diffusion + transcribe in one APE). These are real upstream bugs that we have
workarounds for locally; file them when we have a cycle.

---

## 1. `cosmo_dlopen` in-process host-glibc heap collides with the APE heap arena

**Severity:** high (silent corruption → segfault) · **Affects:** any APE that
`cosmo_dlopen`s a host `.so` (Vulkan/CUDA/ROCm backends) and then does sustained
host-side allocation (diffusion is the heaviest case).

### Symptom
`--gpu vulkan diffusion` completes generation (`generate_image completed …`)
then crashes writing the output PNG:

```
error: Uncaught SIGSEGV (SEGV_MAPERR) at 0x8 on <host>
```

llama/whisper on Vulkan are fine because their host-side allocation is small;
diffusion (2.7 GB SD model + driver churn) tips it over. CPU-only is unaffected
(no host `.so` is loaded, so no second heap exists).

### Root cause
`cosmo_dlopen` runs the host `ld-linux` **inside** the APE process. The host
glibc then initialises its **brk-based** main-arena heap from the process `brk`
— which sits at the base of the APE's own reserved heap arena (cosmo APEs have
`bss=0`; the ~30 GB heap region is runtime-reserved starting exactly where the
kernel `brk` cursor begins, right after the data segment).

Under sustained allocation the two independent heap managers (cosmo's arena
allocator + host glibc's brk allocator) hand out memory from the **same region**
and clobber each other's live chunks. The corruption hit the in-process loader's
`link_map`, so the next **lazy PLT binding** crashed in the host loader's
`_dl_fixup` (NULL search-list pointer).

### Evidence (from the core dump)
- Faulting RIP resolved (via `NT_FILE` + `readelf`/`objdump`) to **`_dl_fixup`**
  in the in-process `ld-linux`, at `mov 0x8(%rax)` with `rax == NULL`.
- The `link_map` argument (in `rdi`) had garbage fields (`l_name == NULL`,
  `l_addr == 0x8`) — i.e. a live-but-corrupted structure.
- That `link_map` lived deep inside the APE's reserved heap arena, well past the
  data segment — the region both heaps share.
- `size` on the APE: `bss = 0`, confirming the arena is runtime-reserved, not
  a link-time BSS, and starts at the `brk` position.

### Local workaround (what chimerafile ships today)
After a GPU backend loads, the dispatcher calls the **host** libc's
`mallopt(M_MMAP_THRESHOLD, 0)` (via `cosmo_dlopen("libc.so.6")` +
`cosmo_dlsym("mallopt")`). This forces every host-glibc allocation through
`mmap` (high addresses) instead of `brk`, so the host heap stays out of the
shared region. Verified: diffusion-Vulkan saves PNGs; llama/whisper unaffected.
See `chimerafile.cpp` → `isolate_host_glibc_heap()`.

### Where to file
- **Cosmopolitan (root cause):** the in-process foreign loader should not let the
  host glibc's `brk` heap grow into the APE's reserved arena. Proper fix is in
  cosmo's `dlopen`/foreign-helper setup (e.g. give the in-process glibc a private
  heap base, or pre-configure it to mmap). Attach the core-dump trail above.
- **llamafile (defensive workaround):** a small PR to call
  `mallopt(M_MMAP_THRESHOLD, 0)` on the host libc right after a successful
  `cosmo_dlopen` in `gpu_backend.c`. Benefits **all** llamafile GPU backends
  (CUDA/ROCm also use `cosmo_dlopen`), not just diffusion. Low risk.

### Repro
1. Build a llamafile (or chimerafile) APE with the Vulkan backend bundled.
2. `./ape --gpu vulkan diffusion -M img_gen -m sd15.safetensors -p "a cat" -o x.png`
   on an Intel/AMD iGPU (any UMA Vulkan device).
3. Crash at PNG save on the unpatched build; succeeds with the `mallopt` fix.
