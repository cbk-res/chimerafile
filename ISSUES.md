# Chimerafile issue ledger

Working ledger of every known issue on the live mellum serve (127.0.0.1:18080) and the
opencode-on-Mellum experience. Updated as things are confirmed or fixed. Statuses:
OPEN / IN-PROGRESS / FIXED / INFO.

Production build: checkpoint-fix era (81428a4), invocation
37cc2fe1bd024b42ba4a5aceee10db18, deployed 2026-10-23 23:54:25.
Pinned llama.cpp: `0177dcc` (submodule). Pinned tree changes only via `ci-patches/`.

---

## 1. Memory-corruption crash family (OPEN - C3 fixed, C2/C4 under watch)

Four instance crashes, all SIGSEGV/SIGABRT, all heap corruption surfacing in different places.
The corruption is not in one function; it is wild heap damage that crashes whichever
operation touches the damaged memory next.

| # | Time (Oct) | PID | Signal | Surface point (resolved via APE dbg ELF) |
|---|-----------|-----|--------|------------------------------------------|
| C1 | 22, 23:42:29 | 44712 | SEGV | `seq_pos` set header corrupted (root pointer = 0x9). Original incident. Core still present (351 MB) but pre-watchdog build. |
| C2 | 23, 20:22:46 | 251694 | SEGV | `server_slot::prompt_clear -> llama_kv_cache::seq_rm -> llama_kv_cells::seq_rm -> std::__tree::__erase_unique -> __tree_remove`. Bad node pointer mid-operation. Triggered by my probe (new prompt reusing a slot). |
| C3 | 23, 20:35:19 | 252834 | ABRT | **FIXED (2026-10-23, commit 81428a4).** `common_prompt_checkpoint::load_tgt` -> GGML_ABORT "checkpoint size mismatch: expected %zu, got %zu". State restore threw mid-stream ("failed to restore kv cache"); partial byte count tripped the assert. Trigger: LCP partial reuse (f_keep=0.622). Fixed by upstream f174553 + e467e5b (section 2); live C3-shape probe on the new build: all 200, LCP path fired (f_sim 0.995, f_keep 0.973 on the probe), no abort, no core. |
| C4 | 23, 21:33:31 | 260480 | SEGV | **New.** HTTP httplib worker thread, destroying a `nlohmann::json` in a POST route handler: `json_value::destroy -> std::string::~string -> free -> __mspace_free` (corrupted heap chunk). Full 26-frame stack resolved (core 612 MB, `/tmp/opencode/core-segv-260480`). Killed the instance mid-conversation - this is the crash that stalled the skills-config chat at 21:33 (4-min reload gap) and caused the 36k-token full re-prefill at 21:42. |
| C5 | 24, 00:26:56 | 297621 | SEGV | **Recurrence of the C2 shape on the checkpoint-fix build - RESOLVED as OOM-induced** (headroom re-run 2026-10-24 11:23 passed clean, no core, zero OOM). GPU-lane run, test server workdir 297572. `server_context_impl::pre_decode` -> `common_context_seq_rm` -> `llama_kv_cache_iswa::seq_rm` -> `llama_kv_cache::seq_rm` -> `llama_kv_cells::seq_rm` -> `std::__tree::__erase_unique` -> `__tree_remove` (bad node pointer, same surface as C2). Trigger: a new request (f_sim 0.111, f_keep 0.000) landed on the slot left by the 25-turn LCP scenario - a single ~15k-cell truncation down to ~4 tokens. Core 268 MB, `/tmp/opencode/core-segv-297621`. No checkpoint involvement in the stack. |

### Analysis so far

- kv-cells seq_pos patch (map->set rewrite, `ci-patches/kv-cells-seqpos.patch`):
  **static logic cleared**. Every mutator (seq_rm/seq_keep/seq_add/rm/set/cp, pos_add,
  pos_div, apply_ubatch overwrite, state_read restore) maintains seq_pos consistency:
  re-key via seq_pos_rm+seq_pos_add on position change; rm() does seq_pos_rm before
  clearing; allocation paths call pos_set before seq_add. No stale-entry path found.
- C4 moves the target: the crash was on an **HTTP thread** destroying a JSON object,
  far from the KV cache. Either the same wild writer damaged a JSON heap chunk, or a
  JSON object was double-destroyed (use-after-free of the json itself).
- Leading suspects, in order:
  1. Data race on shared state (llama_kv_cache / slots / json bodies) between the server
     loop thread, HTTP threadpool, and/or the llama threadpool (n_threads=8).
  2. Wild write from a third-party path (Vulkan backend callbacks, APE runtime).
  3. For C3 specifically: also a plausible *logic* bug in the checkpoint/SWA interaction
     (see section 2 - upstream already fixed one such bug). **CONFIRMED for C3**: the
     stale-checkpoint restore was exactly the upstream bug class; fixed and live-verified
     2026-10-23 (section 2). C1/C2/C4 remain the open wild-write question (C5 resolved
     as OOM-induced, 2026-10-24 11:23 headroom re-run).

### C5 (2026-10-24 00:26) - RESOLVED: OOM-induced, not a corruption recurrence

- The checkpoint-fix build crashed on the GPU-lane stress run: the 25-turn LCP scenario
  (small tail truncation every turn) passed, then a new request landed on the surviving
  slot (f_sim 0.111, f_keep 0.000) and `pre_decode`'s single ~15k-cell truncation walked
  into a corrupted `seq_pos` tree node (`__tree_remove`). Same surface as C2, different
  entry point (slot switch, not prompt_clear).
- The checkpoint port is **not** a suspect: for Mellum (iSWA) the old iswa code gated
  the base cache under PARTIAL_ONLY and the new code makes the plain base cache
  early-return - behaviorally identical; f174553 only changes which checkpoints get
  invalidated. Neither touches the seq_rm/seq_pos path.
- Only one task was active at the moment (no decode in flight, no draft context), so a
  racing llama thread is less likely here than for C4 - but the corruption can accumulate
  earlier in the run and surface on the first tree walk that touches the damaged node;
  the boundary-only watchdog gives no timestamp for the damage.
- **OOM alternative (raised by user, partly supported):** the machine has 23 GiB RAM +
  8 GiB swap. The GPU-lane run loaded a *second* full Mellum (model + 131k KV) on top of
  the live instance + desktop + browser - structurally over budget. The kernel's **global
  OOM killer fired at 00:13:06** (invoked by the APE runtime, pid 297627) and **killed
  the user's Brave browser** (the death the user noticed). C5 landed 13 min later, while
  the system was still swap-thrashing, with no further OOM invocations. APE cores always
  store si_code=0/fault_addr=0 (verified on every core, incl. C4), so the fault address
  cannot be read from the core to tell NULL-deref (plausible if the runtime allocator
  fails to NULL under pressure) from a wild write.
- **Discriminator run (2026-10-24 11:23, post-reboot): PASSED CLEAN.** Full matrix
  exit 0; mellum lane (LCP 25-turn scenario + checkpoint test) 4 passed in 535s,
  including the exact C5 trigger shape - a new low-similarity request on the LCP
  slot (single ~15k-cell truncation via seq_rm). Live Mellum never loaded (18 GiB
  headroom), zero OOM invocations, no core. **Verdict: C5 was OOM-induced** - an
  allocation failure under the 00:13 OOM / sustained swap thrash (most plausibly
  the runtime allocator failing to NULL), not a confirmed wild-corruption
  recurrence. Caveat: a probabilistic wild write might simply not have re-fired;
  the corruption family stays open on C1/C2/C4 (all on the single-instance live
  service, where OOM was not a factor) - C5 drops out of that set.
- Repeatable stress vehicle: the GPU-lane mellum suite (LCP + checkpoint scenarios,
  ~15 min). ASan on that workload is now the cheap definitive path (held on fan noise -
  needs user OK).
- **Only ASan can pin a wild write definitively.** ASan soak is HELD by user (fan noise on
  iGPU box) - needs explicit OK before relaunching.

### Watchdog limitation (known)

The seq_pos watchdog (commit `71b1834`, live-armed on all 3 instances) checks set headers
at init/dtor only - **boundary-only trap**. It cannot catch mid-operation corruption
(C2, C4) and it never fired. Keep it (cheap, catches the C1-style header damage), but do
not count on it.

---

## 2. Prompt-checkpoint feature bugs (FIXED - f174553 + e467e5b ported 2026-10-23, commit 81428a4)

Production runs the checkpoint path: `serve-mellum.json` does not set `n_ctx_checkpoints`,
and the default is **32 per slot** (`common/common.h:613`). So every partial-reuse
request (f_keep < 1.0 - i.e. every new conversation while an old one's KV lingers, and
every interrupt+resend) walks the LCP + checkpoint-restore path.

Upstream commits after our pin (`0177dcc`) that fix this neighborhood:

| Upstream commit | What it fixes | Status |
|-----------------|---------------|--------|
| `f174553` "server: fix swa checkpoint invalidation logic (#30666)" | Checkpoints with `n_tokens > n_prefix` (tokens not shared with the new prompt) were kept and restored. Fix: invalidate them, using `n_prefix = n_past` captured **before** any checkpoint restore. Ships a failing-then-passing test (`test_completion_checkpoint_after_unrelated_prompt`). | **PORTED** as `ci-patches/ckpt-invalidate.patch` (applies clean, includes the upstream test in `tools/server/tests/unit/test_completion.py`). Direct fix for the C3 class. |
| `e467e5b` "kv-cache: honor PARTIAL_ONLY in the plain KV cache (#30514)" | Under `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (the flag checkpoint load uses), the plain cache serialized every cell and the iSWA wrapper gated its base cache inconsistently. Fix: each cache applies the flag itself; iswa delegates. | **PORTED** as `ci-patches/kv-partial-only.patch` (the `dsa-iswa` and `memory-hybrid` parts are not ported: those files do not exist at the pin, and Mellum is iSWA, not hybrid-recurrent). |
| `2639a3b` "server: save only restorable slot states to the prompt cache (#30498)" | Idle-slot save loop stored non-restorable states (rerank/embedding/decision/infill), filling the prompt cache with entries that can never be read back. | **DEFERRED**: no decision models, and no embedding/rerank tasks, at the pin; production is chat-only, so the bug has no live surface. Revisit if mellum ever serves embedding/rerank. |
| `23b0202` "server: leave a busy slot untouched when a request pins it (#30295)" | A request that pins a slot could disturb a busy slot. Also touches `llama-kv-cells.h`. | **DEFERRED**: merge commit whose parent is missing from the shallow store (cannot diff against parent); touches `llama-kv-cells.h`, which our seq_pos patch rewrites - revisit once the parent is fetched and the interaction is checkable. |

Verification of the ported pair (commit 81428a4):
- CPU test matrix: all suites passed (58 tests, ~4 min), incl. the new
  `tests/test_checkpoint_reuse.py` (CPU lane, mechanism pin: growing conversation +
  divergent probe, child-pid stable, LCP path proven to fire from the router log).
- Live (deployed 23:54, instance child 291120): the C3 shape - 3 truncated turns on
  one slot, then a system-shared divergent probe - all 200; router log shows the probe
  taken by LCP similarity (f_sim 0.995, f_keep 0.973) against the conversation slot, i.e.
  through the checkpoint lookup/restore path; no abort, no new core in coredumpctl.
- The GPU-lane run of `test_checkpoint_reuse.py` on `lcp-mellum.json` (real SWA
  checkpoint data, ~1 min extra) is available as the standing regression gate; not yet
  run (fan noise, user call).

---

## 3. Mellum 2.1 model-competence issues (OPEN - not server bugs)

Evidence: opencode session DB, session `ses_e959297bfffei6RR2qfUDEUOyg`
("Opencode skills configuration", 21:21-21:45, model chimerafile-serve/mellum high).

- **S1. Code Mode tool-confusion spiral.** The user asked Mellum to install 3 skill repos
  (android/skills, Kotlin/kotlin-agent-skills, TheQtCompanyRnD/agent-skills; answer:
  "Install all three"). Mellum correctly called the direct `skill` tool once
  (`skill id=opencode` -> succeeded), then to run any shell command it kept routing
  through `execute` (Code Mode) - where only the arxiv+opencode MCP catalog exists:
  `tools.shell(...)`, `tools.execute(...)`, `tools.read(...)`, `glob(...)`,
  `tools.subagent(...)`, even `1+1` - ~20 calls over 15 minutes, all "Unknown tool" /
  ReferenceError, zero progress. Its own final reasoning: "couldn't execute shell,
  missing tools, maybe configuration issue" - it told the user it was an environment
  problem. It is not: the direct `shell`/`read`/`glob` tools were available the whole
  time (direct `read` calls succeeded in the same session). A 12B model misreading the
  Code Mode scope in the system prompt.
- **S2. worktree_create failures.** Session directory was `/var/home/cbk8650` (home, not
  a git repo) -> "Unable to create worktree test/skills-repo". Expected; not a bug.
- **S3. (earlier, 20:14-20:15 window) ReferenceError 'glob'/'read' in execute** - same
  root cause as S1.

No server-side action. If it keeps biting, options are a model swap for config-type
tasks or a stronger local model; document for the user.

- **S4. (RESOLVED, verified) question-tool drop.** v0.4.1 (`e89aed6`) dropped tool
  schemas with doubly-nested arrays of objects - the question tool shape - because the
  cosmo build corrupted the child in the schema-to-grammar path. Root cause fixed in
  `ccbc66e` (stack overflow in `common_peg_arena::save()`, 80 KiB cosmo worker stack;
  to_json now built with moves, schema JSON no longer serialized), and the drop
  workaround was removed (`DROP_NESTED_ARRAY` absent from the pinned tree). Live
  verification: question tool active - 43 successful calls in the sdcpp-verify session
  + 1 in the skills chat (21:24); the 11 "errors" are user dismissals/interrupts, not
  failures. NOT related to the S1 skill confusion.

---

## 4. Interrupt / prefill behavior (INFO - confirmed, by design)

User observation: "when I interrupt the model, it prefills from scratch when I send the
next message?"

Confirmed with a nuance, from the live instance log (router 248246, instance 54135,
child-time offsets; instance started 21:33):

- 21:43:00 (child 10.29): user interrupt - `http client error: Connection handling
  canceled` -> `cancel task, id_task = 3143` 4.5 s after start. The task was
  **mid-prompt** (only 534 of a ~7k-token prompt processed; no generation started).
- 21:44 (child 10.35): next message -> `f_sim_best = 0.954, f_keep = 0.933`, prompt eval
  **39 s / 6477 tokens** - i.e. everything after the interrupt point re-prefilled.
- 21:42 (child 9.56): after the 21:33 crash (C4) the instance had no KV at all, so the
  next request re-prefilled the **entire** ~36k-token conversation (179 s / 36,376
  tokens).

Mechanics (`server-context.cpp`): slot selection picks max LCP
(`tokens.get_common_prefix(task.tokens)`, line 3123); `n_past` = that LCP, clamped to
what the KV actually holds. Interrupt mid-prompt => KV holds only the partial prefix =>
next message re-prefills from the interrupt point. Not from scratch, but close enough to
feel like it. Crashes make it literally from scratch (KV gone).

---

## 5. Operational notes (INFO)

- **Unpushed commits** (await user OK): `e689a88` (kv-cells seq_pos), `231ade0`,
  `ccbc66e` (nested-tool), `a6baeec` (img2img), `71b1834` (watchdog),
  `81428a4` (checkpoint fix + test, deployed 2026-10-23 23:54).
- **ASan soak HELD** by user (fan noise). Do not relaunch without explicit OK.
- **router.log forensics caveat:** instance log lines arrive in flush bursts and their
  child-time offsets do not interleave cleanly with router timestamps (the [48087]
  series contains content generated 11 min after the "exit" that precedes it in the
  file). Trust coredumpctl timestamps + session DB over log-file ordering.
- APE core forensics: gdb cannot unwind these cores (APE runtime notes, NT_PRSTATUS
  descsz=336). Workflow that works: `coredumpctl info <pid>` stack trace +
  `/tmp/opencode/resolve.py` / `syms.pkl` (74,276 symbols from
  `chimerafile.new.com.dbg`; process addrs == ELF vaddrs, .text 0x401000-0x29E910d).
- coredumpctl inventory (Oct 22-23): 23:42:29 C1; 03:51/07:39/07:59 old-build SEGVs
  (unresolvable, pre-patch); 19:28:44 pid 244739 (my startup-crash test); 20:22:45 C2;
  20:35:19 C3; 21:33:31 C4. No cores since the 23:54 deploy of the checkpoint-fix build.
- Current instance: serve parent 290708 (started 2026-10-23 23:54), model child 291120,
  port 45085. Note: the router unloads idle model instances (clean "exit status 0", not
  a crash) and respawns them on the next request - an unloaded model is normal.

---

## 6. Cleanup plan (ordered)

1. ~~Port the upstream checkpoint fixes~~ **DONE (2026-10-23, 81428a4)**: `f174553` +
   `e467e5b` ported as ci-patches, built, CPU matrix green, deployed via the established
   restart (23:54). `2639a3b` + `23b0202` deferred with reasons (section 2).
2. ~~Verify the C3 repro is dead~~ **DONE (2026-10-23)**: live C3-shape probe (truncated
   conversation + system-shared divergent probe) all 200, LCP path fired, no abort, no
   core. Standing regression: `tests/test_checkpoint_reuse.py` (CPU lane in the matrix;
   GPU lane on `lcp-mellum.json` when convenient).
3. **C2/C4 wild-corruption audit (C5 exonerated)** - the 11:23 headroom re-run passed
   clean, so C5 is treated as OOM-induced (C5 section) and the recurrence question is
   down to C1/C2/C4 on the single-instance live service. Proceed with the data-race /
   double-free audit of everything that allocates or frees `llama_kv_cells` nodes and
   seq_pos entries, with the GPU-lane mellum suite as the repeatable trigger and the
   C2 core (251694) as the reference. Threads to clear: server queue loop (pre_decode
   seq_rm), llama threadpool (decode writes), Vulkan backend workers, APE runtime.
   Then ASan on the same workload (needs user OK - fan noise).
4. **Model-competence items (section 3)**: no code; tell the user the skills chat was
   Mellum's tool confusion + the 21:33 crash, not a broken config.
5. **Push the 9 unpushed commits + tag v0.4.2** when user says go.
