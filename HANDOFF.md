# HANDOFF

*Updated 2026-09-26, night. Read this first. Then read [plan.md](plan.md), [AGENTS.md](AGENTS.md) and the
gitignored `AGENTS.local.md`, which holds this machine's paths and facts.*

## Where things stand

- **Stage: Wave 1 is merged; Wave 2 is under way.**
  - Merged into `main` and pushed: WP00–WP03, WP01, WP10, WP12, WP13, WP14, WP16 (with follow-ups),
    WP15, WP17, WP18 (draft, with its follow-ups), WP19, WP20, WP12's follow-ups, contracts 1.1–1.6.2,
    and the private-text guard. Design revision 5.9. Main has 2780 tests passing; the Qwen worker's own
    suite has 152 and the QA worker's 82.
  - **Every branch gets an independent read-only reviewer before merge.** Findings are fixed before
    merge, and a branch that had a BLOCK or a data-loss finding is re-verified by its reviewer.
  - **Private text:** the bake-off's scripts and transcripts are the owner's private story, and the
    repo is public.
    - `tools/check_private.py` runs in the git hooks, including pre-push.
    - The lead runs it on every branch before pushing, and every reviewer runs it.
    - The main checkout's gitignored `.dev/private-text.txt` and `.dev/private-terms.txt` configure it for
      every worktree.
  - **WP22** (QA worker): building; GPU runs in bounded slots under the lock.
  - **WP30** (daemon): re-reviewed, the BLOCK is closed; fixing a takeover race and four small items,
    then merging with design revision 5.10 (§4, §4.1, §17 item 9).
  - **WP31** (job engine): fixing its review's findings (F1 High), splitting `engine.py`, adding §4 item 3.
  - **Follow-ups:**
    - WP16's second set (`wp/16-worker-followups`): in review (contracts 1.6.3).
  - **Gate H1:** the owner approved 8 of the 10 texts. In `ladder-080` and `align-03` the owner heard an
    artefact near the start of the take. The lead is investigating (`.dev/h1/investigate/`, local): is it the
    text or the sampling, and would the design's QA catch it? Then WP18 applies any text changes and freezes
    the manifests.
- **GPU:** the owner lifted the hold on 2026-09-26 evening. Agents take the GPU lock in bounded runs.
- **Hangs and restarts:** the machine restarted three times on 2026-09-26.
  - The first two followed Avast's Auto-Sandbox taking custody of venv launcher `.exe`s, so agents run
    every tool as `uv run python -m …`.
  - The third was a screen that would not wake, preceded only by a display-driver error. No agent was using
    the GPU, and Avast logged nothing.
  - Details are in the gitignored `AGENTS.local.md`. After a restart, check each worktree for uncommitted
    files before resuming its agent.
  - A main checkout's worker venv synced before a worker changed shape needs `uv sync --locked` again: the
    main checkout's `workers/qwen3tts/.venv` lacked the new package after WP20 merged.
- **Agents:** at most 6 at once. Opus 5.5 by default; Fable for the hardest problems.
- A removed worktree can leave an empty folder that Windows reports busy (a shell's working directory).
  It is gitignored; delete it later.

## Decided (details in plan.md §1.4 and §1.5)

- **A public project** with a public project's rigour, and **no local paths in tracked files**.
- **Licences:** the same as the evolution simulator's. The project is source-available and
  non-commercial.
- **Windows first**, with only the cheap preparation for other platforms now.
- **Distribution:** a git clone plus uv for v1. **Merges:** pull requests with CI, merged with
  `--no-ff`.
- **Models:** copied and downloaded into `<repo>\.dev\models\`, hash-verified.
- **DC-1 to DC-3 are applied** (design revision 5.2):
  - DC-1: no GPL code; pyin, Boersma HNR and CPPS;
  - DC-2: the backoff contract;
  - DC-3: the canary is designed at install time.
- **Agents:** Opus 5.5 by default. Fable is kept for the hardest problems, such as the adversarial review
  of the frozen contracts.
- **Text (lead ruling, for WP10):**
  - tab, LF and CR are whitespace, and the canonical form collapses them;
  - every other C0 and C1 control character is refused (`TEXT_REFUSED`).

## Paused by the owner (2026-09-26, 23:50), resume here

The owner needed the machine, so every agent was told to commit its work in progress, write its status
file as a handoff (`blocked:paused by owner`), release the GPU lock and stop. Nothing was merged or pushed
after `1814b3b`. Resume in this order:

1. **Check the machine is clean**: `tools/gpu_lock.py status` is free (release it if WP22 left it held,
   after checking no WP22 run is alive); no `-m narration.daemon` or `narration_worker` process is left.
2. **WP16 follow-ups** (`wp/16-worker-followups`, b131baa): re-verified, merge with follow-ups (all Low).
   No refusal changed across 128 load cases; the WAV writer is byte-identical; no cycles; standard library
   only. Merge first, with the lead's design note (§11.2 step 7, the four `CUE_UNALIGNED` reasons; the
   lead's script bumps the revision). Contracts 1.6.3. Low follow-ups: `model` refs are checked by three
   implementations (the fake, qwen3, WP22's), so share one `check_snapshot_ref` or reword the CHANGELOG's
   "the Qwen worker's own checks" for `model`; WP22 imports the shared patterns; the writer can still raise
   `struct.error` past 4 GiB of data.
3. **WP30** (`wp/30-daemon`, 05f99f3, state `review`): all fixes done, including the third review's
   three last ones (an answered stop from the future is not trusted; only a `stopped: true` answer stops a
   waiter; `[0-9]` in `_STORE_TIME`), each with a test that failed before it. Full suite 2968 passed. The
   lead checks the diff since c7e15c6, renumbers the contract to 1.6.4 on rebase, and merges with the
   drafted design text for §4, §4.1 and §17 item 9 (next revision).
4. **WP31** (`wp/31-jobs`, b98d4a5, status handoff 2cb9905): fixed; its reviewer's re-verification was cut
   short. Done: both checks, `tests/jobs` once (107 passed), a full read (F1–F5 closed on reading, the F9
   replay and the invariants sound). A candidate Medium, not yet confirmed: the lease keeper starts a
   thread per piece of work, and the store opens one connection per thread and closes it only at
   `store.close()`, so each renewal may leave a connection open for the daemon's life. Resume with the
   reviewer's probes P1–P8 in the worktree's gitignored `.pytest-tmp/review31/test_review31b.py` (P6 is
   the connection count), then `tests/jobs` twice more, ruff, basedpyright and the full suite.
   After WP30 merges: rebase, contracts 1.6.5, the seam re-export, `DEFAULT_RUNNER`, `host.platform`,
   `CUDA_DEVICE_ORDER=PCI_BUS_ID`, one exported log name. The lead's decisions are in `status/WP31.md`:
   one job at a time (priority, then first come; affinity keeps the resident group, never reorders);
   `embed`'s `"cuda"` is the QA group's device; cross-job grouping in a round (§4 item 3) is put to the
   owner as a scope question (about 2–3 days; saves one model swap per extra queued job per round).
5. **WP22** (`wp/22-qa-worker`, 99ab00d, on main 7731587, not rebased): paused at a safe stop.
   - Done: the `qa` role's handler (load, unload, transcribe, embed, f0, align, profile) passing the
     shared contract; `"English"`/`"en"` accepted for ASR; `embed` on the loaded group's device; load
     refusals matching the fake and qwen3; WP15's F4 (out of memory is not a broken install).
   - KNOW: acceptance through the protocol matches the bake-off (WER exact on six takes; similarity and
     52 voicelock pairs within 0.0001). WavLM revision `main` (`feb593a6`), whose weights equal
     `refs/pr/8`'s. VRAM: Whisper + WavLM resident 3.6 GB; word timestamps about 3 GB more; WavLM
     embedding grows with the square of clip length (0.6 GB at 30 s, 8.4 GB at 119 s).
   - **Decisions for the lead:** ADR 0004 (proposed): the design's greedy decoding conditioned on the
     previous window repeats itself on 2 of 6 takes (WER 0.40), so the worker uses five beams without
     conditioning, which reproduces the bake-off. Also: a windowed embedding for clips over 30 s to cap
     WavLM's memory (not built without a decision).
   - Next steps are in `status/WP22.md` (two CPU spikes, the VRAM figure, the GPU tests under the lock,
     the shared patterns after WP16's follow-ups merge, CHANGELOG, rebase, review).
   - **The GPU lock, to check:** WP22 reports releasing it at 22:49:29 after each of its runs. The lead
     read `status` as held by WP22 (start 22:38:51, pid 14180, not stale) at 23:45 and about 23:52, and
     free at about 23:57. The two accounts disagree. Check `tools/gpu_lock.py`'s release and status
     paths, and WP22's run scripts, before the next GPU work.
6. **Gate H1**: 8 of 10 approved. `ladder-080` and `align-03` had an artefact at the start of take 1 only;
   the owner found both second takes clean, so it is most likely the sampling. The design already names
   this failure (`HEAD_INSERTION`, reference bleed, §11 step 7). The investigation was stopped after its
   measuring half; its scripts and outputs are in the local `.dev/h1/investigate/` (no conclusions were
   returned). To finish: re-run the measure and research agents from those files, then the 8-seed
   reproduction on the GPU, then recommend to the owner whether to approve both texts as written.

## Waiting on the owner

- **The Avast Auto-Sandbox exception** for the projects folder, or Auto-Sandbox off: the durable fix for
  the hangs.
- **Gate H1:** the owner has listened and approved 8 of 10; the last two wait on the lead's investigation
  of the artefact at the start of their takes.
- Whether GitHub's private vulnerability reporting is the route `SECURITY.md` should name.
- Later:
  - one run of the daemon under the owner's real MCP client: spike (g) modelled a session's job object,
    but not a real client's;
  - the GitHub description, which still says voices are "locked" (WP44);
  - gates H2 to H4.

## Next steps

1. Merge as each branch comes back. Before any push, run
   `py -3.12 tools/check_private.py --commits main..<branch> --base main`; the pre-push hook runs it too.
   Then: full suite, PR, CI, `--no-ff`, remove the worktree.
   - WP22 after an independent review.
   - WP30 after its fixes and its reviewer's re-verification (it had a BLOCK).
   - WP31 after its review and after WP30. It then re-exports WP30's seam and swaps the daemon's
     `DEFAULT_RUNNER` for `narration.jobs.runner:default_runner`.
   - WP16's second set of follow-ups after a quick review.
2. The lead's edits at merge:
   - WP30: design §4.1 (the daemon runs as `pythonw.exe`), and §17 (workers start in their project folder,
     with `NoDefaultCurrentDirectoryInExePath=1` on Windows and `PATH` kept, and `python -P`).
   - WP31: its lead-authorised contracts 1.6.3 (`hello` on `interfaces.WorkerClient`; `JobRecord.result`'s
     docstring); §7.3 and §8 say that `outcome` is `needs_attention` only for a verdict-fail suggestion or a
     segment with no take.
3. Follow-ups not yet assigned:
   - **WP36 must restamp** cached QA results with the request's segment id and exact-span offsets, and turn
     `QaUnavailable` into the segment's `QA_UNAVAILABLE`. Its backend write calls must finish well inside
     the front-end's 30 s shield deadline (`server.WRITE_DEADLINE_S`). It passes `reply=None` to the
     aligner's `resolve` only for an `ALIGNMENT_ERROR` reply.
   - **WP22**:
     - package `workers/qa`;
     - wire WP15's `AlignOp` into `QaHandler`;
     - write any WAV with the shared byte-reproducible writer (WP16's second follow-ups);
     - accept the language as the job engine sends it ("English"), or map it;
     - measure the aligner's memory beyond about 40 s of audio.

     The aligner's `_is_broken_file` treats any non-memory `RuntimeError` as a damaged snapshot; narrow it.
   - **WP32**: assemble the aligner and the QA model pins in the job engine's `installed_engine`.
   - **WP38**: re-set the aligner's snap reaches (ASSUME) from a hand-marked boundary next to an unplaced
     cue, and consider a separate reach for wildcard edges.
   - **Store (low, from WP12's re-verification):** treat a zero file id (`st_ino == 0`) as unknown in the
     undo's check; close the thread's connection if a ROLLBACK ever fails, so the write lock is released.
     FAT and exFAT reuse file ids, which makes the check no stronger there, never weaker.
   - **WP16 (deferred):** whether the fake refuses a bare load once WP30 and WP31 have merged; moving
     qwen3's settings parser into `narration_worker` so the fake checks every sampling value.
   - **WP37**: `daemon stop` posts only when `running_daemon` says a daemon runs. A worker's
     `BACKEND_NOT_INSTALLED` sticks for the daemon's lifetime, so after
     `narration-admin install` repairs a worker, the daemon must restart (or gain a `reset_workers` command).
   - **WP36** starts the daemon with `narration.daemon.ensure_daemon`. It reads the aligner's
     `measured_error` from the current benchmark, not from a cached analysis (a re-benchmark of the same
     method does not change the analysis key).
4. **Wave 2**, in dependency order:
   - on the CPU: WP30 and WP31 (building), then WP36 and WP37;
   - on the GPU: WP22 (QA worker, building), then WP32 (engine profiles and canary; it also assembles
     the job engine's `installed_engine`), and WP38 (alignment benchmark; needs gates H2, H3).
5. Gate H1 (the owner listens to about 10 renders) once the full pipeline renders.

## Things a new session should know

- The bakeoff that preceded this project is private and read-only. Its reusable code, golden numbers
  and models are mapped in plan.md §1.2. The findings from its installed code are in §1.3: the effective
  `max_new_tokens` of 8192, the normaliser's spelling map, and the proportional reference cut.
- The consumer's requirements (R1–R14) are mapped in design §21. The service must never depend on that
  caller (the owner's principle, at the top of the design).
- uv needs `UV_NATIVE_TLS=1` on this machine, because something intercepts TLS (`AGENTS.local.md`).
- The PR flow:
  1. Push the branch.
  2. Write the PR body to `.dev\pr\<wp>.md`, and check it with `tools/check_tracked.py --msg-file`.
  3. Run `gh pr create`, then `gh pr checks N --watch`.
  4. Merge locally with `git merge --no-ff`, and push `main`.
  5. Remove the worktree.
- The Claude Code harness reports "file changed on disk" when a file is touched through `d:\` and then
  `D:\`. That is the path's case, not an outside edit.

## How to resume

1. Read this file, then plan.md §4 (status), §1.5 (design changes) and §9 (log).
2. Check each active worktree's `status\WPnn.md`.
3. Record any new answers from the owner in plan.md and here, then continue with the next step above.
4. Keep this file current at the end of every session and every wave: the stage, what changed, what is
   open, and what is next.
