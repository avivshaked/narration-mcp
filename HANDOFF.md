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
