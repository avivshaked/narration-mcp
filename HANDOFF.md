# HANDOFF

*Updated 2026-09-26, night. Read this first. Then read [plan.md](plan.md), [AGENTS.md](AGENTS.md) and the
gitignored `AGENTS.local.md`, which holds this machine's paths and facts.*

## Where things stand

- **Stage: Wave 1 is merged except WP15 and WP20; Wave 2 is under way.**
  - Merged into `main` and pushed: WP00–WP03, WP01, WP10, WP12, WP13, WP14, WP16 (with follow-ups),
    WP17, WP18 (draft), WP19, contracts 1.1–1.6.1, and the private-text guard. Design revision 5.6.
    Main has 2631 tests passing.
  - **Every branch gets an independent read-only reviewer before merge.** Findings are fixed before
    merge, and a branch that had a BLOCK or a data-loss finding is re-verified by its reviewer.
  - **Private text:** the bake-off's scripts and transcripts are the owner's private story, and the
    repo is public.
    - `tools/check_private.py` runs in the git hooks, including pre-push.
    - The lead runs it on every branch before pushing, and every reviewer runs it.
    - The main checkout's gitignored `.dev/private-text.txt` and `.dev/private-terms.txt` configure it for
      every worktree.
  - **WP20** (Qwen worker): its fixes and DC-4 are done, and its history is rewritten. The lead added its
    CI job (`worker-qwen3tts`). Its reviewer is re-verifying.
  - **WP15** (alignment): the review said merge with follow-ups. F1–F4 and the nits are being fixed.
  - **WP30** (daemon): built, in independent review. **WP31** (job engine): building against WP30's seam
    (`narration.daemon.seam.JobRunner`).
  - **Follow-ups:**
    - WP12's (`wp/12-followups`): items 1–5 and two races are done; WP30's two store races are being
      added.
    - WP18's (`wp/18-followups`): the service's own default design text and WP10's fixture changes, in
      review.
- **GPU:** the owner lifted the hold on 2026-09-26 evening. Agents take the GPU lock in bounded runs.
- **Hangs:** Avast's Auto-Sandbox took custody of venv launcher `.exe`s. Agents run every tool as
  `uv run python -m …`. Avast was off for the session.
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
- **Gate H1:** listening to about 10 renders of the service's own texts before the corpus is frozen. The
  suggested sample is in `status/WP18.md`. It needs the GPU.
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
   - WP20 after its reviewer's re-verification.
   - WP15 after its fixes; its reviewer re-checks F1.
   - The WP12 and WP18 follow-ups, each after a quick review.
   - WP30 after its review. WP31 after WP30: it swaps the daemon's `DEFAULT_RUNNER` for the job engine's.
2. The lead's edits at merge:
   - WP20: design §10.1 for ADR 0002 (`bit_exact`; the encode exemption covers the whole
     `create_voice_clone_prompt`); mark ADR 0003 accepted.
   - WP15: the proposed §11.2 wording (steps 1, 3 and 5) in `status/WP15.md`.
   - WP18: DC-13's row in plan.md §1.5, and design revision 5.7's header line (both proposed in
     `status/WP18.md`).
   - WP30: design §4.1 (the daemon runs as `pythonw.exe`), and §17 (workers start in their project folder,
     with `NoDefaultCurrentDirectoryInExePath=1` on Windows and `PATH` kept).
3. Follow-ups not yet assigned:
   - **WP36 must restamp** cached QA results with the request's segment id and exact-span offsets, and turn
     `QaUnavailable` into the segment's `QA_UNAVAILABLE`. Its backend write calls must finish well inside
     the front-end's 30 s shield deadline (`server.WRITE_DEADLINE_S`). It passes `reply=None` to the
     aligner's `resolve` only for an `ALIGNMENT_ERROR` reply.
   - **WP22**: package `workers/qa`; wire WP15's `AlignOp` into `QaHandler`; write any WAV with a
     byte-reproducible writer (soundfile's float WAV has a time-stamped `PEAK` chunk).
   - **WP37**: a worker's `BACKEND_NOT_INSTALLED` sticks for the daemon's lifetime, so after
     `narration-admin install` repairs a worker, the daemon must restart (or gain a `reset_workers` command).
   - **WP36** starts the daemon with `narration.daemon.ensure_daemon`.
4. **Wave 2**, in dependency order:
   - on the CPU: WP30 and WP31 (building), then WP36 and WP37;
   - on the GPU: WP22 (QA worker) after WP15, then WP32 (engine profiles and canary).
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
