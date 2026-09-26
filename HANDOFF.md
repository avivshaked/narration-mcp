# HANDOFF

*Updated 2026-09-26, late evening. Read this first. Then read [plan.md](plan.md), [AGENTS.md](AGENTS.md) and the
gitignored `AGENTS.local.md`, which holds this machine's paths and facts.*

## Where things stand

- **Stage: Wave 0 done; Wave 1 mostly in review; the GPU lane is running.**
  - Merged into `main` and pushed: WP00, WP01 (contracts, PR #4), WP02, WP03, WP10 (text, PR #6),
    WP18 (material drafts, PR #3) and WP19 (platform, PR #5).
  - **Contracts 1.1** (PR #7) is in CI: the Wave 1 reviews' requests, plus the lead gap-fills DC-5 and DC-6.
  - **In review:** WP12 (store), WP14 (QA), WP16 (workers) and WP17 (MCP front end). A read-only reviewer
    agent is checking each branch (WP16 and WP17 share one). After that: merge each (PR), after
    contracts 1.1.
  - **Running:** WP13 (post-processing), WP15 (alignment, CPU) and **WP20 (the Qwen worker and GPU
    spikes)**.
- **GPU:** the owner lifted the hold on 2026-09-26 evening ("proceed with a working service"). Avast's
  shields are off, and the owner is at the machine to investigate any freeze. WP20 holds the GPU lock in
  bounded runs of 30 minutes or less.
- **Hangs:** two hard hangs on 2026-09-26, caused by Avast's Auto-Sandbox taking custody of venv launcher
  `.exe`s. Agents run every tool as `uv run python -m …`. Details: plan.md §9 and `AGENTS.local.md`.
- **Agents:** at most 6 at once. Opus 5.5 by default; Fable for the hardest problems.

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
- **DC-5 and DC-6** (lead gap-fills, plan.md §1.5): the owner may overrule either.
- **Gate H1:** listening to about 10 renders of the service's own texts before the corpus is frozen. The
  suggested sample is in `status/WP18.md`. It needs the GPU.
- Whether GitHub's private vulnerability reporting is the route `SECURITY.md` should name.
- Later:
  - DC-4 (`max_new_tokens`, from WP20's evidence);
  - the GitHub description, which still says voices are "locked" (WP44);
  - gates H2 to H4.
- Low priority: the default `design_text` (§16) is the bakeoff's reference text. WP18 copied it into the
  canary set as briefed. Keep it, or write a new default?

## Next steps

1. Merge contracts 1.1 (PR #7). Then, as their reviews come back, merge WP12, WP14, WP16 and WP17
   (rebase onto `main`, PR, CI, `--no-ff`). Tell WP13, WP15 and WP20 to rebase after each merge they
   need (WP20 needs WP16, and WP15 needs WP10 and WP16).
2. Open follow-ups:
   - WP17: schema size bounds on `segments`, `cues` and `hints` become `LIMIT_EXCEEDED` (§14 names
     them request-size limits).
   - WP14: switch its word adapter to `narration.text.words`, and its local signal code to
     `codes.SIGNAL_INVALID`.
   - WP12: refuse a reused `idempotency_key` (DC-6).
   - WP18: the fixture changes WP10 listed; switch `tests/material` to `narration.text`.
3. Wave 2 on the CPU once its dependencies merge:
   - WP30 (daemon: WP12, WP16, WP19);
   - WP31 (job engine: WP12 to WP14, WP16);
   - then WP36 (front end to daemon) and WP37 (CLI).
   On the GPU: WP22 (QA worker) after WP15 and WP16, then WP32 (engine profiles and canary). This is the
   path to a working service.
4. Gate H1 (the owner listens to about 10 renders) once WP20's worker renders.

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
