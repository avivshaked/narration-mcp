# HANDOFF

*Updated 2026-09-26, evening. Read this first. Then read [plan.md](plan.md), [AGENTS.md](AGENTS.md) and the
gitignored `AGENTS.local.md`, which holds this machine's paths and facts.*

## Where things stand

- **Stage: Wave 0 is done except WP01. Wave 1 is fanned out.**
  - Merged into `main` and pushed: WP00 (bootstrap), WP02 (scaffolding, PR #2), WP03 (CI, PR #1),
    design revision 5.2 (DC-1 to DC-3), and WP18 (the service's own material as drafts, PR #3).
  - **WP01 (contracts, lead)** is on `wp/01-contracts` in `worktrees\wp01-contracts`. A Fable review of it
    is being applied; the fixes may still change the contracts.
- **Wave 1 runs as a git stack.** WP10, WP12, WP14, WP16 and WP19 are branched from `wp/01-contracts`,
  not from `main`, so they could start before WP01 merged.
  - If a contract they use changes, message those agents.
  - When WP01 merges, each of them rebases onto `main`. The commits already in `main` drop out.
- **Agents working now:**

  | WP | Worktree | Model |
  |---|---|---|
  | WP10 | `worktrees\wp10-text` | Opus 5.5 |
  | WP12 | `worktrees\wp12-store` | Opus 5.5 |
  | WP14 | `worktrees\wp14-qa` | Opus 5.5 |
  | WP16 | `worktrees\wp16-workers` | Opus 5.5 |
  | WP19 | `worktrees\wp19-platform` | Opus 5.5 |

  Each agent's report lands in `status\WPnn.md` in its worktree.
- **Waiting to start:**
  - WP17 (the MCP front end) has spike (j) and ADR 0001 done on `wp/17-mcp`. It resumes after WP01 merges.
  - WP13 and WP15 start as slots free.
  - WP20 is held (see below).
- **Incident, 2026-09-26 at about 17:14 local.** The machine froze hard or lost power (Kernel-Power 41).
  There was no bugcheck and no dump, and nothing was logged in the minutes before.
  - Our load at the time was light, and no model was on the GPU.
  - The machine had an earlier blue screen (bugcheck 0x3B) on 2026-09-22, before this project began.
  - After the restart: `git fsck` was clean, the model files re-hashed and matched, and nothing written
    was lost except WP03's first attempt.
  - **GPU work (WP20, and every GPU WP after it) is on hold** until the owner says the machine is fine to
    load models on.
  - Agents are asked to keep their load moderate: one test suite at a time, and no parallel pytest.

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

- **Whether GPU work may start** after the crash.
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

1. Apply the review's confirmed findings to WP01. Merge it (PR), then tell the stacked agents to rebase.
2. Resume WP17. Start WP13 and WP15 as agents finish.
3. Review and merge each Wave 1 WP as it reaches `review` (plan.md §2.4).
4. Wave 2's CPU-only WPs (WP30, WP31, WP36, WP37) once their dependencies merge. The GPU WPs wait for the
   owner.

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
