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
- **Incidents on 2026-09-26: two hard hangs, cause found.** The first was at about 17:14 and the second at
  about 18:16. Both were Kernel-Power 41 with no bugcheck and no dump.
  - **The cause (KNOW, from the logs):** Avast's Auto-Sandbox took full custody of a venv launcher `.exe`
    1 to 3 minutes before each hang: `basedpyright.exe` the first time, `basedpyright.exe` and
    `pytest.exe` the second. It did that only three times all month.
  - **The fix:**
    - Agents run every tool as `uv run python -m pytest|basedpyright|ruff`, never through the launchers.
      This is in `AGENTS.local.md` and `.dev\brief-common.md`.
    - The owner turned Avast's shields off for the session. With them off, the same launchers ran
      cleanly and Avast logged nothing.
    - The durable fix is the owner's: an Auto-Sandbox exception for the projects folder, or
      Auto-Sandbox off.
  - Not the cause: memory (64 GB, no low-memory events), GPU load (we used none), and the 5-minute
    Hyper-V VM cycle (a coincidence the first time; absent the second time). The machine's older display
    watchdog dumps and GPU timeouts (August to 25 Sep) predate this project.
  - After each restart, `git fsck` was clean. Uncommitted work in the worktrees survived on disk.
  - **GPU work (WP20 and every GPU WP after it) is still on hold** until the owner says the machine is
    fine to load models on.

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

0. **Resume point after the second hang (2026-09-26, 18:30).** Every agent was stopped mid-work, and their
   work is on disk. Resume each one by `SendMessage` to its agent id; the transcripts are saved. Tell each
   one to use `python -m` for every tool, and to re-read its worktree's `AGENTS.local.md`.

   | WP | State | Agent id |
   |---|---|---|
   | WP10 | Uncommitted text, lint and tests. Its `material/` copy is untracked and identical to `main`: delete it before rebasing onto `main`. | `aa4257c5ade2b7a92` |
   | WP12 | Keys committed (d53bcd9); store uncommitted. | `ae9960d6968fad493` |
   | WP14 | QA code uncommitted. | `a8afe69ab845d6718` |
   | WP16 | The whole worker package uncommitted. | `a66dcf85f3bbf472e` |
   | WP17 | Phase 2 just started; ADR 0001 edit uncommitted. | `a7fc4e8bbedc32ede` |
   | WP19 | Seam committed (e14dcb8); tests uncommitted. Its process and Job Object tests have **never run yet**. | `a5266d36bbeec76db` |
   | WP01 | Its Fable records-keys review was interrupted. | `a332abe3753e402f0` (read-only, no file writes) |

   Then merge WP01.
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
