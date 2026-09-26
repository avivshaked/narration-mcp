# HANDOFF

*Updated 2026-09-26, late evening. Read this first. Then read [plan.md](plan.md), [AGENTS.md](AGENTS.md) and the
gitignored `AGENTS.local.md`, which holds this machine's paths and facts.*

## Where things stand

- **Stage: Wave 1 is being reviewed and merged; the GPU lane is running.**
  - Merged into `main` and pushed: WP00–WP03, WP18, WP01, WP19, WP10, WP16, and contracts 1.1–1.4.
    Design revision 5.4. Main has 1689 tests passing.
  - **Every branch gets an independent read-only reviewer before merge.** The reviews found about 40 real
    defects in branches whose suites all passed. Findings are fixed before merge, and a branch that had
    a BLOCK or a data-loss finding is re-verified by its reviewer.
  - **WP12** (store): round 2 fixed; its reviewer is re-verifying. It carries a lead commit (config's
    `allow_sha256` whole match). Rebase onto main when the verification is done, then PR.
  - **WP13** (post), **WP14** (QA), **WP17** (MCP): their agents are fixing the review findings.
  - **WP15** (alignment, CPU) and **WP20** (Qwen worker, GPU) are building. WP20 has run spikes (h)(i),
    (d), (e), DC-4 and its acceptance renders; the audio is in its worktree's `.dev\spikes\`.
- **GPU:** the owner lifted the hold on 2026-09-26 evening. WP20 holds the GPU lock in bounded runs.
- **Hangs:** Avast's Auto-Sandbox took custody of venv launcher `.exe`s. Agents run every tool as
  `uv run python -m …`. Avast was off for the session.
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

1. Merge as each branch comes back: rebase onto `main`, full suite, PR, CI, `--no-ff`, remove the
   worktree.
   - WP12 after its re-verification.
   - WP13, WP14 and WP17 after their fixes. Re-verify WP14, which had a BLOCK.
2. Follow-ups not yet assigned:
   - **WP16** (low; before WP30 and WP20/WP22 lean on the client):
     - (a) `_fail` should stop the writer thread (enqueue the sentinel);
     - (b) a writer error should `_stop` the client, not wait out a timeout;
     - (c) start-up classification: exit 2 only for `RoleUnavailable`/`ImportError`; map torch's DLL
       `OSError` to `BACKEND_NOT_INSTALLED`.
   - **WP18**: the fixture changes WP10 listed; switch `tests/material` to `narration.text` and
     `narration.lint`.
   - **WP36 must restamp** cached QA results with the request's segment id and exact-span offsets
     (contracts 1.2).
3. **Wave 2** once its dependencies merge:
   - on the CPU: WP30 (daemon: WP12, WP16, WP19), WP31 (job engine: WP12–WP14, WP16), then WP36 and
     WP37;
   - on the GPU: WP22 (QA worker) after WP15, then WP32 (engine profiles and canary).
4. Gate H1 (the owner listens to about 10 renders) once the full pipeline renders.

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
