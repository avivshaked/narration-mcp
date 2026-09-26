# HANDOFF

*Updated 2026-09-26, late evening. Read this first. Then read [plan.md](plan.md), [AGENTS.md](AGENTS.md) and the
gitignored `AGENTS.local.md`, which holds this machine's paths and facts.*

## Where things stand

- **Stage: Wave 1 is nearly merged; Wave 2 has started.**
  - Merged into `main` and pushed: WP00–WP03, WP18, WP01, WP19, WP10, WP16, WP12, WP13, and contracts
    1.1–1.6. Design revision 5.5. Main has 2142 tests passing.
  - **Every branch gets an independent read-only reviewer before merge.** The reviews found about 50 real
    defects in branches whose suites all passed. Findings are fixed before merge, and a branch that had
    a BLOCK or a data-loss finding is re-verified by its reviewer.
  - **WP14** (QA): fixing the second re-review's F1 and F3, then merge (no further review needed).
  - **WP17** (MCP): fixing the re-review's four small items, then merge.
  - **WP15** (alignment): built; adding DC-11's wildcard (on condition of its measurement) and
    contracts 1.6's fields; then an independent review.
  - **WP16 follow-ups** (`wp/16-followups`): a–c done; adding (d), DC-4's per-call cap, to the fake
    worker; then the WP16/WP17 reviewer verifies.
  - **WP20** (Qwen worker, GPU): done, under review. After the review it adds the per-call cap (DC-4).
    Its renders are listenable in its worktree's `.dev\spikes\`.
  - **WP30** (daemon): building.
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
  - the GitHub description, which still says voices are "locked" (WP44);
  - gates H2 to H4.
- Low priority: the default `design_text` (§16) is the bakeoff's reference text. WP18 copied it into the
  canary set as briefed. Keep it, or write a new default?

## Next steps

1. Merge as each branch comes back: rebase onto `main`, full suite, PR, CI, `--no-ff`, remove the
   worktree. Next: WP14, WP17, the WP16 follow-ups, then WP20 and WP15 after their reviews.
2. **Start WP31** (job engine) as soon as WP14 merges; its dependencies WP12, WP13 and WP16 are in.
   Tell it:
   - construct `DeliveryPipeline(fade_s=config.delivery.fade_s)`;
   - compute each call's cap with `names.max_new_tokens_for`;
   - use `codes.is_retake_trigger(code, severity, details)`.
3. Follow-ups not yet assigned:
   - **WP12** (low; review round 2):
     1. gc holds the write lock for the whole collection; batch it before `narration-admin gc` exists;
     2. post-commit trash removal is best-effort;
     3. a failed COMMIT in `_publish_dir`;
     4. `put_canary_clip` destroys the clip on failure;
     5. `_collect`'s restore ordering.
   - **WP18**: the fixture changes WP10 listed; switch `tests/material` to `narration.text` and
     `narration.lint`.
   - **WP36 must restamp** cached QA results with the request's segment id and exact-span offsets
     (contracts 1.2), and turn `QaUnavailable` into the segment's `QA_UNAVAILABLE`. Its backend calls,
     except `get_job`'s wait, must finish well inside the front-end's shield deadline.
   - **WP22**: package `workers/qa`; wire WP15's `AlignOp` into `QaHandler`; write WAVs with a
     byte-reproducible writer, never soundfile's defaults (soundfile's float WAV has a time-stamped
     `PEAK` chunk).
   - **WP36**: pass `reply=None` to the aligner's `resolve` only for an `ALIGNMENT_ERROR` reply.
   - **CI**: nothing type-checks `workers/qwen3tts` (WP20's note).
4. **Wave 2**, in dependency order:
   - on the CPU: WP31, then WP36 and WP37;
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
