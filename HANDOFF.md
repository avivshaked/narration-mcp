# HANDOFF

*Updated 2026-09-27, morning. Read this first. Then read [plan.md](plan.md), [AGENTS.md](AGENTS.md) and the
gitignored `AGENTS.local.md`, which holds this machine's paths and facts.*

## Where things stand

- **The owner's priority (2026-09-27): first narration through the MCP as soon as possible**, because it
  blocks the owner's other work. Milestone **M1** is defined in plan.md §6: measure a voice, then narrate
  paragraphs and receive QA'd, cue-aligned takes, from a Claude Code session. Everything on M1's path runs
  in parallel now; the rest of plan.md follows M1.
- **Merged into `main` and pushed:** Waves 0 and 1 (WP00–WP03, WP10, WP12–WP20 with their follow-ups),
  WP16's second follow-ups (PR #24), WP30 the daemon (PR #25) and its test fix (PR #26), one
  configuration rule (PR #27, `narration.config.find_config`), WP31 the job engine (PR #28), contracts
  1.1–1.6.5, the private-text guard. Design revision 5.12.
- **In flight.** Agent ids resume with SendMessage. WP31 merged (PR #28), so the branches stacked on
  its old head `2cb9905` rebase with `git rebase --onto main 2cb9905`.

  | Package | Branch | Agent | State |
  |---|---|---|---|
  | WP37 operator CLI | `wp/37-admin` | a1b119ea6c733e52b (reviewer a9dd8b1a7ed2648b1) | in review; `render` follows WP36 |
  | WP22 QA worker | `wp/22-qa-worker` | a560deac0c56c674c (reviewer ad2cf436a03859564) | in review; acceptance passes (DC-14, DC-15 at 60 s); QA group `vram_need_mb` = 11500 |
  | WP32 engine profiles, canary, `installed_engine` | `wp/32-engine` | a203bffe41405429f | building; the real `installed_engine` done; `engine` commands on WP37's interface next; DC-16 awaits the owner |
  | WP33 `measure_voice` | `wp/33-measure` | a21fe923fab0198e7 (reviewer a9759b8d3287035d1, Fable) | in review |
  | WP36 the tools (front-end ↔ daemon) | `wp/36-backend` | a35085aca80cfb4e8 (reviewer ae39d41a64a79fc93, Fable) | in review; created-`completed` at submit (§7.3) is a follow-up |

- **Every branch gets an independent read-only reviewer before merge.** Findings are fixed before merge,
  and a branch that had a BLOCK or a data-loss finding is re-verified by its reviewer.
- **Private text:** the bake-off's scripts and transcripts are the owner's private story, and the repo is
  public. `tools/check_private.py` runs in the git hooks, including pre-push; the lead runs it on every
  branch before pushing, and every reviewer runs it.
- **GPU:** agents take the GPU lock in bounded runs. `owner.json` stamps UTC; the "discrepancy" of
  2026-09-26 was a UTC stamp read as local time, and the tool is consistent.
- **Hangs and restarts (2026-09-26):** two followed Avast's Auto-Sandbox taking custody of venv launcher
  `.exe`s, so every tool runs as `uv run python -m …`; the third was a display that would not wake. Details
  in `AGENTS.local.md`. After a restart, check each worktree for uncommitted files before resuming its agent.
- A main checkout's worker venv needs `uv sync --locked` again after a merge changes the worker.
- **Agents:** at most 6 at once. Opus 5.5 by default; Fable for the hardest problems.
- The lead's local drafts are in the gitignored `.dev/lead/` (design-revision helper, doc-update scripts).

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
- **Gate H1:** 8 of 10 approved; for the last two the lead recommends approving as written (see
  `OWNER-ACTIONS.md` §1, with two files to listen to if the owner wants).
- **DC-14 and DC-15** (plan.md §1.5): lead gap-fills the owner may overrule.
- **4a in the owner's list:** cross-job grouping (§4 item 3); the lead recommends leaving it out of v1.
- Whether GitHub's private vulnerability reporting is the route `SECURITY.md` should name.
- Later:
  - one run of the daemon under the owner's real MCP client: spike (g) modelled a session's job object,
    but not a real client's;
  - the GitHub description, which still says voices are "locked" (WP44);
  - gates H2 to H4.

## Next steps (the lead)

1. WP30 and WP31 are merged; WP32, WP33 and WP36 are rebasing onto `main`.
2. **WP37 follow-ups:** `render` over WP36's backend once WP36 is on `main`; `doctor` checks the material
   manifests once WP33's corpus loader is; a single-item evict in the store for what `verify` finds damaged
   (the store's area); `reset_workers` stays a proposal.
3. **WP22:** independent review, then merge with design §11.1's text for DC-14 and DC-15.
4. **WP32, WP33, WP36:** reviews and merges, as each is ready. Then WP37 (the CLI; `engine pin` is WP32's).
5. **Gate H1:** the result is in `OWNER-ACTIONS.md` §1 (the fault is the seed's; recommend approving 05
   and 08 as written). On the owner's approval, the lead freezes the manifests (status `frozen`) and adds
   the calibration's design text to `calibration/narration-en.v1/paragraphs.json` as a top-level
   `design_text` item (`segment_id` `cal-design`; WP33's ruling). That unblocks the real `measure_voice`
   runs, WP38 and WP39.
6. **M1 on this machine:** a local config (models in `.dev/models`, d2 and d4 in `allow_sha256`), the worker
   venvs synced, `narration-admin engine pin`, then the owner measures d2 or d4 (20–50 min of GPU, at a time
   the owner chooses) and narrates through a real Claude Code session.

## Follow-ups to fold into the packages

From earlier reviews; the WP36, WP32 and WP22 items are in those agents' briefs.

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
- **WP16 (Low, from the second follow-ups' review):** the snapshot reference is checked by three
  implementations (the fake, qwen3, the QA worker): share one check; the shared WAV writer can still raise
  `struct.error` past 4 GiB of data.

- **QA blind spots found by gate H1's investigation (for WP14's area and WP40's planted faults):**
  - §11.1 step 7's acoustic head check (speech before the first word that matches the voice's
    transcript) is not implemented; `narration.qa.textmatch` checks only Whisper's words.
  - A garbled or dropped first word counts as one word error (a warning on a 13-word cue), not
    `HEAD_INSERTION`, so it is not retaken; two garbled words fail and are retaken.
  - Fillers at the start ("um", "uh") are removed by the normaliser and never counted.
  - WP40 should plant "garbled first word" and "filler at the start" as faults; a voice-quality glitch that
    Whisper cannot hear is a known limit of v1's QA.

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
