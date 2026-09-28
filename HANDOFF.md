# HANDOFF

*Updated 2026-09-28. Read this first. Then read [plan.md](plan.md), [AGENTS.md](AGENTS.md) and the
gitignored `AGENTS.local.md`, which holds this machine's paths and facts.*

## Resumed 2026-09-28, with the machine kept free (the owner's rule)

**The owner needs the machine free: agents run no local pytest and no basedpyright.** They run ruff, check_private
and check_tracked only. They push, and CI on GitHub is the test runner (`gh run watch <id> --interval 60`).
GPU work (the re-measure, the ladder, WP38–WP42) waits for a window the owner chooses.

- **Merged 2026-09-28:**
  - the pace hotfix (PR #42, fda412f; DC-19; contracts 1.6.7);
  - PR #40.

  The hotfix is live: no daemon was running at the merge, and the owner's `narration.toml` `[qa] profile` now
  says `default.v4` (the key is inert). No re-measure was needed: the measurement key does not include the QA
  profile (KNOW, and verified by PR #42's reviewers). The owner's Claude Code session reads PR #40's tool texts
  after a `/mcp` reconnect.
- **PR #37 merged** (1b05644, 2026-09-28; CI green on it merged with main). The daemon now starts only once it is in
  no Job Object at all.
  - **KNOW (2026-09-28): under Claude Code (the VS Code extension), processes sit in a Job Object, and a child
    started with `CREATE_BREAKAWAY_FROM_JOB` ends up in no job at all.** So the daemon starts detached under
    Claude Code (a probe from the lead's session with PR #37's `_windows._in_job`).
  - BELIEVE: the owner's narrating session behaves the same.
  - If it ever answers `DAEMON_UNAVAILABLE` (`left_in_job`), the route is `narration-admin daemon start
    --foreground`.
  - Its follow-ups are in WP49. Its worktree folder is empty but still locked by some process; remove
    `worktrees/wp30-escape` later.
- **Merged 2026-09-28, later:**
  - **PR #46, WP49's follow-ups:**
    - doctor warns on a QA-profile mismatch;
    - a log line when a job the daemon ran ends;
    - PR #37's follow-ups;
    - the refusal's messages are constants in `narration.platform`.

    Its worktree and branch are removed, and so is the `wp30-escape` folder.
  - **PR #43, design revision 5.15:** DC-17, DC-19 and the Job Object rule; two reviews.
  - **PR #39, WP34 `design_voice` and `profile_voice`:** contracts 1.6.8. The owner's session sees the two tools
    after a `/mcp` reconnect.
  - **WP34's follow-ups (all low):**
    - the description check does not refuse VoiceDesign's own non-`<|` special and added tokens (`<tts_pad>`,
      `<tts_text_bos>`, `<think>`, `<tool_call>` …; they are in the pinned tokenizer's config);
    - CLIP_TOO_LONG's retake column says `always`, but the flag is never a retake trigger;
    - provenance is added before publishing, so a crash in between can leave a line for a clip that is designed
      again (harmless on a bit-exact engine);
    - no cache answers a repeated design or profile;
    - no WER warn level;
    - a batch design that gives way loses its unpublished clips;
    - §14's CLIP_TOO_LONG text is the lead's to write.
  - **PR #45, WP48 `narration-admin failures`** (03c4aa0): every failed take in the store, with its flags,
    metrics and audio path; `--json` for WP46's view. The lead checked by hand that it never writes the store
    (the new `Store.peek_*` and `iter_jobs` are plain SELECTs). Replaces `.dev/lead/failures_now.py`.
  - **PR #41, `get_job` and `cancel_job` revive a dead daemon** (644ae49). A launched daemon that exits before
    it serves gives a retryable `DAEMON_UNAVAILABLE` with `details.log`, and no second launch inside the 90 s
    window. The lead read the fix (`daemon.start.check_launch`, `sweep.launch_alive`, `service._failed_start`).
    - **Decided by the lead:** `narration-admin install`'s stop follows the same rule as `daemon stop` (jobs
      queued before it wait for the next start; the next submit_job starts one).
    - **Follow-ups** (in `status/WP36-liveness.md`): the stopped daemon's launch time or stop reason in
      `run/daemon.json` (approved, a contract change); measure launch-to-status time (four BELIEVE values);
      three contract requests (a `daemon` object in get_job's output, `MeasurementRecord.transcript`, a
      `check_text` flag for a transcript near-miss); design text for §3.2, §4, §4.1, §7.3, §14 and §15.
    - Open for the owner: `measure_voice` with a mistyped transcript still queues a new measurement.
  - Their worktrees and branches are removed.
- **Merge order from here:** WP47 (PR #44, contracts 1.6.9), after its fix round (DC-20, the report's fallback
  for older jobs) and green CI; told to merge main again first. **After WP47:**
  re-measure a-warm-s101 (cached renders, QA only), then the rest of the ladder in stages (remove
  `.dev/service-setup/STOP`), both in a GPU window.
- **Decisions carried into the resumed agents:**
  - **PR #41, L1: option (b′).** A queued job under a `stopped` status counts as an operator's stop only if both
    hold:
    - a `stop` or `stop_now` answered `stopped: true` is among the commands since that daemon started;
    - the job was created before that stop was posted.

    Otherwise a daemon is started (only the note when autostart is off). Why: `stopped` is also written after a
    control-loop error, a supervisor failure, and a submit during an operator's stop.
  - **WP47:** the rate is characters per second of speaking time, excluding silences of 0.25 s or more, from the
    signal stage's speech mask (ASSUME until it is measured on the re-measured voice). Measurement schemas move
    to `/v2` with `PACE_METHOD` in the key, and the QA profile becomes `default.v5`.
    - **Review at merge:** the contract fields it added beyond the lead's approval (each justified in its status
      file).
    - It once tried to measure pauses on the bake-off's clone takes, and the permission classifier refused
      that. Measure on the owner's synthetic voice after the merge.
  - **WP34:** take design_voice and profile_voice out of `NOT_IN_THIS_BUILD` (PR #40's equality test enforces
    it), and keep both #40's design_text wording and its own CLIP_TOO_LONG sentence.
- **Cleaned up:** the worktrees and branches of PRs #40 and #42 (local and remote). `git worktree remove`
  failed with "Filename too long" on Windows (deep `.pytest-tmp` paths). The folders went with PowerShell
  `Remove-Item -LiteralPath "\\?\<path>" -Recurse -Force`.

**The owner's first real narration session (21:46–22:10): 9 jobs ran** through the MCP from the owner's Claude
Code session.
- Every take that failed QA failed on `PACE_FAST` (24 takes; 7 of them also warned `SPK_SIM_LOW`), all with WER 0 and high speaker
  similarity. The owner listened and confirmed they were not fast.
- The cause (KNOW): the pace model is words per minute over a span that includes the pauses between sentences.
  The wpm curve follows the corpus's word lengths, and a one-sentence segment has no pauses.
- The owner decided:
  - **PACE_FAST warns only** (the hotfix);
  - **pace in characters per second with the pauses excluded** (WP47).
- Failed takes are listed by `narration-admin failures` (WP48, merged).
- The daemon now logs a line when a job it ran ends (WP49, merged).

## Where things stand

- **The owner's priority (2026-09-27): first narration through the MCP as soon as possible**, because it
  blocks the owner's other work. Milestone **M1** is defined in plan.md §6: measure a voice, then narrate
  paragraphs and receive QA'd, cue-aligned takes, from a Claude Code session. Everything on M1's path runs
  in parallel now; the rest of plan.md follows M1.
- **Merged into `main` and pushed:** Waves 0 and 1 (WP00–WP03, WP10, WP12–WP20 with their follow-ups),
  WP16's second follow-ups (PR #24), WP30 the daemon (PR #25) and its test fix (PR #26), one
  configuration rule (PR #27, `narration.config.find_config`), WP31 the job engine (PR #28), WP22 the QA
  worker (PR #29), WP37 `narration-admin` (PR #30), WP33 `measure_voice` (PR #31), WP36 the MCP tools and `narration-admin render` (PR #32), DC-16 (PR #33), WP32 engine profiles and the canary (PR #34) and its follow-ups (PR #36), gate H1's freeze (PR #35), contracts 1.1–1.6.6, the private-text guard. Design
  revision 5.14.
- **In flight.** Agent ids resume with SendMessage. WP31 merged (PR #28), so the branches stacked on
  its old head `2cb9905` rebase with `git rebase --onto main 2cb9905`.

  | Package | Branch | Agent | State |
  |---|---|---|---|
  | WP47 pace in characters per second, with the pauses excluded | `wp/47-pace-cps`, PR #44 | af6d02f523a3a52f0 | fixing its review: DC-20 (the flat level), the report's wording for pre-1.6.9 jobs, `Pace.method`; merging main (#41, #45) |

  PR #37's follow-ups (all low; from its verifiers): the admin refusal text still embeds the platform's "this
  client" wording for `breakaway_refused` and `job_check_failed`; the docs name only `--foreground` for a
  host that forbids breakaway, not `[daemon] autostart = false`; the CI log cannot show which branch the
  survival tests took (a `warnings.warn` or `record_property` would); a regression that refuses every detached
  start stays green on GitHub's runner (only a Windows developer run catches it); a docstring line that the
  no-orphan guarantee holds from the moment `Popen` returns, not from `CreateProcess`.

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

- **The readiness audit's decisions** (`.dev/lead/readiness_synthesis.md`, local; section 4):
  - D1: a full ladder for the first voice, or keep the stop at 300 (some of the consumer's paragraphs are
    longer); needs a GPU hold over 30 min, which the lock tool refuses without an override;
  - D2: **decided 2026-09-27: characters per second** (WP47). The first real job's short segments failed
    `PACE_FAST` falsely on every attempt;
  - D3: a compact `get_results` (a segment filter; words off by default) (§7.5);
  - D4: run the service from a pinned checkout, or freeze merges that touch the engine while narrating;
  - D5: whether dev GPU work pauses while the owner narrates;
  - D6: the canary on every Qwen load (12–19 s each) or once per worker process;
  - D7: QA rule changes (a probable-name warning in `check_text`; insertion runs; a length-aware speaker
    margin);
  - D8: loudness: deliveries are at -23 LUFS; confirm what the consumer expects.
- **The Avast Auto-Sandbox exception** for the projects folder, or Auto-Sandbox off: the durable fix for
  the hangs.
- **DC-14 and DC-15** (plan.md §1.5): lead gap-fills the owner may overrule.
- **4a in the owner's list:** cross-job grouping (§4 item 3); the lead recommends leaving it out of v1.
- Whether GitHub's private vulnerability reporting is the route `SECURITY.md` should name.
- Later:
  - one run of the daemon under Claude Code itself, once `wp/30-escape` merges (the smoke run's `mcp`
    Python client killed the daemon on exit);
  - the GitHub description, which still says voices are "locked" (WP44);
  - gates H2 to H4.

## Next steps (the lead)

1. **The narration voice is a-warm-s101** (the owner's choice, 2026-09-27): designed by the lead from the
   owner's description, and measured 21:16 in the real store `.dev/stores/service`: reliable up to 301 spoken
   characters (the ladder stopped at 300, as `narration.toml` notes), pace 154.7 wpm + 15.0 per 100
   characters, pace tolerance 0.111, anchor p5 0.989. Clip, sidecar (the transcript is the corpus's design
   text) and allowlist entry are local (`.dev/voices/`, `narration.toml`). The earlier voice, d2 re-designed
   seed 2006, is a measured spare, kept local only. The owner's guide is the gitignored
   `.dev/scratch/first-narration.md`, and the narrating agent's brief is `.dev/narration-brief.md`. The
   owner's session connected at 21:24. **The rest of the ladder** (350–560) runs later in 30-minute stages
   (`.dev/service-setup/staged_ladder.sh`; remove its `STOP` file first), never while the owner narrates.
   **The owner's `.mcp.json`** should start the server through the venv's interpreter
   (`<repo>\.venv\Scripts\python.exe -m narration.mcp --config <repo>\narration.toml`), not
   `uv run … narration-mcp`, whose launcher `.exe` is what Avast sandboxed (`AGENTS.local.md`).
2. **Merge once each is green and verified:** PR #40 (texts only) whenever its review is clean; after the
   owner's session, PR #37 (the daemon fix), PR #41, then PR #39 (WP34; it removes the "not in this build"
   marks for its tools). Run the full suites then; the agents run only targeted tests while the owner
   narrates.
3. **After M1:** WP34, WP35, WP38, WP39 and WP45, then Wave 3 (WP40–WP44), with the follow-ups below.
4. **WP37 follow-ups:** `doctor` checks the material manifests (WP33's corpus loader is on `main`); a
   single-item evict in the store for what `verify` finds damaged (the store's area); `reset_workers`
   stays a proposal.

## While the owner narrates (lead rules, from the readiness audit)

- **Hold any merge that changes the engine's identity**: `workers/qwen3tts/` (code or lock), `workers/common`'s
  version or lock, `material/canary/`, `src/narration/engine/profile.py`, or anything feeding the render,
  delivery or analysis keys. Each forces a repin or re-QA and orphans the owner's measurement or cached work.
  Other merges change what a newly started server or daemon runs, since the service runs from the main
  checkout (D4); restart the daemon only deliberately.
- **Dev GPU work waits** while `.dev/stores/service/run/daemon.json` shows a job or a loaded worker.
- **Never edit `narration.toml`'s `[measurement]`**; after any other edit, stop the daemon and reconnect.

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

- **After WP32 and WP36 are both merged:** `serve` passes WP32's installed QA pins as `AnalysisPins`
  (until then a plan counts every analysis as needed; WP36's note).
- **WP32's review, deferred:** the editable worker packages (`narration-worker`,
  `narration-worker-qwen3tts`) are not fingerprinted, so a worker code edit with an unchanged version is
  invisible to the hash (the canary would still catch an audio change); `[engines.qwen3_base]
  x_vector_only_mode` must stay `false` (the canary clones with `false`), so document it or make the
  canary follow it; a canary-text change forces new profile ids (a canary-only re-pin under the same id
  would need a store change).
- **WP48, a failure audit (the owner asked, 2026-09-27):** collect every failed or retaken take across jobs,
  with its reasons, for the owner to audit (`narration-admin failures`, an export, a report section, and
  WP46's view). The store already keeps them; what is missing is a cross-job view and retention that waits
  for the audit.
- **WP46, a frontend (the owner asked, 2026-09-27):** plan a view of jobs, takes, voices, the allowlist and the
  engine and daemon state, **after the other work**. Write the plan for the owner's approval first; build nothing before it.
- **WP45 (the owner asked, 2026-09-27; DC-17):** `narration-admin voices allow <clip.wav>` adds a clip
  designed elsewhere to `[voices] allow_sha256`; operator-only, never an MCP tool. Any time after M1.
- **WP32, after M1:** a contract field `CanaryPin.calibration: tuple[float, ...] = ()` (unhashed), so
  `engine show` can print the calibration similarities and a floored threshold is recorded, not inferred.
- **WP34:** show a design candidate's `similarity_pass` as `CANARY_MISMATCH` (info); the VoiceDesign
  gate is a weak alarm by nature (lead decision, plan.md §9, 2026-09-27).
- **WP36, after M1:** §7.3's all-cached submit completes at once (a WP31-owned "complete from the cache
  or say no" function called at submit).

- **Pace on short segments (WP40, from the M1 smoke run):** two takes of a short segment failed
  PACE_FAST against a two-rung ladder's trend; check the trend's slope on a real measurement and plant
  short-segment pace cases.

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
