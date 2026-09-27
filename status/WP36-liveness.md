# WP36-liveness A dead daemon, a mistyped transcript, and cached analyses in the plan
State: blocked:paused-by-owner        Updated: 2026-09-27 (PR #41 review, round 2, paused mid-way)

## PR #41 review, round 2 (paused by the owner; WIP commit, not ready for review)

Done (targeted tests passed before the pause: tests/backend, tests/mcp, tests/daemon without test_process,
test_owned and test_entry: 471 passed; tests/jobs, tests/engine/test_daemon_backend.py, tests/keys: 233 passed;
tests/lint and test_package: 57 passed; ruff check, ruff format --check and basedpyright clean on the changed files):
- L2: `start_detached` records `run/launch.json` (pid, launch time; temp name and rename). `_revive` asks for no
  daemon while a launch is younger than `START_WINDOW_S` = 90 s (`takeover_wait_s` 60 s + 30 s interpreter start
  and imports; BELIEVE, to be measured) and no status was written since. Tests: repeated polls, cancel then
  get_job, a stuck start (one launch per window), a start that died (replaced at once).
- L3: explicit `SHIELDED_TOOLS` (server.py) and `RETRYABLE_TOOLS` (descriptions.py, with get_job); get_job's
  description gets the backoff rule plus one clause; readOnlyHint stays true.
- L4: "asked for one"; the log line only with autostart on; the stale "(daemon state: ...)" is dropped when the
  poll ended on a status change.
- L5: a negative age within the grace counts as just written; the 30 s grace is labelled BELIEVE.
- L6: the wait_s test counts polls instead of timing (a mutation check showed 10 polls against at most 1).
- F1/F2: `details.transcript_mismatch.rewrites`, a hint phrased as steps, `transcript + "
"` as a spelling;
  tests for the double space, the leading space, the ellipsis and the measured trailing newline. CHANGELOG and
  the measures.py docstring say which directions are caught.
- F3: `AnalysisPins.aligner_revision` (no key input), filled by `InstalledPins`; `get_server_status` reports it
  before the benchmark has run.
- F5: one builder, `narration.jobs.plan.analysis_key_inputs`, used by `Stages.key_inputs` and
  `planning.analysis_key_inputs`. Tests: a parity test over the real `Stages.key_inputs` (with hints and an exact
  span), and one key pinned at the value the code computed before the refactor.

Not done:
- L1, NOT BUILT, as asked ("stop and tell me before building"). A `stopped` status is not only an operator's
  decision, so option (b) would leave these queued jobs stuck:
  - (a) the control loop raising: `log.exception(...)`, then EXIT_ERROR; `_run_held`'s finally still writes
    `stopped` (daemon/service.py, `_run_held`);
  - (b) the supervisor factory raising: that goes through the same finally;
  - (c) a job submitted during an operator stop's in-flight segment: the daemon its submit launched gives up
    after `takeover_wait_s` (60 s) while the old one finishes the segment; then the old one writes `stopped`.
    The design intends that job to run: a stop posted before a daemon's launch is not honoured by it;
  - (d) an idle exit that takes more than 60 s after `has_work` said no, while a job is submitted (unlikely);
  - (e) the submit's ensure failed and the old daemon exits cleanly (the caller was told at submit).
  - Proposed (b'): a queued job under `stopped` is an operator's stop only if `store.commands_since(status.started_at)`
    has a `stop`/`stop_now` answered `stopped: true` AND the job was created before that stop was posted.
    Otherwise start one (autostart off: the note only). Waiting on the lead's decision.
- The design text proposals (§4/§4.1 for L1 and L2's launch marker; §3.2/§7.3/§14 for F4, that
  VOICE_NOT_MEASURED's hint depends on a nearby spelling being measured) are still to be written into this file.
- A follow-up to measure start-to-status time (for the 30 s grace and the 90 s window).
- Merge main when the lead says (conflicts expected in descriptions.py from #40, and maybe daemon/start.py from #37),
  then the full suite, and set State to review.


Branch `wp/36-liveness`, rebased onto `main` 78b2fae (docs only since 10270ba). Fixes three readiness-audit findings in WP36's area (the
front-end's backend): 1.3 (mcp-3, rt-4), and from section 2 cf-3/mcp-6 and cf-13/mcp-12/triage-5.

## Done
- **`get_job` and `cancel_job` notice a job whose daemon has gone** (mcp-3, rt-4).
  - `NarrationBackend._revive(job)`: for a job that is `queued`, `running` or `cancelling`, when
    `launcher.running()` says no daemon runs (None, or a status `stopped`), it calls `launcher.ensure()`
    (the existing API only; idempotent; with `[daemon] autostart` off it starts nothing).
  - **What the new daemon does with the job** (read in `daemon/sweep.py` and `daemon/seam.py:return_job`,
    and run in the tests): its start-up sweep, holding the singleton, moves a job left `running` back to
    `queued` by compare-and-set, with the phase cleared and its items, round and progress kept, and the
    message `queued again: the daemon that ran it (pid N) stopped without finishing it`. The job engine then
    takes it again and walks the rounds it finished from the cache (`jobs/engine.py`, "A job taken again").
    A job left `cancelling` is finished as `cancelled`. A `queued` job is left alone and runs in its turn.
  - A daemon that is `stopping` is left alone. One exiting for being idle looks for work once more (§4.1).
    One the operator asked to stop is let stop, and the next call after it has gone starts another.
  - **`get_job`**: the check runs once per call, before the long-poll, and counts against `wait_s`. The job is
    read again after a start. The reply's `message` is the job's message plus a note with the daemon's state
    and what happens next, for example: "… No daemon was running this job (daemon state: stopped), so get_job
    started one; the job is kept, and the new daemon puts it back on the queue, reusing what it finished from
    the cache." With `autostart` off, the note says so and names `narration-admin daemon start`. The
    stored record is unchanged.
  - **A start that fails** is `DAEMON_UNAVAILABLE` with its existing semantics:
    - it is retryable, with the launcher's `retry_after_s` (or DC-2's default on the way out);
    - the message names the job, its status and the launcher's reason;
    - `details` has the launcher's details plus `job_id`, `job_status` and `daemon_state: "stopped"`;
    - the hint is built in the error, not in `descriptions.py`: "Run 'narration-admin daemon start' in a
      terminal, then call get_job again: <what happens to the job>";
    - a start that can never work here (`UnsupportedPlatform`) is not retryable and keeps its own hint.
  - **`cancel_job`** asks for a daemon when it leaves a job `cancelling` with none running, so the sweep
    finishes the cancel. If none can start, it still answers `{cancelling, completed: false}` (the cancel is
    recorded, and nothing runs the job), logs a warning, and the next `get_job` gives the error and hint.
  - **The start-up grace.** A `queued` job written less than `DAEMON_START_GRACE_S` (30 s) ago is left to
    the daemon its submission asked for. Without the grace, the backend's real-daemon test caught `get_job`,
    called right after `submit_job`, asking for a second daemon while the first was still starting (a second
    start that exits at once).
- **`VOICE_NOT_MEASURED` names a transcript that differs from the measured one** (cf-3, mcp-6).
  - The measurement does not keep the transcript it verified. The store finds measurements only by
    `voice_hash` (the Store protocol has no lookup by `clip_sha256`). So "this clip, any transcript" cannot
    be looked up without a contract change (below).
  - What is built instead is `measures.measured_nearby`. It tries the spellings that a slip in sending
    the transcript makes:
    - the edges trimmed;
    - the whitespace collapsed (newlines, tabs, NBSP and double spaces);
    - quotes, primes, dashes and the ellipsis made plain, or straight quotes made typographic;
    - each whitespace form combined with each punctuation form.

    That is at most 8 hashes and lookups, only on the refusal path, through the `Measurements` seam's
    `voice_hash` and `require`.
  - When one of these spellings is measured for this clip, the error changes:
    - `field` is `voice.transcript`;
    - the message ends with "…differs from the one sent only in whitespace, first at character N";
    - `details.transcript_mismatch` holds `measured_voice_hash`, `differs_in`, `first_difference` (an
      index into the transcript sent, counting from 0), `sent` and `measured` (each side's character there,
      as `U+000A LINE FEED` or "the end of the text"), `sent_chars` and `measured_chars`;
    - the hint says to send the transcript exactly as measured and not to measure again, since that would
      make a second voice with its own cache.
  - Otherwise the error keeps `field: voice`, and the hint (`measures.MEASURE_HINT`) says to measure, or,
    if the clip was measured before, to send its transcript exactly as then. `details` has `voice_hash`,
    `clip_sha256` and `engine_profile_id` either way.
  - Neither transcript is quoted in the error, and nothing is logged. The tests assert that the transcript
    appears nowhere in the message, hint or details. The clip is still not read (`read_paths` is empty).
- **`dry_run` counts cached analyses** (cf-13, mcp-12, triage-5).
  - `backend_for` passes `InstalledPins(config)`. It is built from `narration.engine.qa.qa_pins` and
    `aligner_method_id`, the functions the installed job engine keys analyses with, plus `Scorer(config.measurement)`'s
    `profile_version` and `number_reader`.
  - It is None while the QA models are not installed. The plan is then the upper bound, as before, and the
    next call looks again. Once the pins are found, they are kept.
  - Side effect: `get_server_status`'s `alignment.method_id` now names the configured aligner in
    `narration-mcp`. Before, it was null, because the front-end's store is opened without a method id.
  - The join test (`tests/engine/test_daemon_backend.py`) now plans the request again after the installed
    engine ran it, and expects `segments_cached 1, analyses_needed 0, renders_needed 0`. Without the fix it
    fails with `(0, 1, 0)`, which was checked by stashing the fix.
- **No cache key and no engine profile changed.** Nothing touched `workers/`, `material/`,
  `engine/profile.py`, `keys/`, `narration.platform`, the launcher's API or `descriptions.py`.
- One CHANGELOG line per fix, under Unreleased, Changed.

## Tests
- `uv run python -m pytest tests/backend` → 131 passed, 30 of them new: `test_liveness.py` 15, `test_pins.py` 3,
  `test_transcripts.py` 12. Ran three times in a row after the grace fix: 131 passed each
  time.
- `uv run python -m pytest tests/engine/test_daemon_backend.py` → 1 passed (about 15 s).
- `uv run python -m pytest` (the full default suite) on the rebased branch (fe36871, code as reviewed) →
  3737 passed, 15 skipped (Windows symlinks need Developer Mode; POSIX-only checks), 19 deselected, in 8.5 min.
  It ran 21:21–21:29 local, under the lead's "after 21:20, run everything" message. The later message (owner
  narrating: targeted tests only) arrived after it had finished, and nothing heavy has run since. The same
  suite also passed before the rebase (3737 passed, finished about 21:09).
- `ruff check`, `ruff format --check`, `basedpyright` (whole project, same run): clean, 0 errors.
- `tools/check_private.py --commits main..HEAD --base main` → exit 0; `tools/check_tracked.py` → exit 0 (after
  the rebase).

## Decisions made (and why)
- **The note on `get_job` goes in `message`, not a new field.** get_job's success reply has no `details`,
  and contracts are the lead's. `message` is the one text field the caller reads in every reply. A
  structured field is proposed below.
- **`get_job` raises `DAEMON_UNAVAILABLE` when no daemon can start** (the lead's test list), rather than
  returning `queued` forever. The error carries the job's status in `details`, so the caller still sees it.
- **`cancel_job` does not raise when no daemon can start.** The cancel is recorded, and with no daemon
  nothing runs the job, so `{cancelling, completed: false}` is true. Raising after the write would read as
  "the cancel failed". `get_job` explains the stuck `cancelling`.
- **No start while a daemon is `stopping`.** Starting one would override an operator's
  `narration-admin daemon stop` at once (a daemon launched after a stop does not honour it, §4.1). The next
  call after it exits does start one, as a resubmission already did.
- **The grace is for `queued` only.** A job is `running` or `cancelling` only once a daemon that had written
  its status took it. So none running means that daemon has gone, and waiting longer would only delay the
  recovery. 30 s is BELIEVE: interpreter start, imports and opening the store take seconds, and longer when
  Avast inspects a new process first. After the grace, a queued job that no daemon serves is revived.
- **No full lookup by clip for the transcript check.** It needs a contract change, which is proposed below.
  Trying the spellings of the likely slips uses only existing APIs and covers the mismatches the audit
  names: a trailing newline, a double space, and curly apostrophes. It cannot catch a retyped word; for
  that the hint says to send the measured transcript exactly.
- **check_text gets no info flag.** No flag code fits, and inventing one is a contract change (below).
- **Touched a WP32 test file.** `tests/engine/test_daemon_backend.py` is the joint WP32–WP36 test (its
  docstring says so), and the audit's triage-5 fix names it. The change adds three lines at the end of its
  one test. A second end-to-end test elsewhere would have repeated its 15 s measure run.

## Contract change requests
- **`get_job` output: an optional `daemon` object** `{state, started: bool, hint}` (schemas `_get_job_output`),
  present only when the call found no daemon for an active job. It would replace the note in `message`.
- **`MeasurementRecord.transcript`** (design §6 already lists the "verified transcript" on Measurement), plus a
  **Store method `measurements_of_clip(clip_sha256)`**. With both, `VOICE_NOT_MEASURED` could name any
  transcript mismatch for a measured clip, not only whitespace and punctuation slips. The field must be
  optional, so that the owner's existing measurement still reads. No key covers the record's content (BELIEVE,
  from `keys.measurement_key`'s inputs), so it would not re-key anything; please confirm.
- **A flag code for `check_text`** (for example `VOICE_TRANSCRIPT_MISMATCH`, info) and a top-level `flags` on
  its output, to report the same near-miss before any submit (cf-3's "check_text adds an info flag").
- **Design text** (the lead's): §4 says the daemon is started "by the first submission, or by
  `narration-admin daemon start`". Now `get_job` and `cancel_job` also start it for an active job that no
  daemon serves. §7.4 could list `DAEMON_UNAVAILABLE` as get_job's one retryable error.

## Dependency requests
- none

## Questions for the lead / owner
- **`get_job` keeps `readOnlyHint: true`** although it may now start the daemon. Setting it false would put
  get_job into `SHIELDED_TOOLS` (derived from `read_only`), which would break "cancelling the wait ends the
  wait". I read the start as the service restoring its own invariant (a daemon serves while jobs are
  active) for work the caller already asked for. Keep it?
- **For wp/36-guidance:** get_job's description could say that it restarts a stopped daemon for an active
  job, and that a `DAEMON_UNAVAILABLE` from it means "run narration-admin daemon start". I did not edit
  `descriptions.py`.
- **`measure_voice` with a mistyped transcript** still queues a new measurement. It is the costly path
  cf-3 warns about, and the audit did not ask for a change there. A refusal or warning there would need
  a design decision.

## Next
- Review. On a merge: the lead's calls on the contract change requests and questions above. The guidance
  branch could describe get_job's restart.
