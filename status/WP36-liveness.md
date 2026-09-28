# WP36-liveness A dead daemon, a mistyped transcript, and cached analyses in the plan
State: review        Updated: 2026-09-28 (PR #41, round 4 done; CI green at 2756ff6)

Branch `wp/36-liveness`, PR #41. It fixes three readiness-audit findings in WP36's area (the front-end's
backend): 1.3 (mcp-3, rt-4), and from section 2 cf-3/mcp-6 and cf-13/mcp-12/triage-5. Four review rounds
followed. This file says what the branch does now; "Superseded" at the end lists what earlier rounds said
that no longer holds. Since 2026-09-28 the owner needs the machine: no local pytest and no basedpyright ran,
and GitHub's CI is the test runner (the lead's instruction).

## Done

### An active job whose daemon has gone (mcp-3, rt-4; `NarrationBackend._revive`)
- `get_job` and `cancel_job` check once per call, before get_job's long-poll (the check counts against
  `wait_s`), for a job that is `queued`, `running` or `cancelling`. They check in this order:
  1. **A queued job written less than `DAEMON_START_GRACE_S` (30 s, BELIEVE) ago** is left to the daemon its
     submission asked for.
  2. **A daemon that runs** (`idle`, `busy` or `stopping`) is left alone. One exiting for being idle looks for
     work once more; one asked to stop is let stop.
  3. **A queued job left in the queue by a stop posted after it was queued** (`operator_stop`; the stop comes
     from `narration-admin daemon stop`, or from `narration-admin install` after a repair) starts no daemon. The
     note says so, and says the job runs on the next start (`narration-admin daemon start`, or the next
     submit_job while autostart is on). The rule is the lead's option (b'), with one adaptation:
     - A stop counts if it is a `stop` or `stop_now` answered `stopped: true`, posted after the job was created.
       A stop in the job's own millisecond counts as after it, as `posted_after_launch` has it for a launch.
     - It must also have been answered at most `STOP_TO_STOPPED_S` (30 s, BELIEVE) before the `stopped` status
       was written. The adaptation: a `stopped` status keeps no start time (daemon/status.py writes
       `started_at=None`), so the stop's answer time stands in for "since that daemon started".
     - A `stopped` status alone is not an operator's decision: a daemon whose control loop or worker supervisor
       failed writes it too, as does an idle exit. The queued jobs of those get a daemon.
     - Only store times are compared. What remains: a daemon launched by a later submission that fails within
       30 s of the stop's answer makes older jobs read as stopped. The later submission's own job still starts a
       daemon on its next poll, and that daemon runs them all.
  4. **A launch in progress** (`daemon.start.check_launch` over `run/launch.json`: launched less than
     `START_WINDOW_S` = 90 s ago, BELIEVE, with no status written since): while the launched process runs, no
     other daemon is asked for (the note says one is starting). Once that process has gone without writing a
     status, the start failed. It died before it held the singleton, or failed through its `finally`, which
     writes `stopped` with no start time. `get_job` then answers `DAEMON_UNAVAILABLE`:
     - it is retryable, with `retry_after_s` the rest of the window, at least `DAEMON_RETRY_S`;
     - `details.log` is the store's `logs/daemon.log`, plus `launched_pid` and `launched_at`;
     - the hint says to run `narration-admin daemon start` in a terminal (or with `--foreground`, to see why it
       exits).

     It does not launch again at once. Past the window, the next call launches one again.
     - The launched process decides between starting and failed. It must still exist, and must have been
       created at the launch (`sweep.launch_alive`: from 2 s before `launched_at` to 10 s after it; a process
       created outside that span reused the pid).
     - So a `stopped` status newer than the launch, written by the exiting daemon while the launched one waits
       for the singleton, stays "starting".
     - Under a venv the pid is the launcher's, which ends with its daemon: BELIEVE from design §4.1. KNOW
       (2026-09-28, on this machine): psutil reports a killed venv python launcher as gone, and its child with
       it.
  5. **Otherwise it asks the launcher for a daemon** (`ensure`; with `[daemon] autostart` off it starts
     nothing, and the note says who must start one). The new daemon's start-up sweep puts a job left `running`
     back on the queue (its items, round and progress kept; the engine walks what it finished from the cache)
     and finishes a `cancelling` job as `cancelled`.
- **The note** goes in get_job's `message`: the job's message plus what was found and done. The note:
  - says "asked for one" (the launcher may start nothing);
  - names the daemon's state only while the job's status is the one seen then;
  - is logged as "asked the launcher for one" only with autostart on.
- **A start that fails** is `DAEMON_UNAVAILABLE`:
  - it names the job and its status, with the launcher's reason;
  - `details` has `job_id`, `job_status` and `daemon_state`;
  - the hint is to run `narration-admin daemon start` in a terminal;
  - `UnsupportedPlatform` is not retryable and keeps its own hint.

  `cancel_job` records the cancel and answers `{cancelling, completed: false}` even then (it logs a warning).
- **`run/launch.json`** (`daemon.start.record_launch`; `StoreLayout.launch_json_path()` names it) holds the pid
  `spawn_detached` got back and the launch time.
  - It is written only after `spawn_detached` returns, and PR #37's `spawn_detached` returns only for a daemon it
    resumed. So a refused start records nothing and holds back no later launch.
  - It is the service's operational state, never a caller's (sections 0.2, 2).
  - It is advice, read and written without a lock: two front-ends that poll in the same instant may both
    launch, and the singleton makes that harmless (start.py's docstring says so).
- **The MCP surface.**
  - `SHIELDED_TOOLS` (server.py) and `RETRYABLE_TOOLS` (descriptions.py) are explicit sets, not derived from
    `read_only`.
  - get_job is in `RETRYABLE_TOOLS`, so its description states the backoff rule. Its description also says:
    "If the daemon died, get_job asks for one again (with [daemon] autostart on), and its message says what was
    done." get_job keeps `readOnlyHint: true`.
  - Measured lengths (2026-09-28, with the test's 99,999-day retention): get_job 1382 characters, the longest
    get_results 2013, within `CLIENT_TEXT_LIMIT` (2048). With the default retention: 1377 and 2008.

### `VOICE_NOT_MEASURED` names a transcript that differs from the measured one (cf-3, mcp-6; `measures.py`)
- The measurement keeps no transcript, and the store finds measurements only by `voice_hash`. So the refusal
  path tries the spellings a slip makes (`transcript_variants`):
  - the edges trimmed;
  - the whitespace collapsed;
  - a trailing newline added;
  - quotes, dashes and ellipses made plain;
  - straight quotes made typographic;
  - each whitespace rewrite with each punctuation rewrite.

  That is at most 9 spellings, each one hash and one lookup.
- When one is measured for this clip:
  - `field` is `voice.transcript`;
  - `details.transcript_mismatch` holds `measured_voice_hash`, `rewrites` (the names of the rewrites that give
    the measured transcript), `differs_in`, `first_difference`, `sent`, `measured`, `sent_chars` and
    `measured_chars`;
  - the hint gives the rewrites as steps, and says not to measure again.
- Otherwise the general hint (`MEASURE_HINT`). Neither transcript is ever quoted.
- Which directions are caught:
  - caught: whitespace the transcript sent has and the measured one has not; typographic marks sent for plain
    ones; straight quotes sent for typographic ones; a trailing newline the measured one had;
  - not caught: other whitespace the measured one had, and a typographic dash or ellipsis it had.

### `dry_run` counts cached analyses (cf-13, mcp-12, triage-5)
- `backend_for` passes `InstalledPins(config)`: the QA models and the aligner the installed engine keys analyses
  with (`qa_pins`, `aligner_method_id`), plus the scorer's profile and number reader. It is None until the QA
  models are installed (the plan is then the upper bound).
- A plan and the engine build the analysis key's inputs with one function, `narration.jobs.plan.analysis_key_inputs`.
  A parity test runs the real `Stages.key_inputs`, and one key is pinned at the value the code computed before
  the refactor.
- `get_server_status`'s `alignment` names the configured aligner's method id. Before the benchmark has run, it
  also gives its pinned revision (`AnalysisPins.aligner_revision`, which enters no key).

### Also
- The wait_s test counts polls instead of timing. A mutation check showed 10 polls against at most 1.
- No cache key, engine profile or contract changed. `workers/`, `material/`, `engine/profile.py`, `keys/` and
  `narration.platform` are untouched.
- Outside WP36's area:
  - `daemon/start.py` (the launch marker) and `daemon/sweep.py` (`launch_alive`);
  - `store/layout.py` (`LAUNCH_JSON`, `launch_json_path`);
  - `jobs/plan.py` and `jobs/stages.py` (the shared builder);
  - `mcp/descriptions.py` and `mcp/server.py` (the explicit sets);
  - tests in `tests/daemon/` and `tests/engine/test_daemon_backend.py`.
- The CHANGELOG has a line per change under Unreleased.
- origin/main was merged three times (never rebased): at 1b05644 (PRs #40, #42 and #37), at 01f3102 (WP34 and
  design 5.15), and at 21dba8b (WP49). None of the last two conflicted.

## Tests
- Round 4 (2756ff6, merged with origin/main 21dba8b): push run 36421819386 and pull-request run 36421825243,
  both green in every job:
  - tests (windows-latest): 3975 passed, 11 skipped, 20 deselected, 3 warnings. The warnings are WP49's notes
    that the runner's Job Objects forbid breakaway, from test_process.py and test_windows_processes.py; none is
    from this branch;
  - tests (ubuntu-latest): 3893 passed, 16 skipped, 20 deselected;
  - types (basedpyright), lint (ruff), schemas, tracked, and worker qwen3tts on both OSes.

  No skip is in a file this branch touches. Locally only ruff check and ruff format --check on the changed
  files, `tools/check_private.py` and `tools/check_tracked.py` (both exit 0).
- Round 3 (732f254): push run 36413614355, pull-request run 36413619558, green in every job. Windows: 3909
  passed, 11 skipped. Ubuntu: 3827 passed, 16 skipped. No skip was in a file this branch touches.
- New in round 4:
  - tests/backend/test_liveness.py:
    - a launched daemon that exits before it serves;
    - one that failed through its `finally`;
    - a `stopped` status while the launched process still runs;
    - `launch_alive`'s pid-reuse guard;
    - `check_launch`'s states;
    - a stop in the job's own millisecond;
    - the layout path.
  - The launch tests now run a real sleeping process per launch (`LaunchingPlatform`), so "pid alive and still
    starting" is a real process.

## Decisions made (and why)
- **The note goes in `message`**, not a new field: get_job's reply has no `details`, and contracts are the
  lead's. A structured field is a contract request below.
- **`cancel_job` does not raise when no daemon can start.** The cancel is recorded; raising after the write
  would read as "the cancel failed". `get_job` explains the stuck `cancelling`.
- **No start while a daemon is `stopping`**: that would override an operator's stop at once.
- **The grace is for `queued` only**: a `running` or `cancelling` job was taken by a daemon that had written its
  status, so none running means it has gone.
- **A failed start is an error, not another launch**: the next daemon would most likely fail the same way, and
  the error names the log. The window still bounds the wait; after it, one more launch.
- **An unreadable process creation time (access denied) counts as alive**: the window bounds the wait, and no
  start is called failed on a guess.
- **Touched a WP32 test file**: `tests/engine/test_daemon_backend.py` is the joint WP32–WP36 test, and the
  audit's triage-5 fix names it (three lines).

## Contract change requests
- **`get_job` output: an optional `daemon` object** `{state, started: bool, hint}` (schemas `_get_job_output`),
  present only when the call found no daemon for an active job. It would replace the note in `message`.
- **`MeasurementRecord.transcript`** (design §6 lists the "verified transcript") plus a **Store method
  `measurements_of_clip(clip_sha256)`**, so `VOICE_NOT_MEASURED` could name any transcript mismatch for a
  measured clip. The field must be optional. No key covers the record's content (BELIEVE, from
  `keys.measurement_key`'s inputs).
- **A flag code for `check_text`** (for example `VOICE_TRANSCRIPT_MISMATCH`, info) and a top-level `flags` on
  its output, to report the same near-miss before a submit (cf-3).

## Dependency requests
- none

## Questions for the lead / owner
- **`measure_voice` with a mistyped transcript** still queues a new measurement, the costly path cf-3 warns
  about. The audit did not ask for a change there; a refusal or warning would need a design decision.
- **`narration-admin install`'s stop** says "the next job starts a fresh one", but under (b') the jobs queued
  before it wait for the next start, as after `daemon stop`. That matches the stop's meaning; say if install's
  stop should instead let queued jobs restart the daemon.

## Design text proposals (docs/design.md is the lead's; nothing edited there)
- **§4, who starts the daemon.** Proposed: "by a submission, by `narration-admin daemon start`, or by `get_job`
  or `cancel_job` for an active job that no daemon serves: a job left `running` or `cancelling`, or a `queued`
  job that no stop posted after it left in the queue (§4.1). A launch is recorded in `run/launch.json` (the
  launched pid and the time). While a launch is younger than the start window (90 s: the singleton takeover
  wait of 60 s plus start-up; BELIEVE until measured) and no daemon status has been written since, `get_job`
  and `cancel_job` launch no other while the launched process runs, and answer `DAEMON_UNAVAILABLE` (naming
  the daemon's log) once it has gone: a start costs one launch per window, even when it hangs or fails."
- **§4.1, what a stop means for queued jobs.** Proposed addition: "A stop answered `stopped: true`
  (`narration-admin daemon stop`, or `narration-admin install` after a repair) holds for the jobs queued before
  it was posted: `get_job` starts no daemon for them, and says they run on the next start. A job submitted
  after the stop starts a daemon as any submission does, even while the stop's in-flight segment finishes. A
  `stopped` status alone is not an operator's decision: a daemon whose control loop or worker supervisor
  failed, and one that exited idle, write it too, and their queued jobs get a daemon. The front-end tells them
  apart from the store: the stop's answer is written just before `stopped` (within 30 s, BELIEVE)."
- **§15, the store's layout.** Proposed line: "`run\launch.json`: the last daemon launch (the pid
  `spawn_detached` got back and the launch time), written only for a daemon the platform let run; advice to
  the next launcher, the service's operational state (§0.2)."
- **§3.2, a voice's transcript.** Proposed addition: "The voice hash covers the transcript character for
  character, so a clip measured under one transcript and sent with another spelling is another voice. When a
  spelling a slip makes (the edges trimmed, the whitespace collapsed, a trailing newline added, quotes, dashes
  and ellipses made plain, or straight quotes made typographic) is measured for the clip,
  `VOICE_NOT_MEASURED` says so (§7.3)."
- **§7.3, submit_job's `VOICE_NOT_MEASURED`.** Proposed: "`field` is `voice.transcript` and
  `details.transcript_mismatch` is {`measured_voice_hash`, `rewrites`, `differs_in`, `first_difference`,
  `sent`, `measured`, `sent_chars`, `measured_chars`} when a nearby spelling of the transcript is measured for
  this clip; the hint gives the rewrites as steps and says not to measure again. Otherwise `field` is `voice`,
  with the general hint. Which hint a caller gets depends on whether such a spelling is measured. Neither
  transcript is quoted."
- **§14, the code table.** `VOICE_NOT_MEASURED`'s hint: "Run measure_voice first; or, when
  `details.transcript_mismatch` is present, send the transcript as the clip was measured with it (its
  `rewrites`), and do not measure again." `DAEMON_UNAVAILABLE` gains get_job as a tool that returns it, with
  `details.log` when a launched daemon exited before it served.

## Follow-ups
- **The stopped daemon's launch time, or why it stopped, in `run/daemon.json`** (approved by the lead as a
  follow-up, not in this PR; WP30's writer and a contract change). It gives `operator_stop` an exact signal
  instead of the 30 s reading, and lets `check_launch` tell a failed start from an operator's stop.
- **Measure start-to-status time** (from a launch to the daemon's first status write, on Windows with the virus
  scanner on and off, cold and warm). It would turn four BELIEVE values into measurements:
  - `DAEMON_START_GRACE_S` (30 s);
  - `START_WINDOW_S` (90 s);
  - `STOP_TO_STOPPED_S` (30 s);
  - `sweep.SPAWN_TOLERANCE_S` (10 s).

## Next
- CI green, then the lead merges.

## Superseded (what earlier rounds said that no longer holds)
- Round 1: "`launcher.running()` says no daemon runs (None, or a status `stopped`), so it calls `ensure()`".
  Now the grace, the operator's stop, and the launch check come first (Done, steps 1 to 4).
- Round 1: the note "so get_job started one". Now "asked for one".
- Round 1: `details.transcript_mismatch` without `rewrites`, and the hint "send the transcript exactly as
  measured". Now `rewrites`, and the hint gives them as steps.
- Round 1: "at most 8 hashes and lookups". Now at most 9, with the trailing newline added.
- Round 1: "Nothing touched `narration.platform`, the launcher's API or `descriptions.py`". Rounds 2 to 4
  changed `descriptions.py`, `daemon/start.py` and `daemon/sweep.py`; `narration.platform` is still untouched.
- Round 1's local test counts (3737 passed at fe36871) are history; CI is the record now.
- Round 3: "a launched daemon that fails through its `finally` is replaced after the window, not at once". Now
  it is a failed start: `DAEMON_UNAVAILABLE` with the log, within the window.
- Round 3: "the operator's stop is `narration-admin daemon stop`". `narration-admin install` posts one too, and
  the note names both.
- Answered and dropped:
  - get_job keeps `readOnlyHint: true` (the lead: keep it; L3);
  - get_job's description now covers the restart (round 4 wording).
