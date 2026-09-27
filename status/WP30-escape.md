# WP30-escape: the detached daemon must leave every Job Object of its client
State: review        Updated: 2026-09-27T20:25+01:00

Base: rebased onto `main` at 9bb7e96 (clean). Branch `wp/30-escape`, five commits.

## The bug, reproduced (KNOW; spike k, `spikes/k-job-escape/`)
- Under the MCP Python SDK's `stdio_client` (mcp 2.2.0) with the server started as the venv's `python.exe`,
  `start_detached` succeeded, the real daemon served, and, asked with the SDK's own job handle
  (`IsProcessInJob`), **the daemon's launcher and the daemon were inside the client's kill-on-close job**.
  When the client exited, `run/daemon.json` still said `idle` and the process was gone: no stop, as the
  lead saw in `.dev/stores/m1-smoke`.
- The same happens with a hand-made `KILL_ON_JOB_CLOSE` job, whether the server is assigned after the
  spawn (as the SDK and libuv do) or from its first instruction.
- With the server run as the base interpreter (no launcher) under the same client, `CreateProcess` refused
  and `DAEMON_UNAVAILABLE` (`breakaway_refused`) came back, as section 4.1 describes.

## The diagnosis (KNOW, measured in isolation: 13 nested-job cases)
- The venv launcher's job has `KILL_ON_JOB_CLOSE | SILENT_BREAKAWAY_OK` and holds only the interpreter.
- Windows walks the caller's jobs from the innermost outward: the child leaves each job that allows
  breakaway (`BREAKAWAY_OK` or `SILENT_BREAKAWAY_OK`) and is placed in the first that does not, with that
  job's ancestors. `CreateProcess` fails with access denied **only when the innermost job forbids it**.
- So: the SDK's job (forbids) around the launcher's (allows, silently) → the call succeeds and the daemon
  is left in the SDK's job. The lead's hypothesis holds exactly.
- libuv's job (Node's; `KILL_ON_JOB_CLOSE | BREAKAWAY_OK | SILENT_BREAKAWAY_OK | DIE_ON_UNHANDLED_EXCEPTION`,
  measured with Node v22.22.2) around the launcher's → the daemon leaves every job.

## Done
- **Fix** (`narration.platform._windows.WindowsPlatform.spawn_detached`, commit 3440e3d): the daemon is
  created with `CREATE_SUSPENDED` as well; Windows is asked whether it is in any job
  (`IsProcessInJob(child, NULL)`); only a daemon in none is resumed (psutil's `resume`, which is
  `NtResumeProcess`). One still in a job is ended before its first instruction and
  `NarrationError(DAEMON_UNAVAILABLE)` is raised (retryable, `retry_after_s` 60, the code's hint):
  `details.reason` is `left_in_job`; `breakaway_refused` stays for the access-denied case; `job_check_failed`
  when Windows cannot answer (also ended, never let run). The three reasons are module constants.
  `start_detached`, `DetachedLauncher` and `narration-admin daemon start` needed no code change: they
  already turn `DAEMON_UNAVAILABLE` into the retryable error with the hint, or the `--foreground` offer.
- `_JobObject` takes `silent_breakaway` and answers `contains(pid)`; `_in_job` and `_resume` are module
  helpers the tests reuse.
- **Docs**: docstrings of `spawn_detached` ("Nested jobs"), `narration.daemon.start`,
  `narration.backend.launch`, `narration.admin.daemon` (module and the operator's message),
  `narration.platform`; a CHANGELOG "Fixed" entry.
- **Spike k** saved: script, Node client, `results/before-fix.*`, `results/after-fix.*`, README with the
  findings labelled.

## Tests
- New, default suite, Windows only (skip cleanly elsewhere, as the whole module does):
  - `tests/platform/test_windows_processes.py`:
    - `test_a_daemon_that_would_stay_in_a_clients_job_is_refused_s4_1[silent|breakaway]`: a real
      `KILL_ON_JOB_CLOSE` job (the SDK's) around a nested job that allows breakaway (silently, like the
      launcher's; or explicitly). The child is started in the outer job from its first instruction
      (`start_in_job`: suspended, assigned, resumed), so the topology does not depend on `sys.executable`
      being a launcher. Expects `left_in_job`, no `spawned_pid`.
    - `test_a_daemon_leaves_nested_jobs_that_allow_breakaway_s4_1`: libuv's flags around a silent job; the
      daemon is in no job (`contains` False, `_in_job(pid, None)` False) and is alive after the job closes.
    - `test_a_job_says_which_processes_are_in_it_s4_1`, `test_a_silent_breakaway_job_keeps_the_members_children_out_s4_1`.
    - Fake-`Popen` tests: the suspended flags and the resume; `left_in_job` ends the child and never resumes;
      `job_check_failed` ends the child; a failed resume ends the child and raises `OSError`.
  - `tests/daemon/test_process.py::test_a_session_in_a_clients_job_around_a_launchers_starts_no_daemon_s4_1`:
    the real `start_detached` and daemon, in the lead's topology: `left_in_job`, no `run/daemon.json`, no
    `daemon.log`.
- **Mutation check**: with the `still_in_job` branch disabled, the two real-job regression tests and the
  session test fail with a `spawned_pid` inside the client's job (run here, file restored from git).
- `python -m pytest` (the default suite, before the rebase) → **3711 passed, 15 skipped**, 19 deselected
  (opt-in markers), in 7 min 40 s. The skips are other WPs' off-Windows, symlink and POSIX checks.
- After the rebase onto 9bb7e96 (no conflicts; main's five commits touch `narration.engine` and docs):
  `tests/platform`, `tests/daemon/test_process.py`, `tests/daemon/test_start.py`,
  `tests/admin/test_daemon_commands.py`, `tests/backend` → **367 passed, 13 skipped** (the same skips).
- `python -m ruff check`, `python -m ruff format --check`, `python -m basedpyright` → clean, on the rebased tree.
- `tools/check_private.py --commits main..HEAD --base main` → exit 0; `tools/check_tracked.py` (tree and
  `--commits main..HEAD`) → exit 0.
- Spike k ran three times in all (a smoke run and the two saved runs); every scenario gave the same answer each time.
- **Not run here**: Linux CI. BELIEVE it passes: the new tests are in Windows-only modules that skip at
  import; `_support.start_in_job` imports `_windows` lazily.

## Decisions made (and why)
- **Verify the outcome, not the chain.** There is no public API to enumerate a process's nested jobs, and
  the rule has an asymmetry (refusal only from the innermost). `IsProcessInJob(child, NULL)` on the
  suspended child is the authoritative answer, whatever the client's or a launcher's jobs are.
- **`CREATE_SUSPENDED`, not check-after-start.** A launcher that runs even briefly can start its interpreter,
  which could take the singleton and write status before being killed; suspended, nothing runs.
- **The launcher stays.** The daemon still runs as the venv's `pythonw.exe` (its launcher's own job holds only
  the interpreter and dies with the launcher, which waits for it). Starting the base interpreter would remove
  that job but add an undocumented `__PYVENV_LAUNCHER__` dependency, and the job was not the problem.
- **No route around a client's job.** A job that forbids breakaway can be escaped only by a process outside
  it (WMI, a scheduled task, a helper the operator runs): that is circumventing the client's policy, and the
  design's answer is `DAEMON_UNAVAILABLE` plus `narration-admin daemon start` in a terminal. Not added; your
  call if ever wanted.
- **Unknown membership is refused**, not resumed: "never leave a daemon silently inside a kill-on-close job".

## Contract change requests
- **Docstring only, `narration.contracts.interfaces.Platform.spawn_detached`**: "raises `DAEMON_UNAVAILABLE`
  when breakaway is refused" → "… when the process cannot be made to leave this process's Job Objects
  (breakaway refused, or the child left in an enclosing job and ended before it ran)". No signature change.
- **`codes.py`, `DAEMON_UNAVAILABLE`'s description** ("cannot be started detached (breakaway refused), or it
  is stopping"): consider "(breakaway refused or incomplete)". Wording only.

## Design change proposal (for plan.md §1.5)
- **§4.1, Detachment**: add after "If the client's Job Object forbids breakaway, `CreateProcess` fails with
  access denied": "Only when that is the front-end's innermost job. Since Windows 8 jobs nest, and the venv's
  `python.exe` launcher puts the interpreter in a nested job that allows silent breakaway; then `CreateProcess`
  succeeds and leaves the daemon in the client's job (KNOW, spike k). The front-end therefore creates the
  daemon suspended, lets it run only once `IsProcessInJob` says it is in no job, and otherwise ends it and
  returns `DAEMON_UNAVAILABLE` (`details.reason`: `left_in_job`)."

## Claude Code (the owner's client)
- **KNOW**: Claude Code 2.1.258 here is a Bun-compiled binary whose file carries `Bun.spawn`, `bun.sh`,
  `libuv` and `uv_spawn`; the shell it started for this work was in a job, and a venv launcher that shell
  started was in none (so that job breaks children's children away silently, as libuv's does). Under real
  Node v22 and under a libuv-flagged job, the daemon left every job and survived the client, before and after
  the fix.
- **BELIEVE**: Claude Code starts MCP servers through libuv's `uv_spawn` with libuv's global job, so the
  daemon survives it, before and after the fix. Not measured with Claude Code itself as the client; one
  manual check with the owner's client (as WP30's status already asked) would make it KNOW.
- The SDK-driven flow (the lead's driver, `narration-admin`'s own tests, any Python client) now gets
  `DAEMON_UNAVAILABLE` with the hint instead of a daemon that dies. For such a client the operator starts
  the daemon once from a terminal (`narration-admin daemon start`); it then serves every client.

## Dependency requests
- none

## Questions for the lead / owner
- Whether to record the `__PYVENV_LAUNCHER__` route (base interpreter) anywhere: not used, only measured.
- Aside, seen while measuring (not changed): `WorkerSupervisor` adds a worker to the daemon's kill-on-close
  group after the spawn, like the SDK; a worker launcher's interpreter born before the `add` is outside that
  group. It still dies with the daemon, because the launcher's own job is kill-on-close and the launcher
  is in the group. If you want the group to hold the interpreter too, `start_in_job`'s suspended start is
  the way.

## Next
- Review; then merge. Nothing else open.
