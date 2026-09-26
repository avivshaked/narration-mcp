# Spike (g): a daemon started from an MCP session outlives that session

Plan: WP30. Design sections 4 and 4.1. Run on 2026-09-26 on Windows 11 (build 10.0.26200), Python 3.12.10,
psutil 7.2.2, from a shell started by a coding agent (itself an MCP-style client). No GPU, no model, no
network: the daemon runs the fake worker role (`--fake-workers`) and the test runner
`narration.daemon.testing:FakeWorkerRunner`.

## Question

An MCP client starts a stdio server as its child and ends it when the session closes. Some clients run it
in a Job Object that kills everything in it when the job closes. The front-end (WP36) starts the daemon
with `narration.daemon.start.start_detached`, which goes through `Platform.spawn_detached`
(`CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`). Does the daemon keep running,
and keep serving, once the session that started it is gone? What happens when the session's job forbids
breakaway? Does the daemon, or anything it starts, get a console?

## Files

| File | What it is |
|---|---|
| `spike_g.py` | The script. With no arguments it runs every scenario and writes `results/`; with `--session <json>` it is the session stand-in |
| `results/spike-g.json` | Every scenario's record: the session's Job Object facts, the start error if any, whether the daemon served before and after the session exited, whether the session proved the daemon it started was its own (`daemon_identity_proven`), a `release_gpu` round trip, a one-segment job rendered after the session exited, the daemon's process tree by role (names and `-m` arguments only, no paths), whether each process owns a window, and the `stop` result |
| `results/summary.md` | The same as a table |
| `status_read_race.py`, `results/status-read-race.json` | Part 2: what a reader of `run/daemon.json` gets while the daemon replaces it |

Reproduce (about 20 s):

```
uv run python spikes/g-detached-daemon/spike_g.py
```

It writes its stores under `<repo>/.dev/spike-g/`. Every process it measures or stops is one it can prove
it started (`tests/daemon/owned.py`). While the daemon runs, the session checks that the process at the pid in
`run/daemon.json` was created before the status it wrote, and that it is the child of the launcher the
session got back. It records both processes' exact creation times. The orchestrator then acts only on a pid
whose creation time is exactly the recorded one, through a `psutil.Process` object, and never by a bare
pid. `tests/daemon/test_process.py` pins the behaviours WP30 depends on, in the default suite on Windows.

## How a session is modelled

The session stand-in is a child process. It sets up its Job Object as the scenario says and starts the
daemon. It then waits until the daemon's `run/daemon.json` says `idle` with a pid, and exits at once with
`os._exit(0)`, which closes any job it made. The orchestrator then waits 1.5 s and checks the daemon from
outside. If it is alive, the orchestrator posts `release_gpu`, submits a one-segment job, records the
process tree, and posts `stop`.

| Scenario | The session | How it starts the daemon |
|---|---|---|
| `plain` | started as `python spike_g.py --session …` | `start_detached` |
| `uv-run` | started as `uv run --no-sync python spike_g.py --session …`, as a client configured with `uv run` would | `start_detached` |
| `job-breakaway-ok` | joins a kill-on-close Job Object that allows breakaway | `start_detached` |
| `job-no-breakaway` | joins a kill-on-close Job Object that forbids breakaway | `start_detached` |
| `control-no-breakaway-flag` | joins a kill-on-close Job Object that allows breakaway | `subprocess.Popen` with `DETACHED_PROCESS \| CREATE_NEW_PROCESS_GROUP`, without `CREATE_BREAKAWAY_FROM_JOB` |
| `console-python` | as `plain` | `spawn_detached` with the venv's `python.exe` in place of `pythonw.exe` |

## Findings

Labels as in AGENTS.md section 8. **KNOW** means shown by `spike_g.py`, with its output in `results/`.

### Survival

- **KNOW** In `plain`, `uv-run` and `job-breakaway-ok`, the daemon was alive 1.5 s after its session
  exited. It answered `release_gpu` (in 0.015 to 0.046 s) and rendered a job with its worker, both after the
  session was gone. `stop` then ended it: its last `run/daemon.json` says `stopped`, and the daemon and
  its launcher were both gone.
- **KNOW** In `control-no-breakaway-flag`, the daemon served (it wrote `idle` 0.32 s after the session
  began). It was dead 1.5 s after the session exited, and so was its launcher: the kill-on-close job
  took them with it. `running_daemon` then returned None for its leftover `run/daemon.json`, whose pid was
  gone. So `DETACHED_PROCESS` alone does not survive such a session; the breakaway does.
- **KNOW** In `job-no-breakaway`, `start_detached` raised `DAEMON_UNAVAILABLE` with
  `retry_after_s` 60 and details `{reason: "breakaway_refused", winerror: 5, in_job: true,
  job_allows_breakaway: false}`. It started nothing: no `run/daemon.json` was written. It did not fall
  back to a daemon that is not detached.
- **KNOW** Every session here was already inside a Job Object before it made its own. Its innermost job
  allowed breakaway: `session_job_before` is `[true, true]` in every scenario, `uv-run` included. That job
  is the venv launcher's (`Scripts\python.exe` is a launcher, and the interpreter runs as its child).
  Breakaway from the nested jobs worked whenever the job the session made allowed it.
- **BELIEVE** A real MCP client whose job forbids breakaway gets `DAEMON_UNAVAILABLE` as in
  `job-no-breakaway`. The user then starts the daemon from a terminal (`narration-admin daemon start`,
  WP37). This spike did not run under a real MCP client's own job.

### Consoles and windows

- **KNOW** The daemon started by `start_detached` runs as `pythonw.exe` (launcher and interpreter), and
  neither has a console: there is no `conhost.exe` child of either.
- **KNOW** Each worker's launcher (`python.exe`, started with `CREATE_NO_WINDOW`) has a `conhost.exe`
  child. So `CREATE_NO_WINDOW` still makes a console. Microsoft documents the flag as running a console
  program "without a console window".
- **KNOW** In `console-python`, the daemon's interpreter (`python.exe`, the launcher's child) got a
  `conhost.exe` of its own. A detached console launcher's child gets a new console.
- **KNOW** No process in any daemon's tree owned a window: PowerShell's `Get-Process … MainWindowHandle`
  was 0 for every one, `console-python` included.
- **BELIEVE** The console that `console-python` got can be shown as a window, which a user could close,
  ending the daemon. On Windows 11, a console may be hosted by a terminal application outside the
  process's own tree. This spike did not look at processes it did not start, so it did not measure that.
  Running the daemon as `pythonw.exe` avoids the question: it has no console to show or close. This is
  why `start_detached` asks the platform for the windowless interpreter (`python_for(..., console=False)`
  picks `pythonw.exe` on Windows), a deviation from design section 4.1's identity table, which names
  `python.exe`. The identity marker, `-m narration.daemon --store <root>`, is unchanged; the daemon and
  its workers are started with `-P` before it (the tree in `results/spike-g.json` shows it).

### Reading `run/daemon.json` while the daemon replaces it (`status_read_race.py`)

The front-end, `narration-admin` and a starting daemon read `run/daemon.json` while a daemon may be
renaming a new one over it. `status_read_race.py` measures that for 8 s at a time
(`results/status-read-race.json`; `uv run python spikes/g-detached-daemon/status_read_race.py`, about 30 s).

- **KNOW** `NarrationStore.get_daemon_status` in one process, while another process keeps writing the
  status with `put_daemon_status`, raised `PermissionError` on 417 of 10,983 reads (3.8 %). No other
  error was seen.
- **KNOW** `os.path.realpath` of a file that another thread renames away returned a `\\?\`-prefixed path
  (so not under its folder, as the store's path check compares) for 6,716 of 25,071 calls. That is the
  mechanism, at its worst: CPython's `ntpath.realpath` keeps the prefix when the file is gone by its
  second look. `tests/daemon/test_process.py` once saw it on `run/daemon.json` itself: `get_daemon_status`
  raised `StorePathError` ("… resolves to \\?\…, outside the store root"). That was one run in about ten,
  before WP30 read the status through `read_status`.
- **KNOW** Through `narration.daemon.sweep.read_status`, which reads again after 20 ms, ten reads at most,
  5,440 reads under the same writer raised nothing.
- Since WP12's follow-ups, the store itself reads again while Windows refuses the file
  (`files.read_retrying`), and its path check strips the `\\?\` prefix (`platform.real_path`).
  `read_status` now only turns a torn file into `StatusUnreadable`. The numbers above were measured
  before that change, with the retry that `read_status` had then.

### What was not tested

- **ASSUME** A logoff or a reboot ends the daemon like any other program. The next daemon's orphan
  sweep then re-queues its jobs (`tests/daemon/test_process.py` kills a daemon hard and checks that).
- A real MCP client's session (Claude Desktop, an IDE) was not used; the scenarios model its process
  set-up. Such a client's job settings are not known here.
