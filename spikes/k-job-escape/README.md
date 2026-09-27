# Spike (k): does the detached daemon leave every Job Object of its client?

Plan: WP30 (follow-up `wp/30-escape`). Design section 4.1. Run on 2026-09-27 on Windows 11 (build
10.0.26200), Python 3.12.10, mcp 2.2.0, psutil 7.2.2, Node.js v22.22.2. No GPU, no model, no network: the
real daemon runs with the `NullRunner` and idles.

## Question

The lead saw a detached daemon die when its MCP client exited, without logging a stop, although section 4.1
requires the daemon to outlive its client (Phase 4: "the client quits mid-job and the job completes"). The
client was the MCP Python SDK's `stdio_client`, which puts the server in a kill-on-close Job Object without
`BREAKAWAY_OK`; the server was the venv's `python.exe`, a launcher whose child is the interpreter.
`start_detached` reported success (the launcher's pid), the daemon served, and it vanished with the client.
Spike (g) had found that a daemon survives a session whose job allows breakaway; it had not run under a job
that forbids it while the venv launcher nested a job inside.

Hypothesis (the lead's, BELIEVE at the time): the launcher puts its interpreter in a job of its own, nested
inside the client's, and a breakaway from the innermost job leaves the daemon inside the outer one.

## Files

| File | What it is |
|---|---|
| `job_escape.py` | The script. With no arguments (`--label <name>`) it runs every scenario and writes `results/<name>.json` and `results/<name>.md`; `--role server \| client \| sdk-client \| nest` are its stand-ins |
| `node_client.js` | A Node.js client stand-in (`child_process.spawn`, which is libuv's `uv_spawn`) |
| `results/before-fix.*` | Every scenario on the code as merged (main at 7a717dd) |
| `results/after-fix.*` | The same scenarios with the fix in `narration.platform._windows.spawn_detached` |

Reproduce (about 40 s; the stores go under `<repo>/.dev/spike-k/`):

```
uv run python spikes/k-job-escape/job_escape.py --label <name>
```

Every process the script measures or ends is one it started, proven by its creation time before anything acts
on it (`tests/daemon/owned.py`). The results hold no paths.

## How it is measured

**Part 1, end to end.** A *client* starts a *server* stand-in the way a real client would; the server reports
its own job facts (`IsProcessInJob`, and the innermost job's limit flags and member pids from
`QueryInformationJobObject`), then calls the real `narration.daemon.start.start_detached` for the real daemon,
waits until `run/daemon.json` says `idle`, and proves the daemon its own (`capture_daemon`). While the
client's job still exists, the client asks Windows whether the daemon's launcher and the daemon are in it
(`IsProcessInJob` with the job's handle; for the SDK, the handle the SDK itself holds). Then the client
ends, and the orchestrator checks 1.5 s later whether the daemon is alive, and stops it through the store.

| Client | How the server is put in the job | Job flags |
|---|---|---|
| `sdk` | `mcp.client.stdio.stdio_client`, mcp 2.2.0: assigned after the spawn | `KILL_ON_JOB_CLOSE` (`mcp/os/win32/utilities.py`) |
| `job:sdk-like` | a hand-made job; assigned after the spawn (`post`), or from the first instruction (`suspended`: created suspended, assigned, resumed) | `KILL_ON_JOB_CLOSE` |
| `job:libuv-like` | a hand-made job, `suspended` | `KILL_ON_JOB_CLOSE \| BREAKAWAY_OK \| SILENT_BREAKAWAY_OK \| DIE_ON_UNHANDLED_EXCEPTION` |
| `node` | `child_process.spawn` in Node.js v22.22.2 (libuv's `uv_spawn`) | whatever libuv sets: measured, see below |
| `none` | `subprocess.Popen`, no job | |

Each client runs the server as the venv's `python.exe` (`launcher`: the interpreter is the launcher's child)
or as the base interpreter (`base`: `sys._base_executable`, told about the venv with `__PYVENV_LAUNCHER__`,
which is what the launcher itself sets for its child; no launcher, so no extra job).

**Part 2, the rule.** One process (the base interpreter) puts itself in an outer job, then in an inner job
nested in it, starts a suspended child with or without `CREATE_BREAKAWAY_FROM_JOB`, and asks Windows which
jobs the child is in. 13 cases.

## Findings

Labels as in AGENTS.md section 8. **KNOW** means shown by `job_escape.py`, with its output in `results/`.

### The rule (part 2; `results/*.json`, `nested_jobs`; identical in both runs)

- **KNOW** `CreateProcess` with `CREATE_BREAKAWAY_FROM_JOB` fails with access denied (winerror 5) only
  when the caller's **innermost** job forbids breakaway (a single job with `KILL_ON_JOB_CLOSE` alone).
- **KNOW** When the innermost job allows breakaway (`BREAKAWAY_OK`, or `SILENT_BREAKAWAY_OK`) and an
  enclosing job does not, the call **succeeds** and the child is **in the enclosing job**: outer
  `KILL_ON_JOB_CLOSE`, inner `KILL_ON_JOB_CLOSE | SILENT_BREAKAWAY_OK` → child in outer, not in inner;
  the same with inner `KILL_ON_JOB_CLOSE | BREAKAWAY_OK`. Nothing tells the caller.
- **KNOW** The child leaves every job in the chain only when every job allows breakaway: outer
  `BREAKAWAY_OK`, inner `SILENT_BREAKAWAY_OK` → in no job. libuv's flags around the launcher's → in no job,
  with or without `CREATE_BREAKAWAY_FROM_JOB`.
- **KNOW** `SILENT_BREAKAWAY_OK` alone takes a member's children out of that job (and no further): outer
  `KILL_ON_JOB_CLOSE | BREAKAWAY_OK`, inner `KILL_ON_JOB_CLOSE | SILENT_BREAKAWAY_OK`, no flag → child in
  the outer only.
- **KNOW** The venv launcher's own job has `KILL_ON_JOB_CLOSE | SILENT_BREAKAWAY_OK` and holds only the
  interpreter: the orchestrator, a launcher's child, reports exactly that as its innermost job, with its own
  pid as the only member. So the interpreter's children are never in the launcher's job, and the launcher
  is in whatever job its parent put it in.

So the rule Windows applies, as measured: it walks the caller's jobs from the innermost outward, the child
leaves each job that allows breakaway, and it is placed in the first job that does not, together with that
job's ancestors; only when that first job is the innermost does the call fail. The hypothesis holds.

### End to end, before the fix (`results/before-fix.md`)

- **KNOW, the lead's case reproduced.** `sdk` + `launcher`: the SDK made a job; the server's launcher and
  the server (whose innermost job is the launcher's) are in it; `start_detached` returned a pid; the daemon
  served; and, asked with the SDK's own job handle, **the daemon's launcher and the daemon were both in the
  client's job**. 1.5 s after the client exited, `run/daemon.json` still said `idle` (no stop) and
  `running_daemon` found no process. The same with the hand-made `KILL_ON_JOB_CLOSE` job, assigned after the
  spawn or from the first instruction.
- **KNOW** `sdk` + `base`: the server runs directly in the SDK's job (innermost `KILL_ON_JOB_CLOSE`), so
  `CreateProcess` refused, `start_detached` raised `DAEMON_UNAVAILABLE` (`breakaway_refused`, winerror 5),
  and no `run/daemon.json` was written. This is what section 4.1 describes; the launcher is what changes it.
- **KNOW** `job:libuv-like` (both interpreters), `node` (both) and `none` (both): the daemon's launcher and
  the daemon were in no job of the client's (for `node`, the daemon's own launcher was in no job at all),
  the daemon was alive after the client, answered `stop`, and its final status was `stopped`.
- **KNOW** Under real Node.js v22.22.2, a child's innermost job has libuv's flags
  `KILL_ON_JOB_CLOSE | BREAKAWAY_OK | SILENT_BREAKAWAY_OK | DIE_ON_UNHANDLED_EXCEPTION`, and Node's own pid
  is a member too. That matches libuv's `uv__init_global_job_handle` (`src/win/process.c`) as read; the
  flags were verified here by measurement, not only from the source.

### End to end, after the fix (`results/after-fix.md`)

The fix (`WindowsPlatform.spawn_detached`): the daemon is created with `CREATE_SUSPENDED` as well, Windows is
asked whether it is in any job (`IsProcessInJob(child, NULL)`), and only a daemon in none is resumed. One
still in a job is ended before its first instruction and `DAEMON_UNAVAILABLE` is raised with
`details.reason = left_in_job`; one whose membership cannot be read, the same with `job_check_failed`.

- **KNOW** `sdk` + `launcher` and both `job:sdk-like` + `launcher` cases: `DAEMON_UNAVAILABLE`
  (`left_in_job`), `retry_after_s` 60, and no `run/daemon.json`: nothing ran. `sdk` + `base` and
  `job:sdk-like` + `base`: `DAEMON_UNAVAILABLE` (`breakaway_refused`), as before.
- **KNOW** `job:libuv-like`, `node` and `none`, both interpreters: unchanged; the daemon is in no job of
  the client's, survives it, and `stop` ends it (`stopped`).
- The regression is pinned in the default suite on Windows: `tests/platform/test_windows_processes.py`
  (`…would_stay_in_a_clients_job_is_refused…`, both inner-job kinds; `…leaves_nested_jobs_that_allow_breakaway…`)
  and `tests/daemon/test_process.py` (`…in_a_clients_job_around_a_launchers_starts_no_daemon…`, the real
  daemon). With the check disabled, all three fail with a `spawned_pid` inside the client's job.

### Claude Code

- **KNOW** This machine's Claude Code (2.1.258) is a Bun-compiled binary; its file carries the strings
  `Bun.spawn`, `bun.sh`, `libuv` and `uv_spawn`. The shell Claude Code started for this work was in a Job
  Object, and a venv launcher that shell started was in none: so that job takes children's children out of it
  silently, as libuv's does.
- **BELIEVE** Claude Code starts its MCP servers through libuv's `uv_spawn` with libuv's global job, like
  Node does. The daemon then leaves every job and survives the client, before and after the fix
  (`job:libuv-like` and `node` above are that topology). Not measured with Claude Code itself as the client.

### What was not tested

- A real MCP session with the real front-end under the SDK (the lead's driver): the server stand-in calls the
  same `start_detached` the front-end calls, but no MCP messages were exchanged.
- The SDK assigns the server to its job **after** the spawn. A launcher that started its interpreter before
  the assignment would leave the interpreter, and so the daemon, outside the job (the SDK's own comment
  notes this). In three runs here the assignment always came first; the tests use the suspended start, which
  has no such race.
- Whether Windows Terminal, VS Code's terminal or other hosts put a terminal in a job that forbids
  breakaway. `narration-admin daemon start` reports `DAEMON_UNAVAILABLE` in that case either way, and
  offers `--foreground`.
