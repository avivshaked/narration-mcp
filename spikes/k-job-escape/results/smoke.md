# Spike (k), run `smoke`

Windows 10.0.26200, Python 3.12.10, mcp 2.2.0, node v22.22.2, run at 2026-09-27T19:41:10.

The orchestrator's own innermost job (the venv launcher's): ['KILL_ON_JOB_CLOSE', 'SILENT_BREAKAWAY_OK'].

## End to end

| Client | Server assigned | Server runs as | Server in a job (innermost flags) | Start | Daemon launcher in client's job | Daemon in client's job | Daemon alive after client | After |
|---|---|---|---|---|---|---|---|---|
| `none` | - | launcher | True ['KILL_ON_JOB_CLOSE', 'SILENT_BREAKAWAY_OK'] | started | False | True | True | stop → stopped (exited: True) |
| `none` | - | base | False | started | False | True | True | stop → stopped (exited: True) |

## The nesting rule (part 2)

| Outer job | Inner job (nested) | CREATE_BREAKAWAY_FROM_JOB | CreateProcess | Child in outer | Child in inner | Child in any job |
|---|---|---|---|---|---|---|
| KILL_ON_JOB_CLOSE | KILL_ON_JOB_CLOSE \| SILENT_BREAKAWAY_OK | True | ok | True | False | True |
| KILL_ON_JOB_CLOSE | KILL_ON_JOB_CLOSE \| SILENT_BREAKAWAY_OK | False | ok | True | False | True |
| KILL_ON_JOB_CLOSE | KILL_ON_JOB_CLOSE \| BREAKAWAY_OK | True | ok | True | False | True |
| KILL_ON_JOB_CLOSE | KILL_ON_JOB_CLOSE \| BREAKAWAY_OK | False | ok | True | True | True |
| KILL_ON_JOB_CLOSE \| BREAKAWAY_OK \| SILENT_BREAKAWAY_OK \| DIE_ON_UNHANDLED_EXCEPTION | KILL_ON_JOB_CLOSE \| SILENT_BREAKAWAY_OK | True | ok | False | False | False |
| KILL_ON_JOB_CLOSE \| BREAKAWAY_OK \| SILENT_BREAKAWAY_OK \| DIE_ON_UNHANDLED_EXCEPTION | KILL_ON_JOB_CLOSE \| SILENT_BREAKAWAY_OK | False | ok | False | False | False |
| KILL_ON_JOB_CLOSE \| BREAKAWAY_OK | KILL_ON_JOB_CLOSE \| SILENT_BREAKAWAY_OK | True | ok | False | False | False |
| KILL_ON_JOB_CLOSE \| BREAKAWAY_OK | KILL_ON_JOB_CLOSE \| SILENT_BREAKAWAY_OK | False | ok | True | False | True |
| KILL_ON_JOB_CLOSE | none | True | error 5 | None | None | None |
| KILL_ON_JOB_CLOSE \| SILENT_BREAKAWAY_OK | none | True | ok | False | None | False |
| KILL_ON_JOB_CLOSE \| SILENT_BREAKAWAY_OK | none | False | ok | False | None | False |
| KILL_ON_JOB_CLOSE \| BREAKAWAY_OK | none | True | ok | False | None | False |
| none | none | True | ok | None | None | False |
