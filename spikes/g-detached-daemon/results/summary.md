| Scenario | Session job (in job, innermost allows breakaway) | Start | Alive after session | Answers / renders after | Consoles in the tree | Windows | Stop |
|---|---|---|---|---|---|---|---|
| `plain` | [True, True] | started | True | True / True | console-of-worker-launcher | none | stopped (exited: True) |
| `uv-run` | [True, True] | started | True | True / True | console-of-worker-launcher | none | stopped (exited: True) |
| `job-breakaway-ok` | [True, True] | started | True | True / True | console-of-worker-launcher | none | stopped (exited: True) |
| `job-no-breakaway` | [True, False] | DAEMON_UNAVAILABLE |  |  |  |  |  |
| `control-no-breakaway-flag` | [True, True] | started | False |  |  |  |  |
| `console-python` | [True, True] | started | True | True / True | console-of-daemon, console-of-worker-launcher | none | stopped (exited: True) |
