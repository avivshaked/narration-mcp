# Operator guide

This is for whoever runs `narration-mcp` on their own machine: the commands under `narration-admin`, what
they check and change, and what to do about the error codes an operator (rather than a calling agent) is
the one who can fix. It assumes you have already followed the [README](../README.md)'s Install section.
Source of truth: [docs/design.md](design.md) revision 5.16 (sections 4, 4.1, 7.1, 14, 15, 16, 17) and the
code; where they differ, this guide follows the code, and says so.

Every command below takes `--config <path>`; where it is left out, it is found the same way
`narration-mcp` finds it (`--config`, else the file `NARRATION_CONFIG` names, else `narration.toml` in the
service's own folder). All of them are run through the environment's own interpreter,
`<venv_python> -m narration.admin`, never through the `narration-admin.exe` launcher some antivirus
software sandboxes (the README's [Wire it into Claude Code](../README.md#wire-it-into-claude-code)
explains why).

## The daemon

The daemon is the process that actually loads models and does the work; the MCP server (`narration-mcp`)
is a thin, stateless front-end per client that starts it on the first job and asks it to do things.

- **`narration-admin daemon start [--wait S] [--foreground]`** starts it detached, unless one already
  serves this store, and by default waits (30 s) until it says it serves. It exits after
  `[daemon] idle_exit_min` minutes without work (0 means it exits as soon as it is idle).
- **`narration-admin daemon stop [--now] [--wait S]`** asks the running daemon to stop. Without `--now` it
  finishes the segment in flight, unloads and exits; `--now` ends the work in flight at once and puts it
  back on the queue. Nothing is posted when no daemon runs. Queued jobs survive either way and resume on
  the next start.
- **`narration-admin daemon status [--json]`** prints `run/daemon.json` (state, pid, the current job, the
  GPU, the workers) and whether that daemon still runs. Once the daemon has stopped, it says why
  (`stop_reason`): `operator` (a `daemon stop`, or `install`'s stop), `idle` (it had no work for
  `[daemon] idle_exit_min`), `interrupted` (Ctrl+C under `--foreground`) or `error` (its log says what
  failed).

**The Job Object rule (Windows).** The daemon is started detached
(`CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`) so that it outlives the client
that launched it. Windows groups processes into nested Job Objects, and **the daemon only ever runs in no
Job Object at all**: if the client's own job forbids breakaway, Windows either refuses to create the
process or would leave it inside that job — where it would die with the client mid-job — so the front-end
checks this and refuses to start such a daemon at all, rather than start one that is not really detached.
The job stays queued, and the call answers `DAEMON_UNAVAILABLE` (see [Error codes an operator
sees](#error-codes-an-operator-sees) below).

This is common on a CI runner, or a program with its own built-in terminal. Two ways out:

- run `narration-admin daemon start` from a plain terminal whose Job Objects allow breakaway (not a
  program's built-in one), or run the daemon in the current one with **`--foreground`**, which keeps it
  attached to that terminal (Ctrl+C stops it at once, and the job in flight goes back to the queue) —
  the terminal must then stay open while it works;
- on a host whose jobs always forbid breakaway (a CI runner, say), set **`[daemon] autostart = false`** in
  `narration.toml`, so no client ever tries to start a daemon for you, and start one yourself with
  `narration-admin daemon start --foreground`, in a process of its own, before any job is submitted.

The daemon and `narration-mcp` both read `narration.toml` only when they start. After any change other
than `[measurement]` (never edit that while the store holds measurements made under the old values), stop
the daemon (`daemon stop`; the next job starts it again if `[daemon] autostart` is on, otherwise
`daemon start` does) and reconnect your MCP client (`/mcp` in Claude Code).

## `doctor`

`narration-admin doctor [--quick] [--json]` is the first thing to run after anything changes: it checks
the platform, the configuration, the store, the pinned models (hashed again against what `install`
recorded; `--quick` skips the hashing and only checks the files are there), each worker's venv, the GPU
(through NVML — this starts no CUDA context), and whether an engine profile is pinned. Each check prints
`ok`, `info`, `warn` or `fail`, and every `warn` or `fail` says what to run next. Exit code is 1 if
anything failed.

A `warn` on the `qa` check ("every take is scored with QA profile …") means `[qa] profile` in
`narration.toml` names a different profile than this build actually scores with — the setting is
informational only (it changes no behaviour), so this is never a failure, only a reminder that your
configuration and your build have drifted apart in what they say.

## `install`

`narration-admin install [--dry-run] [--from-cache <hub cache>] [--models-only | --workers-only]`
downloads every pinned model's files into `[server] models_root`, each one verified against the hash
Hugging Face publishes for that revision before it is kept, and syncs each worker's uv project
(`workers/qwen3tts`, `workers/qa`) from its own committed `uv.lock`. Run it again any time to repair a
missing or corrupted file — nothing half-written is ever left in a model's snapshot folder, and a file
already there that already matches is left alone. If a running daemon holds a worker or model that
`install` repaired, it is asked to stop after its segment in flight, since a daemon keeps a worker it once
found broken marked broken for its whole lifetime. Jobs queued before that stop wait for the next start
(`narration-admin daemon start`, or the next `submit_job` while `[daemon] autostart` is on), and that
daemon runs the whole queue.

TLS: downloads are verified against the system's certificate store (plus `SSL_CERT_FILE` /
`REQUESTS_CA_BUNDLE` if set). If your network intercepts TLS, set those, or `UV_NATIVE_TLS=1` for uv's own
downloads; verification is never turned off — if it still fails after that, something is genuinely wrong
with the certificate and it should not be bypassed.

## `engine pin` / `repin` / `bridge` / `show`

The **engine profile** is what "the same voice sounds the same" depends on: the exact model revision, the
worker's package versions, the CUDA/driver stack, and the generation settings that are pinned explicitly
rather than left to library defaults (design section 10.1). Two profiles are pinned, one for the Base
model (cloning) and one for VoiceDesign.

- **`engine pin [--json]`** records the profile this installation and this machine currently match. It
  also **designs the canary**: a short clip rendered from a fixed description, text and seed, whose hash
  and speaker embedding are kept with the profile. A later render is checked against the canary to catch a
  driver, CUDA or library change that would make output subtly different without anyone editing a version
  number — a raw hash match means bit-identical output; a canary similarity match, on a stack whose
  numerics only reproduce closely, still confirms "close enough".
- **`engine repin [--force]`** makes a new profile the one in use wherever the installation or the
  machine's GPU/driver/CUDA/cuDNN has changed since the last pin (or, with `--force`, unconditionally for
  both engines). A new profile means new render keys: every voice must be measured again before it can be
  used for narration (`measure_voice`).
- **`engine bridge <old> <new>`** compares how the canary and the calibration corpus sound under two
  profiles, so you can judge whether a repin actually changed anything a listener would notice.
- **`engine show [--json]`** lists the profiles pinned, which is currently in use for which engine, its
  determinism tier and its canary threshold.

`pin`, `repin` and `bridge` load Qwen on the GPU, so run them only while no daemon is using the store
(they take the store's singleton lock themselves and refuse if a daemon holds it), and expect enough free
VRAM (see the README's Requirements).

## Voices: `voices allow` / `voices list`

**The service clones a clip only if it is synthetic** (design section 17.4): either the service designed
it itself (it is on the service's own provenance list, kept automatically), or its sha256 is in
`[voices] allow_sha256`. A recording of a real person must never be cloned, whoever asks.

- **`narration-admin voices allow <clip.wav> [--yes]`** reads the clip (a local WAV, at most 20 MB),
  prints its path, length and sha256, and asks you — a person, at a terminal — to type `yes` to confirm it
  is synthetic. Only then is its hash added to `[voices] allow_sha256` in your configuration file, with a
  comment naming the clip; every other line and comment is kept exactly. `--yes` skips the question and is
  for your own scripts only, once you already know the clip is synthetic — never for a calling agent, and
  there is deliberately no MCP tool for this: only an operator, at this machine, decides what may be
  cloned.
- **`narration-admin voices list [--json]`** prints the list as it stands.

Because `narration-mcp` and the daemon read the allowlist only when they start, allowing a clip changes
nothing until both are restarted: stop the daemon first (`daemon stop`), then reconnect every MCP client
that runs the front-end.

## `gc` and `verify`

- **`narration-admin gc [--apply] [--json]`** removes cached renders, takes, analyses, voice profiles,
  designs and finished jobs unused for `[retention] retention_days`, and measurements unused for
  `measurement_retention_days`; a dry run by default (`--apply` actually removes them). The provenance
  list, engine profiles and canaries, and alignment benchmarks are never collected. It is safe to run
  while the daemon works. It also reports, apart, how many failed or replaced takes are in the store, how
  old the oldest one is, and how many this run would take out of `narration-admin failures`' view — so you
  can export what you need before it is gone.
- **`narration-admin verify [--json]`** re-hashes every immutable file in the store against its index, runs
  SQLite's own integrity check, and re-hashes every installed model file against `install`'s record.
  Changes nothing; exit code 1 if anything is missing, changed or unreadable. A damaged model file is
  repaired by `install`; a damaged store file is not repaired automatically — keep the report.

## `failures`: auditing what went wrong

**`narration-admin failures [--since DATE] [--job ID] [--voice SHA256] [--code FLAG] [--json] [--export
DIR]`** lists every take of a narration job that either failed QA or was replaced by an automatic retake,
across every job in the store, newest first: the job, the segment, the attempt and seed, the take's WAV
path, every fail and warn flag, the key QA metrics (adjusted WER, speaker similarity, pace) against their
thresholds, which attempt finally filled the slot, and the segment's text as it was sent. It changes
nothing in the store (not even a "last used" time), so listing a take never keeps it from `gc`.

`--export <dir>` copies each listed take's WAV to `<dir>/<take_id>.wav` with a JSON sidecar of its
reasons, plus `index.csv` (safe to open in a spreadsheet: a cell that looks like a formula is escaped).
The export folder must be outside the store. Run this before `gc` removes what you want to keep a record
of — `gc`'s own output tells you how many failed takes it is about to take out of this view.

## `render`: a quick end-to-end test from the terminal

**`narration-admin render --voice <clip.wav> --transcript <text> --text <text> [--takes N] [--out
<file.wav>] [--dry-run] [--json]`** sends one `submit_job` request with a single segment, waits for it,
and prints the result — a thin client of the service's own backend, so it is subject to every check a
caller would face (a measured, synthetic voice; the text and size limits). `--dry-run` shows the plan
without rendering anything; `--out` copies the suggested take's delivery WAV to a file. Running the same
command again returns the same (possibly still-running) job, so it is safe to re-run after an interrupted
wait.

## Logs

| What | Where |
|---|---|
| The MCP front-end (one log per running `narration-mcp`) | `<store_root>\logs\narration-mcp.log` |
| The daemon (rotated) | `<store_root>\logs\daemon.log` |
| The daemon's live status | `<store_root>\run\daemon.json` (also `narration-admin daemon status`) |
| The last detached launch the platform let run | `<store_root>\run\launch.json` |

An `INTERNAL` tool error's `details.log` names the front-end's own log file for that session.

## Error codes an operator sees

Most tool errors are for the calling agent to act on (see [docs/tools.md](tools.md)'s Error codes table
for the full list and what each one's own hint says). These are the ones that usually need something from
*you*, the operator, rather than a change to the request:

- **`DAEMON_UNAVAILABLE`** — retryable. Either the daemon could not be started detached (see [the Job
  Object rule](#the-daemon) above: use `--foreground` or a plain terminal), or it is stopping, or the last
  launch failed or exited within its 90-second start window (`details.log` names the daemon's log; run
  `narration-admin daemon start --foreground` in a terminal to see why it exits, or check
  `<store_root>\logs\daemon.log`). Once you have started a daemon successfully (`daemon status` shows one
  running), the same request works.
- **`BACKEND_NOT_INSTALLED`** — model weights or a worker's environment are missing. Run
  `narration-admin install`, then `narration-admin doctor` to confirm.
- **`ENGINE_DRIFT`** — a worker's fingerprint no longer matches the pinned engine profile, or the canary's
  similarity fell below its threshold: something about the installed stack (packages, CUDA, driver)
  changed without a re-pin. Either restore the pinned environment, or run `narration-admin engine repin`
  to pin a new profile for what is installed now — which means every voice must be measured again
  (`measure_voice`) before narration can use it.
- **`ENGINE_CHANGED`** — a request's own `expect_engine_profile` no longer matches the service's current
  one (usually because you repinned). This is the caller's to resolve (accept the new hash, or ask you to
  restore the old one) rather than yours, unless the repin itself was unintended.
- **`VOICE_NOT_MEASURED`** — no measurement exists for this exact voice (clip and transcript) under the
  current engine profile. There is nothing to fix on your side beyond making sure a GPU is free: the
  caller runs `measure_voice`, a 20-to-50-minute job. This is also what you will see for *every* voice
  right after an `engine repin`, since a new profile has no measurements of its own yet.
- **`VOICE_NOT_SYNTHETIC`** — the clip is neither one the service designed nor one you have allowlisted.
  See [Voices](#voices-voices-allow--voices-list) above: confirm the clip is synthetic, then
  `narration-admin voices allow <clip.wav>`, then restart the daemon and reconnect the client.
- **A job stuck at phase `waiting_for_gpu`** (not an error code — a `get_job` phase) means the job is
  queued behind free VRAM: another job, or another program on the same GPU, is using it. Check
  `get_server_status`'s `admission.gpu` (or `narration-admin daemon status`) for what is holding it and
  how much is free. The MCP tool `release_gpu` (there is no admin-command equivalent) unloads an *idle*
  model at once, so a caller about to do other GPU work need not wait for the idle timeout; it changes
  nothing while a job is actually running.

## Re-measuring a voice after a pace-method or corpus change

A voice's measurement is keyed on, among other things, the calibration corpus's version and the pace
method used to score it (`narration.contracts.names.PACE_METHOD`, `CORPUS`). **A change to either bumps
the measurement's own schema version**, so every measurement made under the old one becomes invisible —
not silently wrong: a request against an old measurement reads exactly as if the voice had never been
measured (`VOICE_NOT_MEASURED`). There is nothing to migrate or clear by hand: the next `measure_voice`
call for that voice simply measures it again, under the current corpus and pace method, and the old
measurement stays in the store until `[retention] measurement_retention_days` and `gc` remove it (or you
can leave it — it costs disk, not correctness). This is also what happens automatically after an `engine
repin`, since the engine profile is part of the same key.
