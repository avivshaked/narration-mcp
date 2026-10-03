# A view of what the service is doing: the plan (WP46)

*Status: **a proposal for the owner's approval.** Nothing in it is built, and nothing will be until the owner
approves it (plan.md, WP46). Written 2026-09-28 against design revision 5.17 and `main` at 7afa6ee.*

This document plans a read-only view of a running narration-mcp installation: its jobs, the takes each job
made, the failed takes, the measured voices, the allowlist, the engine pins and the canary, and the state of
the daemon and the GPU. It weighs the two ways plan.md names for building it, recommends one, maps every part
of the view to where its data lives today, and splits the work into packages.

It is written for someone who has not read the rest of the repository. The terms it uses are the design's:
a **job** is one request the daemon works through; a **segment** is a paragraph of cues; a **take** is one
rendered, post-processed delivery of a segment, with a **verdict** (`pass`, `warn` or `fail`) and **flags**
from QA; the **store** is the folder (`<store_root>`) where the service caches all of that
([design.md §15](design.md)).

**Evidence labels** (AGENTS.md §8): **KNOW** is measured or read in the code here, with the source named;
**BELIEVE** is expected and still to be verified; **ASSUME** is a placeholder, such as an estimate.

---

## 1. The decision in short

- **Recommended: a static HTML report written by `narration-admin`** (option A), with a `--watch` mode that
  rewrites a small status page every few seconds while jobs run. It needs no network socket, no new
  dependency and no change to the design's "no process listens on a socket" rule (§0 item 11, §17.1).
- **A local dashboard on 127.0.0.1** (option B) gives a live view with less friction, but it opens a socket,
  which every local user and every web page in the browser can try to reach, and it needs a design change,
  several security controls that each have to be right, and about twice the work. It stays a possible
  phase 2, if the owner finds the report's freshness too slow in practice.
- **Both options need the same first step:** a store reader that opens the database read-only. The store's
  own `NarrationStore` writes when it opens and repairs its index when it reads (KNOW, §4), so no view may
  use it as it is.
- **Nine questions for the owner** are in §10. Question 1 (static report or dashboard) is the one the rest
  depends on.

---

## 2. Constraints

From plan.md's WP46 row, with the design rules they rest on:

| # | Constraint | Where it comes from | What it means for the view |
|---|---|---|---|
| C1 | **Read-only: it approves nothing** | §0.2 (stateless), §7.1 | No button or endpoint changes anything: no approve, choose, retake, cancel, `release_gpu`, `gc`, or `voices allow`. It does not even record that a take was listened to. It changes no row, file or last-use time in the store, so viewing a take never keeps it from `gc` (the rule `narration-admin failures` already follows, §7.1). |
| C2 | **Local only** | §0 item 14, §17.1, §17.12 | Nothing leaves the machine: no CDN, no web font, no analytics, no update check. No port reachable from another machine. |
| C3 | **Every dependency licence-checked** | plan.md §1.4, §18, AGENTS.md rule 9 | Permissive licences only; no GPL, AGPL, non-commercial or research-only code, in tests either. |
| C4 | **Never competes with a running job for the GPU** | plan.md WP46, §4 | The view loads no model, imports no torch, starts no CUDA context and talks to no worker. It reads GPU facts through NVML only, as `doctor` does. |

Four more rules follow from how the service works today:

| # | Rule | Why |
|---|---|---|
| C5 | **Never hold the database's write lock, and keep every read transaction short** | The daemon writes with `BEGIN IMMEDIATE` and a 30 s busy timeout (`store/db.py`). A view that takes that lock delays the daemon. A read transaction left open stops SQLite from resetting its write-ahead log, which then grows (KNOW, appendix A, step 7). |
| C6 | **The caller's text stays on this machine and out of logs** | A job's request, kept by value in `JobRecord.request`, holds the caller's text. `failures` prints and exports it, but never logs it (§7.1, §17.10). The view does the same. |
| C7 | **A take is reached by its id, never by a path the browser sends** | §17.2's write confinement has a read-side twin here: the view serves or links only files whose path it read from the store's own index, confined under `<store_root>`. |
| C8 | **The view is an operator's tool, never an MCP tool** | Like `gc`, `verify` and `failures`: it shows this machine's cache to the person who runs it (§7.1). |

---

## 3. What the view shows, and where each piece lives

Paths are relative to the repository; `<store_root>` is the store. The store's tables are defined in
`src/narration/store/db.py` (`_V1`), and its files in [design.md §15](design.md). Every record type named
here is a dataclass in `src/narration/contracts/models.py`.

### 3.1 Jobs and their progress

| What is shown | Where it lives | Read with |
|---|---|---|
| Every job: id, kind, label, status, priority, created and updated times | `jobs` table: `job_id`, `kind`, `status`, `priority_rank`, `inserted_at`, `updated_at`, and the whole `JobRecord` as JSON in `record`; also `jobs/<job_id>/job.json` | `NarrationStore.iter_jobs` (a plain SELECT, newest first; `store/store.py`) |
| The queue, in the order it will run | the `queue` view (`status IN ('queued','running','cancelling')`, by priority then FIFO) | `NarrationStore.queued_jobs` |
| Progress: seconds of audio done of the total, segments done of the total | `JobRecord.progress` (`Progress`: `done_s`, `total_s`, `fraction`, `segments_done`, `segments_total`) | the job's `record` |
| Phase (`waiting_for_gpu`, `loading_model`, `canary`, `rendering`, …) and retake round | `JobRecord.phase`, `JobRecord.round`; the running job also appears in `run/daemon.json`'s `current_job` | the job's `record`; `NarrationStore.get_daemon_status` |
| Outcome, error and message | `JobRecord.outcome` (`all_passed`, `needs_attention`), `JobRecord.error` (an `Error` with `code`, `message`, `hint`), `JobRecord.message` | the job's `record` |
| The queue's estimated drain time | `DaemonStatus.est_drain_s` in `run/daemon.json` | `get_daemon_status` |
| Pending operator commands (a `stop` not yet honoured, say) | `commands` table, rows with `done_at IS NULL` | `NarrationStore.pending_commands` |

**Freshness (KNOW, `jobs/record.py`):** the job's `record` is rewritten by the daemon as each segment moves
on, and at the end with its items, outcome and `result`. `run/daemon.json` is rewritten on every phase
change (`DaemonStatus`'s docstring).

### 3.2 Each segment's takes: verdicts, flags and a player

A take is reached from its job, never by browsing the store, because only the job says which segment and
attempt a take was, and what text it was meant to speak.

| What is shown | Where it lives | Read with |
|---|---|---|
| A segment's state, takes passed, retakes used, flags | `JobRecord.items[]` (`JobSegment`: `segment_id`, `state`, `takes_ok`, `retakes_used`, `flags`) | the job's `record` |
| Each attempt: attempt number, seed, round, take, analysis, verdict, flags, whether this job made it | `JobSegment.attempts[]` (`JobAttempt`: `attempt`, `seed`, `round`, `render_id`, `take_id`, `analysis_id`, `fresh`, `verdict`, `flags`) | the job's `record` |
| The segment's text, per cue, as the request sent it | `JobRecord.request`, kept by value | `narration.backend.requests.segments_of` (what `failures` uses) |
| The suggested take and why | `JobRecord.result["suggestions"]` (set when the job completes, `jobs/record.py` `result`) | the job's `record` |
| The consistency report across the job's suggested takes | `JobRecord.result["consistency"]` | the job's `record` |
| **The audio player**: the delivery WAV, its length, loudness and trim | `takes` table (`record` = `TakeRecord`: `delivery.path`, `delivery.duration_s`, `loudness`, `trim`, `flags`); the file is `takes/<ab>/tk_<16hex>/delivery.wav` | `NarrationStore.peek_take` (returns the record and whether the file is still there, and never repairs the index) |
| QA: verdict, every flag with severity and message, WER, speaker similarity, pace, the thresholds used, the transcript heard | `analyses` table (`record` = `AnalysisRecord`: `qa.verdict`, `qa.flags`, `qa.metrics`, `qa.thresholds`, `qa.transcript`); also `takes/<ab>/tk_…/analyses/an_<16hex>.json` | `NarrationStore.peek_analysis` |
| Cue times (start and end of each cue in the take, or null when unplaced) | `AnalysisRecord.alignment.cues[]` (`CueTiming`) and `alignment.flags` | `peek_analysis` |
| The render behind the take: seed, engine profile, whether it hit the token cap, generation time, canary status | `renders` table (`record` = `RenderRecord`: `seed`, `engine`, `hit_token_cap`, `gen_s`, `canary.batch_status`); the raw file is `renders/<ab>/rn_<16hex>/raw.wav` | `NarrationStore.peek_render` |
| The job's markdown report | `jobs/<job_id>/report.md` and `report.json` | a file read. **These exist only after some client called `get_results` on the finished job** (KNOW, `backend/service.py` `_generation_results` writes them there), so the view cannot rely on them and assembles the same content itself. |

**What the player plays:** the delivery WAV, which is what the caller receives (PCM 24-bit mono,
loudness-normalised, `DeliveryAudio.format`). The browser decodes it; the view decodes nothing (C4).

**The data the MCP answer adds:** `get_results` shows each take through `narration.backend.assemble`, which
also reads the voice's measurement and the alignment benchmark and computes the `listen_first` list. That
module reads through `get_render_by_id`, `get_take_by_id` and `get_analysis_by_id` (`backend/assemble.py`),
which **repair** the index: a row whose file is gone is deleted (`store.py` `_drop`). The view must not do
that (C1). WP46-B (§9) either gives `assemble` a read-only reader or re-derives what the view needs from the
`peek_*` reads.

### 3.3 Failed takes

`narration-admin failures` already builds this list (WP48, PR #45).

| What is shown | Where it lives | Read with |
|---|---|---|
| Every failed or replaced take of a `generate` or `analyse` job, newest job first, with job, segment, attempt, seed, flags, metrics, thresholds, the WAV and whether it is still there, and the take that finally filled its slot | Assembled from `iter_jobs` and the `peek_*` reads (no stored list) | `narration.admin.failures.collect(store, filters)` returns `FailedTake`s; `listing_json` gives `narration.failures/v1` (`FAILURES_SCHEMA_ID`, described by `FAILURES_JSON_SCHEMA`) |
| How many of them `gc` would remove, and when | `gc`'s dry run (`narration.admin.gc`, `failures.retention_view`) | phase 2 (§8) |

`collect` takes a `NarrationStore`. It reads only through `iter_jobs` and `peek_*` (KNOW, its module
docstring and the lead's check at merge), so it runs unchanged on a read-only reader that offers the same
methods (WP46-A).

### 3.4 The measured voices

| What is shown | Where it lives | Read with |
|---|---|---|
| Each measurement: voice hash, clip sha256, engine profile, reliable length (`max_segment_chars`, `max_segment_seconds`), pace level and tolerance, similarity baseline, when it was measured | `measurements` table (primary key `voice_hash`, `engine_profile_id`; `record` = `MeasurementRecord`); `measurements/<voice_hash>/<engine_profile_id>/measurement.json` | **no method lists them all today**: `measurements_of(voice_hash)` needs a hash. The reader adds one SELECT. |
| The length ladder: each rung's seeds, verdicts, pace and similarity | `MeasurementRecord.ladder[]` (`LadderRung`, `LadderSeed`) | the `record` |
| **A measurement the service no longer reads** (an older schema, or pace in another method's units) | the same rows; `_measurement` in `store.py` returns None for them | **Useful to show:** such a voice now answers `VOICE_NOT_MEASURED` (the situation after WP47's pace change, HANDOFF.md). The view labels it "measured under an older method; `measure_voice` again". |
| Where the voice came from | designed here: `provenance` table and `provenance.jsonl`, joined to `candidates` (`design_id`, `record` = `Candidate` with its `description`); allowed by the operator: `[voices] allow_sha256` (§3.5) | a SELECT; the allowlist's comment usually names the clip's file |

### 3.5 The allowlist

| What is shown | Where it lives | Read with |
|---|---|---|
| Each allowed sha256 with the comment on its line (usually the clip's file name) | **the configuration file, not the store**: `[voices] allow_sha256` in `narration.toml` | `narration.admin.allowlist.entries(text)`, which `voices list` uses |
| The clips the service designed itself | `provenance` table (`clip_sha256`, `design_id`, `date`) | a SELECT |
| Whether each allowed clip has been measured | the join of the list with the `measurements` table on `clip_sha256` | a SELECT |

### 3.6 The engine pins and the canary

| What is shown | Where it lives | Read with |
|---|---|---|
| The profile in use for each engine (`base`, `design`) | `settings` rows `current_engine_profile.base` and `current_engine_profile.design` | `NarrationStore.current_engine_profile` |
| Every pinned profile: id, hash, model repo and revision, packages, dtype, determinism settings, tier, VRAM need | `engine_profiles` table (`record` = `EngineProfile`); `engines/<engine_profile_id>.json` | `NarrationStore.list_engine_profiles` |
| The canary: its clip, threshold, when it was pinned, the material it came from | `EngineProfile.canary` (`CanaryPin`); the clip is `engines/<engine_profile_id>/canary.wav` | the profile's `record` |
| The canary's result on recent renders | `RenderRecord.canary.batch_status` (`hash_match`, `similarity_pass`, `not_run`) on each render | the `renders` table, newest first |
| The last `ENGINE_DRIFT` refusal | a failed job's `JobRecord.error` with that code | the `jobs` table |
| The alignment benchmark's measured error | `alignment_benchmarks` table; `alignment/<method_id>.json` | `get_alignment_benchmark(method_id)`; the method id is not in the store (§7 item 5) |

### 3.7 The daemon and the GPU

| What is shown | Where it lives | Read with |
|---|---|---|
| State (`stopped`, `idle`, `busy`, `stopping`), pid, start time, why it stopped, its workers, the current job | `run/daemon.json` (`DaemonStatus`, including `stop_reason` since contracts 1.6.11) | `NarrationStore.get_daemon_status` (retries a read that Windows refuses during the daemon's rename) |
| Whether that daemon really still runs | the pid in `run/daemon.json`, checked against the process table | `narration.daemon.sweep.daemon_alive` (what `daemon status` uses) |
| The last detached launch | `run/launch.json` | a file read |
| The GPU as the daemon last saw it: name, total and free VRAM, which model group is loaded, when it unloads, what each group needs, since when a job waits | `DaemonStatus.gpu` (`GpuStatus`) | `get_daemon_status` |
| **The GPU now**, including when no daemon runs | NVML | `narration.jobs.gpu.NvmlProbe` (`nvidia-ml-py`, BSD-3-Clause, already a dependency). `doctor` reads the GPU this way; `NvmlProbe` imports `pynvml` only when it is used, so a machine without the driver still works (KNOW, `jobs/gpu.py`). **BELIEVE:** NVML allocates no VRAM and starts no CUDA context (the operator guide's `doctor` section says so; not measured here). |

`DaemonStatus.gpu.free_mb` is only as fresh as the daemon's last write, and a stopped daemon's file keeps its
last reading. The view shows the NVML reading beside it, each with its time.

---

## 4. Reading the store safely while the daemon writes

This section applies to both options.

### 4.1 What the store does today (KNOW)

- The database is `<store_root>/narration.sqlite`, in WAL mode, with a 30 s busy timeout; every write takes
  the write lock at once with `BEGIN IMMEDIATE` (`store/db.py`).
- **Opening a `NarrationStore` writes.** Its constructor runs `db.migrate` (a `BEGIN IMMEDIATE` transaction
  that sets `PRAGMA user_version`) and `_reconcile_provenance` (a second write transaction). Running the
  migration again on an up-to-date database changed the database file's bytes in the probe (appendix A,
  step 2). `failures`'s docstring says the same: opening the store "touches only SQLite's own file header".
- **`NarrationStore` creates a store that is not there** (`Admin.store`, `admin/cli.py`: "created if it is
  not there yet"). A view pointed at the wrong folder would create an empty store there.
- **Most `get_*` reads repair the index.** `get_render_by_id`, `get_take_by_id`, `get_analysis_by_id`,
  `get_measurement` and others delete a row whose file is missing (`_drop`). The reads that never write
  are the `peek_*` reads, `iter_jobs`, `get_job`, `queued_jobs`, `pending_commands`, the engine-profile
  reads (`get_engine_profile`, `list_engine_profiles`, `current_engine_profile`) and `get_daemon_status`.
- **`touch`** updates last-use times; nothing in a view may call it.

### 4.2 How the view reads (proposed; both options)

1. **A separate read-only reader**, `narration.store.readonly.StoreReader` (WP46-A), that:
   - opens `file:<store_root>/narration.sqlite?mode=ro` as a URI, then sets `PRAGMA query_only = ON`
     as a second guard. SQLite refuses a write on that connection (KNOW, appendix A, step 4);
   - never runs `db.connect` (which sets `journal_mode`) or `db.migrate`. It reads `PRAGMA user_version`
     and refuses a store whose schema is newer than the code's (`SCHEMA_VERSION`), with a hint to use that
     version of the service;
   - refuses a `<store_root>` with no database (it never creates one);
   - offers the never-writing reads the view needs, with the same names and return types as
     `NarrationStore`'s (`iter_jobs`, `get_job`, `queued_jobs`, `peek_render`, `peek_take`, `peek_analysis`,
     `get_daemon_status`, `list_engine_profiles`, `current_engine_profile`, `pending_commands`) plus the
     lists the store lacks (all measurements, provenance, candidates). So `failures.collect` runs on it
     unchanged.
2. **Short snapshots.** Each page or refresh reads in one `BEGIN … COMMIT` (a consistent snapshot, as
   `db.read_txn` does), then ends it before rendering. A read never waits for the daemon's writes and never
   blocks them: in the probe, a read-only reader read at once while a writer held an uncommitted
   `BEGIN IMMEDIATE`, and saw the writer's row only after the commit (KNOW, steps 5 and 6).
3. **Never an open transaction between refreshes.** While the probe's reader held one, a `TRUNCATE`
   checkpoint came back busy and the write-ahead log kept its 402 frames; once the reader ended it, the
   checkpoint finished (KNOW, steps 7 and 8). A dashboard that keeps a connection open between requests is
   fine; one that keeps a transaction open is not.
4. **Files are read after the rows.** A take's WAV can be removed by `gc --apply` between the row read and
   the file read. The view shows "file not in the store" then, as `failures` does, and never repairs
   anything.
5. **Scale.** `iter_jobs` streams rows newest first. The view shows the newest N jobs (ASSUME: 50) with a
   `--since` filter, as `failures` does, so a store with thousands of jobs is not read whole on every
   refresh.

**What the probe did not cover (BELIEVE):** it ran on Linux (SQLite 3.45.1, the version this repository's
venv links). SQLite documents that a read-only connection opens a WAL database when the `-wal` and `-shm`
files exist or can be created; the probe opened one with neither file present. WP46-A repeats the probe on
Windows in CI, including while another process holds the database open, because Windows file locking
differs.

---

## 5. Option A: a static HTML report written by `narration-admin`

```
narration-admin report --out <dir> [--since DATE] [--job ID] [--watch [SECONDS]] [--json]
```

It reads the store through the read-only reader, writes `index.html` (daemon, GPU, engine, queue, recent
jobs), one `job-<job_id>.html` per job shown, `failures.html`, `voices.html` and `engine.html` into `<dir>`,
and prints the path of `index.html` to open in a browser. `--json` writes the same data as
`narration.view/v1` for other tools. `--watch` rewrites `index.html` every N seconds (default 5) until
Ctrl+C.

### 5.1 Security

- **No socket.** Nothing listens; there is nothing for another machine, another local user or a web page to
  connect to. The design's rule (§0 item 11: "SQLite + files, no sockets"; §17.1: "no process listens on a
  socket") stands as it is.
- **The output holds the caller's text.** Segment text and transcripts are in the pages, as they are in
  `failures --export`. The report is written only to a folder the operator names, which must be outside the
  store (the check `failures.export_dir` already makes, including through links). Files are written under a
  temporary name and renamed (AGENTS.md §6).
- **Injection.** Every value from the store is HTML-escaped (`html.escape`), including segment text, labels,
  transcripts and flag messages: they are data, never markup (§17.11). Each page carries a
  Content-Security-Policy `<meta>` that forbids scripts entirely (`script-src 'none'`), allows no network
  fetch (`connect-src 'none'`, `default-src 'none'`), and allows images, media and styles only from the
  page's own files (`img-src`, `media-src file:`, an inline style block by hash). So even a missed escape
  cannot run code or send anything anywhere.
- **Audio links.** Each player is `<audio controls preload="none" src="file:///…/delivery.wav">`, pointing
  at the file in the store, whose path came from the store's own index and was checked to be under
  `<store_root>` (C7). The report therefore only works on this machine, which is what C2 asks.
  **BELIEVE:** Edge, Chrome and Firefox play a `file://` media source from a `file://` page (a media load is
  not a cross-origin read). WP46-C checks this on Windows first; if a browser refuses, `--copy-audio` copies
  the WAVs next to the pages, as `failures --export` does.
- **Stateless.** The report is the operator's output, like `failures --export`; the store records nothing
  about it (§0.2).

### 5.2 Freshness

- A snapshot: the pages are as fresh as the last run. Re-running is instant for a store of the size the
  owner has (**BELIEVE**: well under a second for hundreds of jobs; WP46-C measures it).
- `--watch` keeps `index.html` current: it rewrites it every N seconds, and the page reloads itself with
  `<meta http-equiv="refresh">` (which needs no script). Only the status page refreshes; the job pages do not,
  because a reload would stop the audio player mid-take. A job page says when it was written and how to
  refresh it.
- Without a script, a `file://` page cannot fetch new data by itself (browsers block `fetch` of `file://`
  URLs; BELIEVE), so a live progress bar that moves without a reload is not possible in this option.

### 5.3 Dependencies

**None new.** The standard library's `html`, `string.Template` and `json`, and `nvidia-ml-py`
(BSD-3-Clause), already a dependency, for the GPU. No template engine (Jinja2, BSD-3-Clause, would be
acceptable, but is not needed for pages this simple). No JavaScript and no CSS framework.

### 5.4 Effort

About 5–6 agent-days for version 1, including review (ASSUME; §9).

---

## 6. Option B: a read-only dashboard served on 127.0.0.1

```
narration-admin view [--port N] [--no-browser]
```

A foreground process, separate from the daemon (the daemon stays socket-free), that serves the same pages
from the read-only reader, and a small JSON endpoint the pages poll for status and progress. Ctrl+C stops it.
It opens the browser at a one-time URL.

### 6.1 Security

A port on 127.0.0.1 cannot be reached from another machine, but it can be reached by **every process of every
user on this machine**, and, through the browser, by **every web page the operator has open**. Each of the
following is required, and each is a place a bug would expose the caller's text and audio:

1. **Bind exactly `127.0.0.1`**, never `0.0.0.0`, `::` or the name `localhost` (which can resolve to more
   than one address). A `--host` option is not offered; the server checks the bound address after binding
   and exits if it is not loopback. The port is chosen by the OS (`0`) unless `--port` names one, so there is
   no well-known port to probe. **BELIEVE:** a loopback-only listener does not raise a Windows Defender
   Firewall prompt.
2. **DNS rebinding.** A web page on a hostile domain can make its own name resolve to 127.0.0.1 and then
   read the dashboard's responses as "same-origin". The defence is to refuse any request whose `Host`
   header is not exactly `127.0.0.1:<port>` (421 or 403), before routing.
3. **Other local users and processes.** They can connect to the port directly. The server makes a random
   32-byte token at start and prints and opens `http://127.0.0.1:<port>/?t=<token>`. The first request
   exchanges it for a cookie (`HttpOnly; SameSite=Strict; Path=/`) and redirects to a URL without it. Every
   request without the cookie gets 403. The token is compared in constant time (`hmac.compare_digest`).
4. **CSRF.** There is nothing to forge: every route is `GET` or `HEAD`, and anything else gets 405. As a
   second guard, a request whose `Sec-Fetch-Site` is `cross-site` is refused, and so is one with an `Origin`
   other than the dashboard's own.
5. **Cross-origin reads.** No CORS headers are ever sent, so another origin's script cannot read a
   response. `Cross-Origin-Resource-Policy: same-origin` stops another page embedding the audio.
6. **Headers on every response:** a Content-Security-Policy (`default-src 'self'`, no inline script except by
   hash, `frame-ancestors 'none'`), `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
   `Cache-Control: no-store` (the pages hold the caller's text).
7. **Audio by id, never by path** (C7). `/audio/<take_id>` looks the take up in the index, checks that the
   path is under `<store_root>`, and serves only a take the view has listed. No route takes a path.
8. **It serves no directory listing and no static folder** beyond its own packaged CSS.

§17.1 says of a socket, "if ever needed: 127.0.0.1, an `Origin` check, a bearer token, off by default". This
option follows that, but it is still a **design change**: §0 item 11 and §17.1 say no process listens on a
socket. It needs a DC in `docs/design-log.md` and the owner's approval.

### 6.2 Freshness

Live: the status panel and a running job's progress poll a JSON endpoint (ASSUME: every 2 s while a job runs,
every 15 s otherwise), so they move without a reload, and the player keeps playing.

### 6.3 Dependencies

Two ways to serve:

| Choice | Licence | Notes |
|---|---|---|
| **Standard library** `http.server.ThreadingHTTPServer` | PSF-2.0 | No new dependency. It has no HTTP range support, so the browser's player cannot seek within a take until the view implements `Range` itself (**BELIEVE**, from the module's documented behaviour); the Python docs warn it implements only basic security checks, so every control in §6.1 is hand-written. |
| **Starlette + Uvicorn** | BSD-3-Clause both | Already in `uv.lock` as dependencies of `mcp` 2.2.0 (KNOW: `uvicorn` 0.54.0, `starlette` 1.7.0, `h11` 0.16.0 MIT, installed by `uv sync`), so nothing new is downloaded. Declaring them directly needs the lead's approval (plan.md P7), and pinning them in step with `mcp`. Starlette's file response handles range requests (**BELIEVE**; to verify on the pinned version). |

No JavaScript framework: a few dozen lines of the view's own script for polling, served from the package, with
no CDN (C2).

### 6.4 Effort

About 8–10 agent-days for version 1, including review (ASSUME; §9): everything in option A's reader and data
model, plus the server, the security controls of §6.1 and their tests (a request with a rebound `Host`, a
missing cookie, a cross-site `Origin`, a `POST`, a path in place of a take id), and the design change.

---

## 7. The two options side by side

| | A: static report | B: dashboard on 127.0.0.1 |
|---|---|---|
| Attack surface | None beyond files the operator writes | A port every local process and every open web page can reach; eight controls (§6.1) |
| Design change | None (one command added to §7.1's list) | Yes: §0 item 11 and §17.1 (a DC) |
| Freshness | A snapshot; `--watch` refreshes the status page by reload | Live, without reloads |
| Listening while it updates | Job pages never reload | Uninterrupted |
| New dependencies | None | None (stdlib) or two already in `uv.lock` (BSD-3) |
| Reads the store | Read-only reader, one snapshot per run | Read-only reader, one snapshot per request |
| GPU | NVML only | NVML only |
| Effort, v1 (ASSUME) | 5–6 agent-days | 8–10 agent-days |
| Works with the daemon stopped | Yes | Yes |
| Leaves a copy of the caller's text on disk | Yes, in the folder the operator names | No (unless the browser caches; `no-store` asks it not to) |
| Shareable | The operator can zip the folder (their choice, and their text) | No |

---

## 8. Recommendation and phased scope

**Recommendation: option A, the static report, with `--watch`.** It meets all four constraints without a
socket, adds no dependency, fits the operator commands that already exist (`failures`, `gc`, `verify` all
print or write a view and exit), and leaves the design's no-socket rule intact. Its real cost is freshness:
the owner sees a running job's progress by glancing at a page that reloads itself, not by watching a bar
move, and a job page must be re-generated to show new takes. For a single operator auditing takes and
checking on a job, the lead's judgement (BELIEVE) is that this is enough; question 1 in §10 asks the owner.

The option-B path stays open: the reader and the data model (WP46-A, WP46-B) are the same for both, so a
dashboard later reuses them and adds only the server.

### Version 1

1. **Status page** (`index.html`, refreshed by `--watch`): the daemon's state, pid, start time and stop
   reason, and whether it still runs; its workers; the current job and phase; the queue with the estimated
   drain; pending commands; the GPU as the daemon last saw it and as NVML sees it now; the engine profile in
   use for each engine.
2. **Jobs list:** the newest 50 jobs (or `--since`, `--job`), with kind, label, status, phase, progress,
   outcome and error.
3. **Job page** (generation and analysis jobs): per segment, its state, text per cue, suggested take and why;
   per attempt, verdict, flags (fail first), WER, similarity, pace against its expected value, cue times, and
   a player for the delivery WAV (or "file not in the store"). Replaced attempts are shown, greyed, with
   what replaced them (the rule `failures` and the job report share, `qa.report.replacements`).
4. **Failures page:** `failures.collect` as it is, with a player per take.
5. **Voices page:** every measurement, current or not, with reliable length, pace, similarity baseline and
   ladder; which are no longer read, and why; the allowlist with its comments; which clips were designed
   here.
6. **Engine page:** every pinned profile, the one in use, its canary (threshold, when pinned, a player for the
   canary clip) and the canary status of the latest renders.
7. **`--json`** writes the same data as `narration.view/v1`, with its JSON Schema published like the others
   (no `$ref`, §5).

### Phase 2 (after the owner has used version 1)

- `gc`'s retention view: which takes and measurements `gc --apply` would remove, and when.
- Design jobs (candidates, descriptions, lint findings, profile pictures and players) and profile jobs.
- Measure jobs in progress (the ladder filling in rung by rung).
- The daemon's log tail (it logs no caller text, WP49), and the alignment benchmark.
- Option B, if the owner wants live progress.

### Never (by the constraints)

Approving, choosing or marking takes; cancelling jobs; releasing the GPU; running `gc`; allowing a clip;
anything reachable from another machine; anything served from or sent to the internet; an MCP tool for any of
it.

---

## 9. Work packages and estimate

In plan.md's format. Estimates are agent-days including one independent review and its fixes (ASSUME: taken
from the sizes of comparable packages here, such as WP48 and WP45, not measured).

| WP | Title | Does | Depends | GPU | Estimate |
|---|---|---|---|---|---|
| WP46-A | Read-only store reader | `narration.store.readonly.StoreReader` (§4.2): `mode=ro` + `query_only`, schema check, no create, no repair, the never-writing reads with `NarrationStore`'s names, and the lists the store lacks (all measurements, provenance, candidates). *Accept:* on a store the fake worker filled, every read returns what `NarrationStore`'s does; the database file, `-wal` and every store file are byte-identical before and after (hashes); a write through it raises; a newer schema is refused; a missing store is refused and not created; the probe of appendix A passes on Windows and Linux in CI, including against a second process that holds a write transaction. | the owner's approval | – | 1–1.5 |
| WP46-B | The view's data model | `narration.view` (a new area): builds each page's data from a `StoreReader` (§3), and `narration.view/v1` with its JSON Schema. Reuses `failures.collect`; gives the take's QA and cue times from `peek_analysis`; derives "replaced" with `qa.report.replacements`. *Accept:* a job with planted failures (the fake worker) gives the same verdicts, flags, suggestion and replaced attempts as `get_results` and `failures`; a take whose file `gc` removed shows as missing; no store write (the WP46-A hash check). | WP46-A | – | 2 |
| WP46-C | `narration-admin report` | Option A's command (§5): the pages, escaping and CSP, the `file://` players (or `--copy-audio`), the output-folder checks from `failures.export_dir`, `--watch`, `--json`. *Accept:* every store value appears escaped (a planted `<script>` in a segment's text renders as text); no page references anything but its own folder and the store's WAVs; the export folder inside the store is refused; a run on a store with 500 fake jobs finishes in under 2 s (ASSUME as a target); the store is unchanged. | WP46-B | – | 1.5–2 |
| WP46-D | Docs and design text | The operator guide's section; the command in §7.1's list, §15 (what the view reads), §17 (why it holds no socket); the CHANGELOG line. | WP46-C | – | 0.5 |
| *(phase 2)* WP46-E | Dashboard on 127.0.0.1 | Option B (§6), only if the owner chooses it: the DC first, then the server and every control of §6.1 with a test each. | WP46-B, a DC | – | 3–4 |

**Version 1 (A to D): about 5–6 agent-days.** WP46-A and the part of WP46-B that does not depend on it can
start together; WP46-C follows B. The GPU is never needed, so none of it waits for a GPU window, and none of it
changes the engine's identity (HANDOFF.md's "hold any merge that changes the engine's identity" does not apply).

**Contract and dependency requests** (the lead's to approve):
- A `StoreReader` protocol in `narration.contracts.interfaces`, or the reader kept in `narration.store` only
  (the lead's call): `failures.collect` and the view would accept either `NarrationStore` or the reader.
- `narration.view/v1` as a new schema id in `narration.contracts.names`.
- No new dependency for option A. For option B, Starlette and Uvicorn declared directly (already locked).

---

## 10. Open questions for the owner

1. **Static report or live dashboard?** The lead recommends the static report with `--watch` (§8). A live
   dashboard means a design change and a socket (§6.1). If live progress without reloads matters to you, say
   so and the plan becomes WP46-A, B, E.
2. **Which browser do you use?** The report's players rely on `file://` audio from a `file://` page (§5.1,
   BELIEVE); WP46-C tests your browser first.
3. **Should the report copy the audio by default** (a folder you can move or keep after `gc`, costing disk),
   or link to the store's files (nothing copied, but a link breaks once `gc` removes the take)?
4. **Where should the report go by default?** It holds your text. The plan requires `--out`, outside the store,
   with no default. A default under the configuration's folder is possible if you prefer.
5. **How many jobs by default?** The plan shows the newest 50.
6. **Anything that acts, ever?** The plan has none (C1). `cancel_job` and `release_gpu` exist as MCP tools; a
   view that can call them would no longer be read-only. The lead recommends keeping it that way.
7. **Is the view for this machine's operator only**, as planned, or should its `--json` be a stable interface
   for other tools (which makes `narration.view/v1` a public schema under SemVer)?
8. **Measure and design jobs in version 1?** The plan puts them in phase 2.
9. **Timing.** plan.md schedules WP46 after the other packages. Version 1 needs no GPU and changes no engine
   identity, so it could run beside the GPU work (WP38–WP42) without disturbing a narration session.

---

## 11. What would change this plan

- **The report's freshness proves too slow in use** (the owner keeps re-running it during a job): build
  WP46-E on the same reader and data.
- **A browser refuses `file://` audio** from a `file://` page: `--copy-audio` becomes the default.
- **The read-only connection fails on Windows** while the daemon holds the database (WP46-A's probe): the
  reader copies the database with SQLite's backup API into a scratch file outside the store and reads the
  copy, which costs a copy per refresh but never touches the daemon's file.
- **The store grows past what a full read handles** (thousands of jobs): pages read by `--since` only, and the
  jobs list gets a lighter SELECT of its columns instead of each job's whole `record`.

---

## Appendix A: the SQLite probe (KNOW)

Run 2026-09-28 in a Linux sandbox with this repository's venv (Python 3.12, SQLite 3.45.1), on an empty
store made by the service's own `db.connect` and `db.migrate`, in a scratch folder outside the repository.
The script is kept here rather than in `spikes/`, since this package adds documents only until the owner
approves it; WP46-A turns it into a test.

```python
"""Probe: how a read-only reader behaves beside the store's own connections (WP46 plan evidence)."""

import hashlib, sqlite3, sys, time
from pathlib import Path
from narration.store import db

root = Path(sys.argv[1]) / "store"
root.mkdir(parents=True, exist_ok=True)
path = root / "narration.sqlite"
w = db.connect(path)
db.migrate(w)
w.close()


def state():
    return {p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()[:12]) for p in sorted(root.iterdir())}


print("1 after create+close:", state())
# (a) re-running migrate on an up-to-date store: does anything change?
before = state()
c = db.connect(path)
db.migrate(c)
c.close()
print("2 migrate again changes files:", before != state(), state())
# (b) mode=ro with no -wal/-shm present
uri = f"file:{path.as_posix()}?mode=ro"
r = sqlite3.connect(uri, uri=True, isolation_level=None)
r.execute("PRAGMA query_only = ON")
print(
    "3 ro open, jobs rows:",
    r.execute("SELECT count(*) FROM jobs").fetchone(),
    "journal:",
    r.execute("PRAGMA journal_mode").fetchone(),
)
try:
    r.execute("INSERT INTO settings VALUES ('x','y')")
except sqlite3.Error as e:
    print("4 ro write refused:", type(e).__name__, e)
# (c) reader while a writer holds BEGIN IMMEDIATE and has uncommitted rows
w = db.connect(path)
w.execute("BEGIN IMMEDIATE")
w.execute("INSERT INTO settings VALUES ('a','1')")
t = time.perf_counter()
n = r.execute("SELECT count(*) FROM settings").fetchone()[0]
print(f"5 read during writer txn: rows={n}, {1000 * (time.perf_counter() - t):.2f} ms (not blocked)")
w.execute("COMMIT")
print("6 read after commit:", r.execute("SELECT count(*) FROM settings").fetchone()[0])
# (d) a long read txn blocks checkpoint reset (WAL growth)
r.execute("BEGIN")
r.execute("SELECT count(*) FROM settings").fetchone()
for i in range(200):
    w.execute("INSERT INTO settings VALUES (?, 'v')", (f"k{i}",))
print(
    "7 checkpoint while reader holds snapshot:",
    tuple(w.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()),
    "(busy, log, ckpt)",
)
r.execute("COMMIT")
print("8 checkpoint after reader ends:", tuple(w.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()))
r.close()
w.close()
```

Output (the hashes differ from run to run, since the migration records the time it ran; what matters is
that line 2's differ from line 1's):

```
1 after create+close: {'narration.sqlite': (176128, '3fe905848ffd')}
2 migrate again changes files: True {'narration.sqlite': (176128, '8f67fa700443')}
3 ro open, jobs rows: (0,) journal: ('wal',)
4 ro write refused: OperationalError attempt to write a readonly database
5 read during writer txn: rows=0, 0.02 ms (not blocked)
6 read after commit: 1
7 checkpoint while reader holds snapshot: (1, 402, 2) (busy, log, ckpt)
8 checkpoint after reader ends: (0, 0, 0)
```

What it shows: opening the store the way `NarrationStore` does changes the database file even when there is
nothing to migrate (2); a `mode=ro` connection opens a WAL database with no `-wal` or `-shm` present, and
SQLite refuses a write through it (3, 4); it reads without waiting while a writer holds the write lock, and
sees only committed rows (5, 6); and a read transaction left open keeps the write-ahead log from being reset
(7, 8).
