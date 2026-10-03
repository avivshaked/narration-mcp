# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html); version numbers are written the Python way
([PEP 440](https://peps.python.org/pep-0440/)), so `0.1.0a1` is the first alpha of `0.1.0`. Until `1.0`,
schemas and configuration keys may change between releases, and every such change is listed here.

## [Unreleased]

### Changed

- The design's revision history and its table of design changes now live in `docs/design-log.md`; `docs/design.md`
  states the design as it is.

## [0.1.0a1] - 2026-09-29

The first tagged release: an alpha for testers. Everything below, since the project began, is in it.

### Added

- A "Help test it" section in the README, and a "Tester report" issue form: what a tester needs, how far to
  go, and what to report.

- `audition_pronunciation` now runs (`narration.audition`; design revision 5.18, section 7.6). It renders a
  term with up to four respelling variants, in your carrier sentence or alone, in a voice that need not be
  measured. Each variant's take is rendered, keyed and cached as a `submit_job` take of the same text with
  that respelling as its hint, so sending the audition again renders nothing, and narrating the carrier with
  the chosen respelling reuses the render. Each take is QA'd like a narration take, except that its verdict
  has no speaker or pace check; the respelling counts as the term, so a take that says the respelling is not
  scored as a misheard name. A take that fails QA is retaken, up to `[defaults] max_retakes`. `get_results`
  gives, per variant, its engine text and takes, what the recogniser heard in the term's place (`heard`), and
  each take's speaker similarity to the voice's clip (`spk_sim_clip`, information only, never part of a
  verdict). Contracts 1.6.12: `AuditionVariantResult` gains `spk_sim_clip`; nothing is removed or renamed, and
  no key, seed or cached record changes. The service records no choice. The tool no longer answers
  `BACKEND_NOT_INSTALLED`, and no published tool is marked "not in this build" any more.
- `docs/licences.md`: a licence audit of every package in the three lock files and every pinned model
  (plan.md WP44), and `docs/security-review.md`: a review of the path, synthetic-voice, text, process and
  transport checks, with each finding's severity and state.
- User documentation for installing and running the service (plan.md WP43): the README now covers
  requirements, install, every `narration.example.toml` section, the `.mcp.json` entry, a first run and
  where the store and logs live; `docs/operator-guide.md` covers the daemon (including the Job Object
  breakaway rule), `doctor`, `install`, the engine pins and canary, the voice allowlist, `gc`, `verify`,
  `failures`, `render`, and what to do for each error code an operator sees. `docs/tools.md`, the MCP tool
  reference (every tool's description, input and output fields, and the error and flag codes), is now
  generated from the published tool definitions by `tools/gen_tool_reference.py`, which
  `tests/docs/test_tool_reference.py` keeps current in the default test suite.
- `narration-admin failures` lists every take that failed QA or that a retake replaced, across jobs, newest
  job first, for an audit: the job, segment, attempt and seed; the take's WAV in the store; each fail and warn
  flag with its message; the QA metrics; the take that finally filled the slot; and the segment's text.
  `--since`, `--job`, `--voice` and `--code` (a QA or aligner flag code) filter it, and `--json` prints it
  as data. `--export <dir>` copies each take's WAV with a JSON sidecar of its reasons, plus an `index.csv`
  (UTF-8 with a byte-order mark; a cell a spreadsheet would run as a formula gets a `'` in front). The
  bundle names its files by take id, never by a path in the store. It never overwrites a different file,
  refuses a folder inside the store (also through a link or another name for it), and says what to do when
  a file cannot be written. It never changes the store: a take whose WAV was deleted by hand is listed as
  "file not in the store" and its index row is left as it is. The job report
  (`narration://jobs/{job_id}/report`, and `report.json`) gains a "Failures" section with the same takes,
  by the same rule for which retake replaced which, and `narration-admin gc` lists them apart (how many, how
  old, which the run would remove, and which would only lose their flags and metrics), so an audit can
  finish before retention removes them.
- `design_voice` and `profile_voice` now run (`narration.design`). A design renders 1 to 4 candidates
  with Qwen VoiceDesign, each from its own seed, which is derived from the description, the design text
  and the candidate's number, so the same request designs the same voices. Each candidate's clip is
  checked against its exact transcript by the speech recogniser (`WER_HIGH` if it does not say it),
  profiled, and published with its seed, lint result and profile. Its sha256 goes on the provenance
  list, so `measure_voice` and `submit_job` accept the clip with no `allow_sha256` edit. Each candidate
  carries its flags: `TOKEN_CAP_HIT`, `WER_HIGH` and `CLIP_TOO_LONG` (longer than `measure_voice` takes:
  use a shorter `design_text`) fail it, and the job then ends `needs_attention`; `CANARY_MISMATCH` is
  information, when the design engine passed its canary on similarity rather than on the hash. A
  description that holds the model's chat markup (`<|`, `|>`) or a control character is refused with
  `TEXT_REFUSED`. `profile_voice` profiles any WAV you can read, by path and sha256, on the CPU (or on
  the loaded QA worker as it is), answers the same bytes from the cache, and keeps no copy of the audio
  once the job ends. Neither tool is marked "not in this build yet" any more, and the server's
  instructions and prompts now start the flow with `design_voice`; only `audition_pronunciation` keeps
  the mark.
- `narration-admin voices allow <clip.wav>` adds a clip designed elsewhere to `[voices] allow_sha256`,
  instead of hashing it and editing `narration.toml` by hand.
  - It makes the clip's path absolute (so a relative path is taken from the working folder), checks it
    with the service's path check, and refuses a file that is not a WAV.
  - It prints the clip's path, length and sha256, and asks you to confirm that the clip is synthetic, not
    a recording of a real person. Only `yes`, typed in full, goes on; no answer changes nothing. A person
    must confirm: `--yes` is only for the operator's own scripts.
  - The hash goes into the configuration file with a comment naming the clip. Every other line, comment
    and line ending is kept. The file is replaced whole, and not if it changed while the command ran; that
    is checked just before the rename, so a short window remains (do not run two at once). The new file
    keeps the old one's POSIX mode bits, but not its owner or ACL.
  - It ends by saying to stop the daemon and then restart `narration-mcp`: both read the list only when
    they start.

  `narration-admin voices list` prints the list. Neither is an MCP tool: only someone who can run
  `narration-admin` on this machine, or edit its configuration, can change the list.
- `measure_voice`'s job (`narration.measure`): it checks the clip's transcript with the speech recogniser
  (`REF_TEXT_MISMATCH` if the clip does not say it), renders the calibration set and builds the voice's
  anchor and similarity baseline, then climbs the length ladder from the shortest rung, fits the pace
  trend, and stops at the first rung the voice does not read reliably. It publishes `measurement.json`
  with `max_segment_chars`, `max_segment_seconds`, the pace curve and the ladder table. Every take comes
  from the cache when it can, so a measurement that stopped resumes where it was.
- `narration-admin render --voice <clip> --transcript <text> --text <text> [--out <wav>]` renders one
  text in a voice from the terminal. It goes through the service's own `submit_job`, so every check
  applies (a synthetic, measured voice; the clip's path and sha256; the limits). It waits for the job,
  prints each take's verdict, delivery file and cue times, and copies the suggested take to `--out`.
  `--dry-run` only plans; `--json` prints the results as `get_results` returns them. Running the same
  command again while its job runs waits for that job.
- The same request sent again while its earlier job is being cancelled now makes a new job, instead of
  returning the job that is ending. A job being cancelled also stops holding its `idempotency_key`.
- `narration-mcp` now serves the real tools over the store and the daemon. It finds its configuration
  from `--config`, then `NARRATION_CONFIG`, then `narration.toml` in the service's folder. `submit_job`
  checks the request, plans it against the cache and queues it; a `dry_run` only plans. The same
  request while its job is active returns that job. The server copies the voice clip into the store,
  then starts the daemon. A voice that has not been measured is refused with `VOICE_NOT_MEASURED`.
  `get_job` long-polls and reports progress. `get_results` returns the QA'd, cue-aligned takes and
  writes `report.md` and `report.json` beside the job. `measure_voice` answers a current measurement
  at once, without a job, and otherwise queues one. `check_text`, `cancel_job`, `get_server_status`,
  `release_gpu` and the `narration://` resources also work. Sending a default (`text_mode`, empty
  `hints`, default options) and leaving it out make the same job. Retryable errors
  carry `retry_after_s`, and `[limits]` caps the submit rate and the queue. `design_voice`,
  `profile_voice` and `audition_pronunciation` still answer `BACKEND_NOT_INSTALLED`. README.md says
  how to add the server to Claude Code.
- Engine profiles (`narration.engine`): each pins the Qwen model's revision, every snapshot file's sha256,
  the worker's `uv.lock` and package versions, the determinism switches and every audio-changing setting,
  and hashes them. After every Qwen load the daemon checks the worker against its profile: a changed
  `uv.lock`, weight file or package version fails the job with `ENGINE_DRIFT` before anything renders. A
  weight file replaced by another with the same size and time (moved over it) is found too, without a
  restart.
  The daemon's job engine is now assembled with the cue aligner and the pinned QA models (Whisper
  large-v3, WavLM-base-plus-sv, the wav2vec2 aligner), so jobs no longer fail with
  `BACKEND_NOT_INSTALLED` once the models are installed and the engine is pinned. The daemon runs
  `measure_voice`'s jobs on the same engine, sharing its resident models and its canary gate.
- `narration-admin engine pin | repin | bridge | show`. `pin` records the Base and VoiceDesign engine
  profiles and designs the service's canary on this machine from the text in `material/canary/`: no canary
  audio ships. It repeats the canary render in one worker and in a fresh one to decide the determinism
  tier (`bit_exact` or `similar`), and calibrates the canary's similarity threshold with three more seeds,
  never below 0.10. A pinned canary embedding that cannot be compared (empty, all zero, or of another
  length than the render's) fails the job with `ENGINE_DRIFT`, saying why. `pin` keeps what is pinned and
  refuses an installation that has changed. In a profile it keeps, it records the snapshots' new folder
  after the models root moves, and a new estimate of the VRAM Qwen needs; neither changes the profile's
  hash (DC-16). `repin` makes a new profile the one in use (new render keys; voices are measured again)
  for each engine whose installation changed, or whose machine did (GPU, driver, CUDA or cuDNN, as the
  worker reports them); `repin --force` does it for both engines whatever changed. A GPU or driver the
  worker cannot read now (NVML failed) is not taken for a changed machine: `repin` refuses, says to run
  `doctor`, and names `--force`. A `repin` that keeps both profiles says which it refreshed in place
  (after the models root moved), or that nothing changed, naming `--force`. A pin or repin is all or
  nothing: both canaries are rendered, embedded and calibrated before anything is stored, so a failure on
  either engine leaves both as they were. Every `ENGINE_DRIFT` from the canary gate names `engine repin`;
  where the gate cannot name the cause (the canary moved, cannot be compared, or a worker failed during
  it), it adds `engine repin --force` for when the repin keeps the profile. A worker venv that does not
  match its lock is refused naming the command to run again. `show` prints each profile's tier and canary
  threshold (and whether it is the floor); `pin` and `repin` print the calibration similarities. `bridge`
  reports how similar the canary and the calibration corpus sound under two profiles; a profile replaced
  after the models root moved is rendered from its new folder (a recorded folder that holds none of its
  weight files counts as moved). They run only while no daemon holds the store and the GPU has room for
  Qwen.
- The job engine (`narration.jobs`), which the daemon runs: each round renders on Qwen, post-processes,
  then scores on the QA models, and retakes the takes that fail QA on the next attempt numbers, up to
  `max_retakes`. Work is looked up in the cache first and made once, even by overlapping jobs; the same
  request twice completes from the cache and is never retaken again. It loads one model group at a
  time, waits for free GPU memory (then `GPU_UNAVAILABLE` after `[gpu] wait_timeout_min`), retries an
  out-of-memory error once, lets interactive jobs go first, and reports `poll_after_s`, `retry_after_s`
  and the queue's drain estimate. A clip the worker cannot prepare fails the job with
  `UNSUPPORTED_AUDIO`; a caller's clip is read only through the daemon's path check, and only up to 20 MB.
  A job is `needs_attention` only when a segment's best take failed QA or a segment has no take. The
  daemon runs the job engine by default; until the aligner and the QA models are installed, each job it
  takes fails with `BACKEND_NOT_INSTALLED` and nothing is rendered. An `INTERNAL` error names the
  daemon's log file.
- `tools/check_private.py`, run by the git hooks and by a new pre-push hook on every pushed commit: it
  refuses a commit that copies a passage, a name or a distinctive number from private text the
  developer lists locally (`.dev/private-text.txt`). It does nothing when no private text is listed.
- Each Qwen render passes its own generation cap, min(8192, max(128, ceil(2.5 × characters))), set by
  `max_new_tokens_per_char` and `max_new_tokens_floor` under `[engines.*]`. A render that runs away
  stops after about a minute, not twenty, and is flagged `TOKEN_CAP_HIT`. A render that ends under its
  cap is unchanged by it.
- `[alignment]` gains `low_confidence_below` and `unplaced_below`, the cue-confidence thresholds. A cue
  whose text has no word the aligner can place is reported but not retaken, since no retake can place it.
- Repository bootstrap: the server `pyproject.toml`, worker project skeletons, `pytest`/`ruff`/
  `basedpyright` configuration, and the dev tools (`tools/gpu_lock.py`, `tools/check_tracked.py`, the git
  hooks, the worktree helper).
- Project licensing: `LICENSE` (PolyForm Noncommercial 1.0.0), `LICENSE-DOCS` (CC BY-NC 4.0),
  `COMMERCIAL.md`, `THIRD_PARTY_NOTICES`.
- `docs/design.md`: the design document, revision 5.1, as the source of truth for the implementation.
- `plan.md` and `HANDOFF.md`: the implementation plan and the current-state handoff.
- `CONTRIBUTING.md`: the contribution grant, environment setup with uv, the test tiers and markers, how
  to run GPU tests under the GPU lock, the required checks, and the dependency-licence rule.
- `CODE_OF_CONDUCT.md`: Contributor Covenant 2.1.
- `SECURITY.md`: the service's threat model in plain terms, and how to report a vulnerability.
- `.github/ISSUE_TEMPLATE/`: bug report, feature request and commercial-licence request templates, plus
  issue template configuration.
- `.github/pull_request_template.md`: a pre-submission checklist for pull requests.
- `narration.example.toml`: the shipped configuration file, with every key and default of the design's
  configuration section and no personal or machine-specific values.
- `material/`: the service's own text material, in draft until the owner has listened to sample renders
  (gate H1): the calibration corpus and length-ladder paragraphs (`narration-en.v1`), the alignment
  benchmark (`alignment-en.v1`), the canary's description, text and seeds (`canary.v1`), the Phase 4 demo
  script (`demo-en.v1`), and the text and QA fixtures (`text-v1`, `qa-faults-v1`), each set with a
  manifest of its files' sha256 hashes.
- `narration.platform`: every OS-specific mechanism behind one interface, implemented for Windows (the
  daemon's singleton, detached start, kill-on-close worker groups, below-normal priority, the free-disk
  check, and the path rules: caller paths must be absolute local regular files, network, device and
  reserved-name paths are refused, and store writes refuse links and junctions). On any other OS each
  call reports that v1 runs on Windows only.
- `narration.text`: the text pipeline (text checks `text-1.1.0`). It speaks the text as sent, changing
  only whitespace and Unicode form. It refuses markup and control characters (`TEXT_REFUSED`, listing
  every offender) and applies pronunciation hints per cue, never across a cue (`TERM_SPLIT_ACROSS_CUES`).
  It warns about digits, symbols, unit-like tokens and lone letters (`WRITTEN_FORM_TOKEN`), turns exact
  spans into word ranges, and adds `SEGMENT_TOO_LONG` as a warning only.
- `narration.lint`: the positive-only check of voice descriptions. It warns about negated qualities
  ("not rough") and suggests a positive rephrasing; it never refuses.
- The model-worker runtime (`narration_worker`): the JSON-lines worker protocol's framing and request
  loop, the determinism switches, the CPU thread cap and the `hello` fingerprint, started as
  `python -m narration_worker --role <role> --store <store_root>`; and the daemon's worker client
  (`narration.workers`), which reports a crashed worker at once and stops one that does not reply in time.
  A worker whose environment is missing or broken is reported as `BACKEND_NOT_INSTALLED`, with what to do
  next (`narration-admin install`).
- A `fake` worker role with deterministic synthetic audio and QA outputs, and faults it can plant on
  request of a test (`NARRATION_FAKE_SPEC`), so the service can be developed and tested without a model
  or a GPU; and the worker contract tests every worker runs (`narration_worker.testing`). The fake checks
  a Qwen `load`'s `device`, `dtype`, `attn_implementation`, `determinism` and `settings` with the Qwen
  worker's own checks (`narration_worker.qwen_settings`), and its `model` reference by the same rules as
  that worker, so it refuses them exactly as the real worker does, with the same code and field: all ten
  sampling values and `non_streaming_mode` must be
  given, the ceiling `settings.generation.max_new_tokens` among them. It does not read the snapshot's
  files or look for torch and a GPU.
- `narration.keys`: the cache keys of design section 10.2 (`voice_hash`, measurement, render, delivery and
  analysis keys) over RFC 8785 canonical JSON, the seed of section 10.3, the `rn_`/`tk_`/`an_` ids, and
  ULID job and design ids. Key values are pinned by golden tests: they name every take and never change.
  The delivery key names the post-processing rules' version (`narration.post/1`). Every id and key is
  checked as a whole string, so one with a trailing newline is refused.
- `narration.store`: the local store of design section 15. SQLite in WAL mode with a recorded schema
  version, content-addressed folders published whole (staged, then renamed), read-only immutable files,
  leases so one process produces each key, idempotent job creation and a job queue, `gc` (a dry run unless
  asked) and `verify`. Every path is confined to the store root, refusing `..`, absolute paths, links and
  junctions that lead out, and reserved names.
  - A job with a reused `idempotency_key` and a different request is refused (`INVALID_ARGUMENT` on
    `idempotency_key`).
  - `gc` never removes a key published again while it runs. It lists what it cannot remove and carries
    on. It keeps the takes a live measurement names, for `measurement_retention_days`, and the voice
    profiles of a live design.
  - A failed publish puts the caller's audio back in `scratch/` and keeps the folder it would have
    replaced. The store moves in files only from `scratch/`.
  - The current alignment benchmark is the configured aligner's.
  - `gc` collects in short write transactions (at most 200 items, or about 2 seconds, each), so other
    writers never wait on it for long. It tries a folder in use once and lists it instead of waiting.
  - A publish whose database COMMIT fails is undone, and the caller's audio (a render, a take, a designed
    clip or the canary clip) goes back to `scratch/`. A publish that has committed never reports a failure
    because it could not tidy up afterwards.
  - A `.trash-` name records when it was made (`.trash-<UTC epoch seconds>-<token>-<name>`), and `gc`
    judges its age by that time rather than by the modification time, which a renamed file keeps. So `gc`
    never collects the old copy a publish has just set aside and may still need to put back. For the same
    reason, a file the store moves in from `scratch/` takes the time of the move as its modification time,
    so a canary clip that waited there for days is never collected while its publish waits for the lock.
  - Reading the daemon's status (`run/daemon.json`) while the daemon rewrites it no longer fails on
    Windows. A read refused during the rename is tried again for up to about a second. A store file whose
    `realpath` comes back with a `\\?\` prefix, because the file was replaced during the call, is no longer
    refused as outside the store: `narration.platform.real_path` drops that prefix, and both the store's
    path check and the platform's compare paths through it.
  - A reader never reports a measurement or a profile missing while a publish replaces it: it reads again
    once the publish has finished. A new measurement is no longer lost when `gc` collects the one it
    replaces at the same moment. `gc` bounds each transaction by the items it tries, so folders it cannot
    rename no longer keep one transaction open.
- `narration.post`: delivery post-processing (design section 13). A raw take becomes a 48 kHz PCM_24 mono
  WAV through the relative trim, a pinned resampler, static gain to -16 LUFS (BS.1770-4) with the
  -1.0 dBTP true-peak ceiling winning, and 10 ms fades. Each delivery reports its trim and loudness
  records, the `LOUDNESS_UNDER_TARGET` and `GAIN_HIGH` flags, the tools that enter the delivery key, and
  the signal statistics QA needs. The same raw take always gives the same bytes.
- `narration.post`: delivery post-processing (design section 13, revision 5.4). A raw take becomes a
  48 kHz PCM_24 mono WAV in five steps:
  - a relative trim that ignores a DC offset and never sets its threshold below -70 dBFS;
  - a pinned resampler;
  - static gain to -23 LUFS by default (BS.1770-4, whole blocks);
  - 10 ms fades;
  - the -1.0 dBTP true-peak ceiling on the final file, which wins over the target.

  Each delivery reports its trim and loudness records (loudness is null for a silent take), the
  `LOUDNESS_UNDER_TARGET` and `GAIN_HIGH` flags, the tools and rules version that enter the delivery key,
  and the signal statistics QA needs. The WAV bytes are written by the service itself. On one machine,
  with the same pinned versions, the same raw take gives the same bytes.
- `narration.mcp`: the MCP front-end, built over a backend interface (plan.md WP17). It publishes the
  eleven v1 tools in a fixed order with dereferenced JSON Schemas and descriptions that state the caller's
  rules (no caller state kept, length is the caller's decision, a respelling is a hint, the backoff rule,
  the retention periods). It validates arguments inside the handler, so a bad argument is a tool error
  with `field` and `hint` (`LIMIT_EXCEEDED` when a request is over a size bound), and every failure inside
  a call is a tool error too. A cancelled write still finishes its work, for up to 30 s; only its reply is
  dropped. Results carry a text copy and `file:///` links. It serves the `narration://` resources with
  `ttlMs` and `cacheScope`, accepting a URI as written or as RFC 6570 expands it and refusing a malformed
  id before any lookup, and the four prompts, in both the 2026-07-28 protocol and the legacy `initialize`
  handshake. The `narration-mcp` command is connected to the real backend in a later work package (WP36).
- `narration.qa`: take QA as plain Python (design sections 8, 11.1, 11.3, 12): `wer_raw` and `wer_adj`
  with the word-count rule, exact spans read by the number reader `@2` (Whisper's English normaliser,
  vendored from openai/whisper, MIT, see `THIRD_PARTY_NOTICES`; read in phrases split at punctuation, with
  the "nought" rule and curly apostrophes read as straight ones), hinted-term checks, head and end
  insertions with reference bleed, speaker similarity and pace against the voice's measurement, signal
  checks, the verdict and retake triggers, the suggestion tiers, the per-job consistency report,
  `listen_first`, fit reporting, and the job report in Markdown and JSON.
- The Qwen3-TTS worker (`workers/qwen3tts`, role `qwen3`): loads Base or VoiceDesign offline from a
  snapshot folder named by its revision, prepares ICL voice prompts, and renders with every audio-changing
  setting passed explicitly (never a library default). Each `synthesize` or `design` call takes its own
  required `max_new_tokens` (at most the loaded ceiling, 8192) and the reply echoes it. Each render is
  seeded, reports how generation stopped (`new_tokens`, `hit_token_cap`), and is written as a float32 WAV
  that repeats byte for byte. A snapshot of another model is refused, and a broken one is reported as not
  installed.
- The QA worker (`workers/qa`, role `qa`): loads Whisper-large-v3, WavLM-base-plus-sv and the wav2vec2
  aligner offline from snapshot folders named by their revisions, and refuses a load whose snapshot is
  missing, damaged, misnamed or of another model (`BACKEND_NOT_INSTALLED` or `INVALID_REQUEST`, with the
  field). Its ops:
  - `transcribe`: English (the language may be named, as in "English"; an unknown one is refused), word
    times, sequential long-form, decoded with five beams and not conditioned on the previous window, as
    the bake-off's evidence was (ADR 0004);
  - `embed`: the L2-normalised speaker embedding, on the GPU the group was loaded on or on the CPU (the
    canary's path). Audio longer than 60 s is embedded as equal windows of at most 60 s and their
    embeddings averaged, which bounds its memory; audio of up to 60 s is embedded in one pass, as before;
  - `f0` and `profile`: pitch by pYIN, and the voice measurements (speaking rate, pause ratio, loudness,
    spectral centroid, HNR, CPPS) with a spectrogram and pitch picture, all without GPL code;
  - `align`: the forced alignment on the CPU.

  After each op on the GPU it returns the memory PyTorch keeps cached, so between ops it holds only its
  models on a shared GPU.
- `narration-admin`, the operator CLI: `narration-admin [--config <path>] <command>`, finding the
  configuration by the same rule as `narration-mcp`. `--help` lists every command of design section 7.1;
  a command whose module is not in this build says so (exit code 3) instead of failing, and the others
  still run.
- `narration-admin daemon start | stop [--now] | status`. `start` starts the daemon detached unless one
  serves, and waits until it serves; when the terminal cannot detach it, `start --foreground` runs it in
  the terminal. `stop` posts a stop only when a daemon runs, then waits for its answer. `status` shows what
  the daemon is doing, or `--json`.
- `narration-admin doctor [--quick] [--json]`: checks the platform, the configuration, that the store
  root is writable with enough free disk, that every pinned model is installed with the files its install
  recorded (hashed again unless `--quick`), that each worker venv matches its `uv.lock`, the NVIDIA GPU
  (a machine without one is told plainly that it cannot run jobs), and the engine pin. Each problem says
  what to do next; the exit code is 1 when a check fails.
- `narration-admin gc [--apply] [--json]`: a dry run by default that lists what retention no longer keeps;
  `--apply` removes it. `narration-admin verify [--json]`: hashes the store's immutable files and the
  installed model files again, and runs the database's integrity check; it changes nothing.
- `narration-admin install [--dry-run] [--from-cache <dir>] [--models-only | --workers-only]`: puts each
  pinned model in its snapshot folder, every file checked against the hash Hugging Face publishes for its
  revision before it is renamed into place, and records them in `<models_root>/manifest.json`; syncs each
  worker venv from its `uv.lock` when it is out of date; and asks a running daemon to stop when it
  repaired something the daemon uses. Certificates are always verified (`SSL_CERT_FILE`,
  `REQUESTS_CA_BUNDLE` and `UV_NATIVE_TLS=1` are honoured; design section 17.8), an `HF_ENDPOINT` mirror
  must be `https`, and no redirect leaves `https`. A listed file name that is not a plain path inside the
  model's folder is refused. Files are fetched into `<models_root>/.staging/`, never into a snapshot
  folder.

### Changed

- `narration.example.toml` works as it is: its paths are relative (`store`, `models`, `workers/qwen3tts`,
  `workers/qa`), resolved against the configuration file's folder, where they used to be `<service_root>\…`
  placeholders that had to be edited by hand. A configuration path that still holds such a placeholder is
  now refused with a hint, instead of making a folder named `<service_root>`. `store/` and `models/` at the
  checkout's root are gitignored. The README's install clones the release tag, not `main`, and its disk
  estimate now counts the worker environments (about 25 GB in all).
- `narration-admin install`, when it asks a running daemon to stop after a repair, now says what the stop
  means: jobs queued before it wait for the next start (`narration-admin daemon start`, or the next
  `submit_job` while `[daemon] autostart` is on), and that daemon runs the whole queue. It said "The next
  job starts a fresh daemon", which read as if a queued job would.
- Each take's QA reports its own speaking share, its speaking time over its voiced span, beside `pause_s`:
  in `qa.metrics.speaking_share` (contracts 1.6.10), in the pace flag's `details`, and as a column of
  `narration-admin failures --export`'s `index.csv`. It is information only: nothing judges it. It is null
  when the take's silences were not measured, and in a take scored before.
- Pace is now measured in spoken characters per second of speaking time: the voiced span with its pauses
  (silences of at least 13 frames of 20 ms, 0.26 s) taken out (DC-18; the QA profile is now `default.v5`).
  Words per minute followed the corpus's word lengths rather than the voice, and a one-sentence segment, which
  has no pause between sentences, read as fast against a curve measured on paragraphs. The voice's
  measurement uses the same rule, so its pace curve and tolerance are in these units too, and `measurement.json`
  is now `narration.measurement/v2`, with the voice's pace `level_cps` and a `speaking_share` used for duration
  estimates. Pace is judged against the voice's flat level (DC-20): every ladder rung against the band's
  level, and a take between the curve's points or held flat at its ends; the fitted trend is kept for
  information only. **A voice measured before must be measured again**: it answers `VOICE_NOT_MEASURED` until
  it is. Measuring again reuses every render and take the earlier measurement made. Only QA runs again, plus
  renders for any rung beyond where the earlier ladder stopped, if the new rule lets it go further.
  `get_results` shows `qa.pace.articulation_cps` and `qa.pace_expected.articulation_cps` beside `spoken_wpm`,
  which stays as information; the report of a job finished before keeps its pace in words per minute.
  `narration-admin failures` shows the pace QA judged. `PACE_FAST` still only warns.
- `get_job` and `cancel_job` notice a job whose daemon has gone (a crash, a restart, a daemon that ended
  with its client), which before read `running` or `cancelling` forever. When no daemon runs, they start one
  (as `submit_job` does, following `[daemon] autostart`); its start-up puts a job left `running` back on the
  queue, reusing what it finished from the cache, and finishes a cancel. `get_job`'s `message` then says it
  asked for a daemon and what happens next, or with `[daemon] autostart` off, who must start one. When no
  daemon can be started, `get_job` answers `DAEMON_UNAVAILABLE` (retryable) with the job and the hint to run
  `narration-admin daemon start`. A queued job is left alone for 30 s after it was written, while the daemon
  its submission asked for starts up. A queued job left in the queue by a stop posted after it was queued
  (`narration-admin daemon stop`, or `narration-admin install`) starts no daemon: `get_job` says it runs on the
  next start (`narration-admin daemon start`, or the next `submit_job`). A daemon that stopped because it
  failed, or a stop asked before the job was queued, does not hold the job back.
- A daemon launched by `get_job`, `cancel_job` or `submit_job` is recorded in the store (`run/launch.json`: when
  it was launched, and its pid), once the platform has let it run; a refused start records nothing. While that
  daemon is starting (90 s at most, and until it writes its status), `get_job` and `cancel_job` ask for no
  other, so repeated polls during a start launch one daemon, and a start that hangs costs at most one launch
  every 90 s. A launched daemon that exits before it serves is a failed start: until the 90 s have passed,
  `get_job` answers `DAEMON_UNAVAILABLE` (retryable) with the daemon's log in `details.log` and the hint to run
  `narration-admin daemon start` in a terminal, instead of launching another at once.
- `get_job` states the backoff rule in its description, as the tools that write do: it answers
  `DAEMON_UNAVAILABLE` (retryable, with `retry_after_s`) when no daemon serves an active job and none can be
  started. Its description also says that if the daemon died, `get_job` asks for one again (with
  `[daemon] autostart` on), and that its message says what was done. It stays read-only.
- `submit_job`'s plan (and its `dry_run`) now counts cached analyses: `narration-mcp` and `narration-admin
  render` look the analysis layer up with the installation's QA and aligner pins, as the daemon keys it. A
  resubmission that is all cached reports `segments_cached` and no analyses to do, instead of 0 and every
  analysis, and its `est_wall_s` no longer adds a QA model load. `get_server_status`'s `alignment` names the
  configured aligner's method and, before the alignment benchmark has run, its pinned revision (it was `null`).
- `submit_job`'s `VOICE_NOT_MEASURED` says when the clip is measured under a transcript that differs from the
  one sent only in whitespace or quotes, dashes and ellipses (a trailing newline, a double space, curly quotes
  for straight ones): `field` is `voice.transcript`, and `details.transcript_mismatch` gives the index of the
  first differing character and each side's character there, and `rewrites` names the changes that give the
  measured transcript (`trim_edges`, `collapse_whitespace`, `add_trailing_newline`, `plain_punctuation`,
  `typographic_quotes`, or two of them). The hint gives those changes as steps to take, and says not to
  measure again. Otherwise the hint also says to send a clip's transcript exactly as it was measured. Neither
  transcript is quoted in the error. Caught: whitespace the transcript sent has and the measured one has not,
  typographic quotes, dashes and ellipses sent for plain ones, straight quotes sent for typographic ones, and
  a trailing newline the measured transcript had. Not caught, so still only the general hint: other whitespace
  the measured transcript had (a double space, a leading space), and a typographic dash or ellipsis it had
  where the one sent has a plain one.
- `narration-admin doctor` reports the QA profile the service scores with, and warns when `[qa] profile` in
  the configuration names another: the setting changes nothing, since every take is scored with the
  build's profile. The warning says which profile runs and to update the line; it never fails the check.
  The daemon logs the same warning once when it starts.
- The daemon logs one INFO line when a job it ran ends (completed, failed or cancelled): its id, kind,
  final status and outcome, the number of segments, the retakes used, and the wall time since it took the
  job. A job cancelled while still queued never reached the daemon, so it gets no line. Nothing of the
  request is logged: no text, no transcript, no path.
- `DAEMON_UNAVAILABLE`'s `retry_after_s` is 60 s at every level when the daemon could not be detached (it
  was 30 s at the MCP tool level and 60 s in the platform's own error): the fix needs a person to run
  `narration-admin daemon start`, and a retry sooner than that fails the same way.
- `PACE_FAST` only warns; it never fails a take and never triggers a retake (the QA profile is now
  `default.v4`, DC-19). The words-per-minute pace model failed short paragraphs that listened fine, and
  wasted their retakes. `PACE_SLOW` is unchanged (it only warns). The flag's `details.fast_fail_above` is
  null. A voice measured before stays measured, and cached renders and takes are reused: only their QA
  runs again, on the next request that asks for them.
- The MCP server now tells the calling agent what decides whether real use goes well. Its instructions
  and the `submit_job` and `check_text` descriptions say to send every invented or unusual name as a hint:
  the term alone is enough, and a name without one is scored as misheard words that can fail a take
  (`WER_HIGH`). They also say to keep a job to a scene (about 8 to 10 segments), to call `get_results` with
  `include_words: false` unless word times are needed, and to use each segment's `suggested_take_id`. They
  point to `narration://jobs/{job_id}/report` and name `submit_job`'s options (`dry_run`, `strict_text`,
  `takes`, `max_retakes`, `priority`). A suggestion of tier 4 is a failed take, to resolve or redo before
  keeping it. The voice's transcript must be copied, never retyped. Every tool parameter now has a
  description. The instructions and every tool description fit in 2048 characters, where Claude Code cuts
  server instructions. `design_voice`, `profile_voice` and `audition_pronunciation` are marked "not in this
  build yet" wherever they are advertised, until their handlers land; the prompts show how to hear a
  respelling with `submit_job` meanwhile. `VOICE_NOT_SYNTHETIC`'s hint says that only a person allows a
  clip, with `narration-admin voices allow`, and that the daemon is restarted first, then the client
  reconnected. An upper-case `sha256` is told to lower-case it, and a top-level option such as `takes` is
  pointed to `options.takes`; a field sent inside `controls` is told to leave `controls` out. The README
  starts the server with the environment's Python (`-m narration.mcp`) and says to start the daemon from
  a terminal when the client cannot. Nothing that enters a cache key changed, so no voice needs
  measuring again.
- The service's spoken material is frozen as version 1: the calibration corpus `narration-en.v1`, the
  alignment benchmark `alignment-en.v1`, the canary `canary.v1` and the demo script `demo-en.v1`. The
  corpus now carries the calibration's design text, which `measure_voice` renders first. A frozen set
  never changes; a new text is a new set. A voice measured on the draft corpus is measured again by
  `measure_voice`; its old measurement still serves generation.
- Contracts 1.6.8: a design candidate carries its `flags` in `candidate.json`, so the design resource
  shows them and they survive a crash or a restart of the job. A new fail flag, `CLIP_TOO_LONG`, marks a
  candidate longer than `[limits] max_clip_seconds`, which `measure_voice` would refuse. The design
  seed's scheme id is now in the shared names. (Its first commit calls it 1.6.7; it became 1.6.8 when
  the pace fix above was released as 1.6.7 first.)
- An engine profile's `vram_need_mb` is recorded but no longer part of its hash (DC-16): it only tells the
  GPU scheduler how much free memory to wait for, so a refined estimate keeps every cached take and
  voice measurement.
- A worker that runs out of GPU memory while CUDA creates its context or cuBLAS its handle ("CUDA error:
  out of memory", `CUBLAS_STATUS_ALLOC_FAILED`) now reports `GPU_OOM`, as for any other allocation, so the
  daemon unloads, waits and retries once instead of failing with `INTERNAL`.

- The daemon opens its store with the configured aligner's method id, so every take's alignment reports the
  measured error of that aligner's published benchmark (design section 11.2); an aligner the service cannot
  run is logged at start, and each job then fails with `BACKEND_NOT_INSTALLED`.
- `narration-admin doctor` also checks that the cue aligner can run `[alignment]` as configured, as the job
  engine does: a pinned model that is not a CTC model, or a device other than the CPU, fails the models
  check and says what to set, instead of surfacing only when a job fails.
- The default design text (`[voice_design] design_text`, also the canary's design text in
  `material/canary/canary.v1`) is a new text written for the service: "Good bread asks for patience: the
  dough is mixed, folded and left to rise through the morning. When the loaves come out golden and crisp,
  a gentle warmth fills the whole kitchen." A clip keeps the transcript `design_voice` returned for it.
- `narration.align`: cue alignment (design section 11.2). It spells each cue's spoken text in the
  aligner's alphabet (hinted terms by `align_as`, whatever the hints' order), and aligns each run of words
  it cannot spell (digits, symbols) as one wildcard token, so the speech they stand for no longer pulls
  the words around them away. It refuses audio too short for its text with `ALIGNMENT_ERROR` instead of
  crashing, snaps cue boundaries into pauses (`CUE_BOUNDARY_NO_PAUSE` when there is none), scores each
  cue's confidence (`CUE_LOW_CONFIDENCE`, below `[alignment] low_confidence_below`), and leaves a cue it
  cannot place without times (`CUE_UNALIGNED`, never interpolated), saying why in `details.reason`:
  `no_alignable_words` (the cue's text has no word to place; not a retake trigger), `low_confidence`
  (below `unplaced_below`) or `alignment_error`; QA adds `not_placed` for a cue it finds without times that
  the aligner did not flag, so every `CUE_UNALIGNED` says why. A cue next to one it cannot place never
  takes that cue's speech: when no pause separates them, its edge keeps the aligner's time
  (`CUE_BOUNDARY_NO_PAUSE` with `details.edge`). It cross-checks boundaries against Whisper's words
  (`CUE_ALIGNMENT_DISAGREE`, `max_disagreement_s`). The QA worker's `align` op computes the wav2vec2 CTC
  emissions on the CPU and runs the forced alignment; a worker failure other than `ALIGNMENT_ERROR` is not
  held against the take. A missing, damaged or foreign aligner snapshot is reported as not installed
  (`BACKEND_NOT_INSTALLED`).
  (below `unplaced_below`) or `alignment_error`. A cue next to one it cannot place never takes that cue's
  speech: when no pause separates them, its edge keeps the aligner's time (`CUE_BOUNDARY_NO_PAUSE` with
  `details.edge`). It cross-checks boundaries against Whisper's words (`CUE_ALIGNMENT_DISAGREE`,
  `max_disagreement_s`). The QA worker's `align` op computes the wav2vec2 CTC emissions on the CPU and runs
  the forced alignment; a worker failure other than `ALIGNMENT_ERROR` is not held against the take. A
  missing, damaged or foreign aligner snapshot is reported as not installed (`BACKEND_NOT_INSTALLED`).
- `narration.platform.testing.StandInPlatform`: a `Platform` for tests that runs on any OS. It starts
  no process, answers the path checks with the real platform's text rules, and reports a free-disk
  figure the test sets.
- `narration.daemon`: the background process that owns the GPU and the queue (design section 4), started
  as `python -m narration.daemon --store <store_root> --config <path>`. One daemon runs per store; a
  second one exits quietly. Started from an MCP session, it is detached from the session and keeps
  running after the session ends; when the session's Job Object forbids that, it is not started, and the
  caller gets `DAEMON_UNAVAILABLE`. It starts its model workers at below-normal priority, in a group
  that Windows ends with the daemon, even when the daemon is killed, and starts a crashed worker again,
  but never in a loop. It writes `run/daemon.json` on every change, unloads an idle model after
  `idle_unload_s` and exits after `idle_exit_min` without work. It answers `release_gpu`, `stop`
  (finish the segment in flight, then exit) and `stop_now` (queue the job again and exit, leaving no
  partial file). A daemon honours every `stop` posted after it was launched: one it had not yet read
  when it took over from a daemon that was exiting, and one that daemon had already answered. A stop
  posted while no daemon ran is answered `stopped: false` by the next one, which keeps serving. A daemon that died leaves its jobs running;
  the next daemon puts them back in the queue, and a `run/daemon.json` left unreadable by a crash is
  taken as a daemon that died. A job queued just as the daemon turns to exit for want of work is still
  run. The daemon and its workers start with `-P` and without `PYTHONPATH`, `PYTHONHOME` or
  `PYTHONSTARTUP`, so nothing in their working folder or the caller's environment is imported in their
  place; a daemon started by hand without `-P` logs a warning saying so. Its workers start with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so on a machine with two GPUs `[gpu] device` names the same GPU for the
  free-memory check and for the models.

### Fixed

- Contracts 1.6.11: `run/daemon.json` now remembers why the daemon stopped. A `stopped` status keeps the
  daemon's start time and adds `stop_reason` (`operator`, `idle`, `interrupted` or `error`), and
  `narration-admin daemon status` shows it. So `get_job` holds a job queued before a `daemon stop` only when
  that stop actually ended the daemon: after an idle exit or a failure, even one within 30 s of a stop, the
  job gets a daemon. And a daemon that started, served and then failed within 90 s of its launch is no longer
  reported as one that "exited before it served": `get_job` asks for a new daemon at once. A `daemon.json`
  written by an older daemon still loads, and keeps the old rules. A job runner thread that fails is now
  logged in the daemon's log.
- A wrong sha256 sent for a clip or for `profile_voice`'s audio no longer gives back the sha256 of a file that
  is not a WAV: the WAV check now comes first, so `VOICE_FILE_MISMATCH`'s `details.actual` is only ever a WAV's
  hash, and a job that finds its clip changed no longer reports the new file's hash (security review S1).
- `narration-admin install` and `doctor` no longer run a `uv` found in the working folder on Windows; they
  look only in the folders on `PATH` (security review S2).
- `SECURITY.md` now matches the code: what a caller learns about a file it names, the operator commands that
  write outside the store, the package downloads of the install step, the files the operator must keep
  private, and the control tokens not yet refused.
- Contracts 1.6.10: `CLIP_TOO_LONG`'s retake rule is `never`, as design section 14 gives it (it said
  `always`): a designed candidate has no take slot to retake. `is_retake_trigger` now follows the table's
  retake column for a fail too, so a fail whose rule is `never` is not a trigger; every other fail still
  is, and no job's retakes change.
- A worker that cannot start (a missing or broken venv) could fail its job with `INTERNAL` ("the worker was
  closed") instead of `BACKEND_NOT_INSTALLED`, and be tried again. The daemon's check for dead workers could
  see the starting worker's process gone before the start had read its exit code, and closed it, so the
  start read as a crash. A worker still starting is now left to its start.
- A daemon started from an MCP session could die with the session, without a stop, on Windows. When the
  client runs the server in a Job Object that forbids breakaway (the MCP Python SDK does) and the server is
  the venv's `python.exe`, a launcher that puts the interpreter in a nested job of its own, Windows accepted
  the daemon's breakaway from the inner job but left it in the client's; the client's exit then killed it.
  The daemon is now created suspended and runs only once Windows confirms it is in no Job Object at all.
  One left in a job is ended before it runs, and the caller gets `DAEMON_UNAVAILABLE` (retryable; its
  `details.reason` is `left_in_job`, `breakaway_refused` or `job_check_failed`) with the hint to run
  `narration-admin daemon start` in a terminal. The rule is deliberate: any enclosing job that forbids
  breakaway refuses the detached start, even one that would not end the daemon, since the service cannot
  read such a job's limits or know who will close it. A CI runner is the known case (measured on GitHub's
  hosted Windows runner, whose job forbids breakaway); there, and on any host like it, set `[daemon]
  autostart = false` so that no client tries to start the daemon, and run `narration-admin daemon start
  --foreground` as its own process; or use a host whose jobs allow breakaway.
  Under clients that spawn through libuv, whose job allows breakaway, the daemon leaves every job and keeps
  running (Node.js, measured with Node v22; Claude Code, we believe, since it spawns through libuv too;
  spike k). `narration-admin daemon start` now names only the cause Windows established, and says when the
  daemon exits for want of work (`[daemon] idle_exit_min`), so a client that cannot start the daemon itself
  knows to start it again. Its refusal names the route above, and the platform's messages it quotes speak
  of "this process", not "this client", so they read right in a terminal and in an MCP client alike.
