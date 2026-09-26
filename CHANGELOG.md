# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until the first `0.1.0` release, changes
are tracked here but no version is tagged; nothing described below is installable yet.

## [Unreleased]

### Added

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
  a Qwen `load` as the Qwen worker does, its ceiling `settings.generation.max_new_tokens` included, so a
  load the real worker would refuse fails in the tests too.
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

### Changed

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
  (`CUE_BOUNDARY_NO_PAUSE` with `details.edge`). It cross-checks boundaries against Whisper's words (`CUE_ALIGNMENT_DISAGREE`,
  `max_disagreement_s`). The QA worker's `align` op computes the wav2vec2 CTC emissions on the CPU and runs
  the forced alignment; a worker failure other than `ALIGNMENT_ERROR` is not held against the take. A
  missing, damaged or foreign aligner snapshot is reported as not installed (`BACKEND_NOT_INSTALLED`).
