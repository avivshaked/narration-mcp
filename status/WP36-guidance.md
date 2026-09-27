# WP36 The caller's guidance in the tool texts (readiness audit item 1.1)
State: blocked:paused-by-owner        Updated: 2026-09-27T22:13+01:00

Paused by the owner, with PR #40's last round finished: all six items are done and pushed (0872b72).
Nothing is left but CI and the lead's merge; the full suite is left to CI.
Branch `wp/36-guidance` (PR #40), on `main` at 4e0d85f (WP45 merged). `main` has since gained two
docs-only commits (2a069e0, 8aa11ec); I did not rebase again, because a rebase would need a force push.
Text only, except for the hint selection in `mcp/validation.py`. The audit ids it answers: mcp-2, qa-4,
cf-7, cf-2, mcp-1, rt-1, mcp-7, mcp-8, rt-6, cf-10, mcp-10, mcp-11, triage-9 and cf-11.

## Done
- **Names as hints (mcp-2, qa-4, cf-7).** The server instructions, the `submit_job` and `check_text`
  descriptions, the `narrate_script` prompt and the Hint schema say it. Every invented or unusual name goes in
  as a hint, and the term alone is enough. QA then scores the name as one word: a close spelling, or an
  `asr_alias`, costs nothing. A name without a hint is scored as misheard words, which can fail the take
  (`WER_HIGH`) and use up its retakes. Checked against design §9.1 (lines 1372-1374), §11.1 step 4,
  `qa/textmatch.py` (a found term costs nothing; one not found costs exactly one substitution), and
  `text/hints.py` (a term-only hint is applied and recorded but leaves the engine text alone).
- **Result size (cf-2, mcp-1, rt-1).**
  - The texts say: "Keep a job to a scene, about 8 to 10 segments"; "get_results with include_words false
    unless you need word times"; "use suggested_take_id: takes[] also lists failed and replaced attempts".
  - `narration://jobs/{job_id}/report` exists (`contracts/names.py`, `backend/resources.py`). Where it is
    offered as the smaller read (`get_results`, the resource's description), the text says what it holds:
    each segment's suggested take and every take's verdict and flags, but no file paths and no full cue
    timings (only the spans it lists under "Listen first").
    Those come from `get_results` or `narration://takes/{take_id}`.
- **Tier 4.** A suggestion of tier 4 is the best of the failed takes (`qa/suggest.py`). The
  `suggested_take_id` schema description, `get_results` and `narrate_script` step 7 say: every take failed QA
  (the job's outcome is then `needs_attention`); resolve its flags or redo the segment before keeping it.
  The instructions are left alone: only about 39 characters of room remain.
- **Options by their real names (rt-6, cf-10).** `dry_run`, `strict_text` (it refuses digits, symbols and
  unit-like tokens, and a hinted term split across cues), `takes`, `max_retakes`, and `priority`.
  - `interactive` is claimed first, and a running batch job gives way between pieces (`jobs/runner.py`,
    `jobs/admission.py`).
  - `dry_run` says exactly what the plan reports (`backend/planning.py`): the requested attempts only,
    with no retakes planned; `segments_cached`, the segments whose every attempt is already scored; the
    renders, deliveries and analyses needed; `est_audio_s` and `est_wall_s`. Analyses count as all needed
    when the QA pins are not known.
- **The voice object.** Send the path, sha256 (lower-case hex) and the transcript copied character for
  character. A transcript that differs by one character is another voice, with no measurement yet:
  `keys.voice_hash` hashes the transcript as sent, in NFC. A test checks this.
- **`measure_voice`** says that, unless the measurement comes back at once, it returns a job to follow
  with `get_job` and `wait_s` until it has completed, before `submit_job`.
- **Every input parameter has a description (mcp-8).**
  - A test walks every input schema.
  - `get_results` names its output fields. The cues description says an unplaceable cue has null times
    and `CUE_UNALIGNED`, never interpolated (§11.2).
  - `get_job` says to call again with `wait_s` until the job is completed, failed or cancelled.
- **Tools not in this build (mcp-7).**
  - `design_voice`, `profile_voice` and `audition_pronunciation` are marked, since this PR merges before
    WP34. The mark leads each tool's description and is named in the instructions.
  - While `design_voice` is marked, the instructions start the flow from an allowlisted clip, and the
    `design_narrator_voice` prompt carries the note.
  - The `resolve_flags` and `add_pronunciation` prompts show how to hear a respelling meanwhile: a short
    segment with that respelling as its hint, one respelling per request, then read `qa.terms`.
  - A test holds `NOT_IN_THIS_BUILD` **equal** to the tools whose job kind is not in `RUNNABLE_KINDS`. A
    handler merged without its text change fails, and so does the reverse.
  - The set is kept by hand rather than read from the backend. The texts are built without a backend (the
    front-end serves any `Backend`, whose protocol does not expose the runnable kinds). Importing
    `narration.backend.service` into `narration.mcp` would pull the job engine into the front-end. Reading
    it from the backend cleanly would mean threading the set through `build_front_end`, `build_tools`,
    `build_prompts` and the instructions: a bigger change than this text-only PR.
- **design_voice and profile_voice texts**, checked read-only against `worktrees/wp34-design`
  (`design/handler.py`, `design/profile.py`):
  - candidates go on the provenance list, so no allowlist edit is needed;
  - copy the chosen clip out of the store, since the store's copy expires with retention, and keep its
    sha256 and transcript (the tool text and the `design_narrator_voice` prompt);
  - a candidate flagged `WER_HIGH` fails its measurement (`REF_TEXT_MISMATCH`);
  - `profile_voice` has no speaking rate.
- **VOICE_NOT_SYNTHETIC's hint** (`contracts/codes.py`).
  - Only a person allows a clip, at this machine, after confirming that it is synthetic; a calling agent
    must not run the command. That is DC-17 and `admin/voices.py`.
  - The operator runs `<venv_python> -m narration.admin --config <service_root>/narration.toml voices allow
    <clip.wav>`, then stops the daemon first. With `[daemon] autostart` on, the next submission starts it
    again, else `daemon start`. Then the client reconnects (`/mcp` in Claude Code). That is WP45's
    `restart_advice` order.
- **mcp-10 hint polish** (`mcp/validation.py`):
  - a `sha256` that fails its pattern gets "64 lower-case hex digits; lower-case it";
  - a field sent one level too high is pointed to the object that takes it (a top-level `takes` goes to
    `options.takes`). A field is never pointed into `controls`, nor, when it was sent inside `controls`,
    further in (`controls.factor` is not pointed to `controls.pace`). A test covers both.
- **spec_revision (mcp-11, triage-9).** The `get_server_status` output schema and its description say that
  `spec_revision` is the MCP specification revision, not the design's. No field was added. The plan.md
  follow-up "get_server_status reports spec_revision …" can be closed.
- **README (cf-11).**
  - `.mcp.json` runs `<venv_python> -m narration.mcp --config <service_root>/narration.toml`, never the
    launcher executable; the environment comes from `uv sync --frozen`.
  - When a call answers `DAEMON_UNAVAILABLE`, start the daemon with `<venv_python> -m narration.admin
    --config … daemon start`. It detaches and outlives the terminal; only `--foreground` needs the terminal
    to stay open.
  - It covers `voices allow` (WP45), and stopping the daemon before reconnecting the client after a change.
  - Placeholders only.
- CHANGELOG line under Unreleased / Changed.
- **PR #40's last round.**
  - No error hint offers a tool this build cannot run. `VOICE_NOT_SYNTHETIC`, `REF_TEXT_MISMATCH` and the
    transcript check's own hint (`measure/transcript.py`), and the voice's `transcript` description, name
    `design_voice` only as "where design_voice is available", which stays true after WP34. A test scans
    `codes.ERRORS` and every literal `hint=` under `src/narration` for the tools in `NOT_IN_THIS_BUILD`.
  - A field sent inside `controls` is told to leave `controls` out, since every control is refused
    (`CONTROL_UNSUPPORTED`); it is no longer shown the refused fields as accepted. The test checks the
    whole hint.
  - The report resource's texts say "no file paths and no full cue timings": `report.md` lists the
    spans to listen to first, with their times, and no other cue times.
  - The README says the next job starts the daemon again only with `[daemon] autostart` on.
  - `dry_run` says that scored attempts are counted only when the service knows its QA pins; until
    PR #41, `backend_for` passes none, so `segments_cached` is 0 and every attempt counts in
    `analyses_needed`.

## Tests
- The full default suite ran once, early: `uv run python -m pytest` gave 3737 passed and 15 skipped
  (platform and Developer Mode skips, and one "engine.admin is in this build"), with 19 deselected. That was
  before the later edits. **While the owner narrates, the full suite is left to CI**, as the lead asked; I
  ran only targeted suites after it.
- On the final code (PR #40's last round), targeted only, while the owner narrates:
  - `uv run python -m pytest tests/mcp tests/contracts tests/measure/test_transcript.py
    tests/measure/test_handler.py`: 559 passed;
  - `ruff check` and `ruff format --check`: clean; `basedpyright` on the changed files (`codes.py`,
    `schemas.py`, `validation.py`, `descriptions.py`, `measure/transcript.py`, `test_descriptions.py`):
    0 errors;
  - the instructions are 1993 characters and `get_results`' description 2008, both within 2048.
- On the PR #40 review fixes before that:
  - `uv run python -m pytest tests/mcp tests/contracts`: 534 passed;
  - `uv run python -m pytest tests/backend/test_steps.py tests/admin/test_render.py`: 34 passed. These
    exercise the touched schemas and hint.
  - `ruff check` and `ruff format --check`: clean.
- `basedpyright`:
  - the whole tree ran at 21:21, before the review fixes: 0 errors;
  - on the changed files after the fixes (`descriptions.py`, `validation.py`, `schemas.py`, `codes.py`,
    `test_descriptions.py`): 0 errors.
- `py -3.12 tools/check_private.py --commits main..HEAD --base main`: exit 0. `py -3.12
  tools/check_tracked.py`: exit 0.

## Decisions made (and why)
- **Length limit, 2048 characters.**
  - KNOW: Claude Code 2.1.258 cuts server instructions longer than 2048 characters. Its binary has
    `kP=2048` and "Server instructions truncated from N to 2048 chars".
  - BELIEVE: it holds tool descriptions to the same length. Its telemetry slices them at that constant, but
    I did not find the cut itself.
  - The instructions are 1993 characters with the three tools marked, and 1924 with none. The longest tool
    description is `get_results`, at 2008. A test holds every text to `CLIENT_TEXT_LIMIT`.
  - So the instructions leave the three per-tool rules (§7.1) and the backoff rule to the tool
    descriptions, which all carry them.
- **The negation lint does not apply to these texts.** KNOW: it lints voice descriptions (§3.5). No test
  runs it on MCP texts, and it already finds 25 phrases in `main`'s tool texts.
- **No key is affected.**
  - KNOW, in a fresh interpreter: importing `narration.keys` loads neither `narration.mcp` nor
    `contracts.schemas`, directly or transitively. A test runs this in a subprocess.
  - `contracts.codes` *is* loaded, transitively, through `config` and `errors`. But nothing under
    `narration.keys` imports or names it, and a test checks the source.
  - The key functions hash only their arguments: requests, text and pins. `request_sha256` hashes the fixed
    string `narration.request/v1`, not a schema.
  - No schema default, bound or field changed, so no measurement or cached take is invalidated.
- **Contracts edits were text only**: descriptions in `contracts/schemas.py`, and one default hint in
  `contracts/codes.py`.
- **sha256 case: only the hint changed.** The validator does not lower-case the value itself; that is
  proposed below.
- **Boilerplate stays.** The design (§7.1) and a test require the three rules on every tool.

## Merge notes
- **A conflict with WP34 (PR #39) in `contracts/schemas.py`**, on `design_voice`'s `design_text` and `takes`
  (and nearby lines). My earlier note said there were no conflicts; that was wrong. WP34 adds a
  `design_text` description with a sentence about `CLIP_TOO_LONG`. This branch adds a different one, and
  descriptions for `takes`. `CLIP_TOO_LONG` is not on `main`, so it is not added here. WP34 resolves the
  conflict when it merges `main`.
- **WP34 must also take `design_voice` and `profile_voice` out of `NOT_IN_THIS_BUILD`.** The equality test
  fails until it does. WP35 does the same for `audition_pronunciation`.
- `wp/36-liveness` also edits `get_job`. My only `get_job` changes are its `ToolText` wording and its two
  parameter descriptions, so any conflict is textual.

## Contract change requests
- none made. Proposals, for the lead and owner:
  - `get_server_status`: add `design_revision` ("5.14") next to `spec_revision` (mcp-11).
  - `get_results`: a `segment_ids` filter and/or a compact view (per segment: the suggested take's path,
    sha256, verdict and cue times), and perhaps `include_words` defaulting to false (cf-2, mcp-1). This is a
    design §7.5 change.
  - `mcp/validation.py`: lower-case `voice.sha256` and `audio.sha256` before validating.

## Dependency requests
- none

## Questions for the lead / owner
- none

## Next
- None. The full suite is left to CI while the owner narrates.
