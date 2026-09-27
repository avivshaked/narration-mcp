# WP36 The caller's guidance in the tool texts (readiness audit item 1.1)
State: review        Updated: 2026-09-27T21:15+01:00

Branch `wp/36-guidance`, rebased onto `main` at 78b2fae. Text only, except for two hint selections in
`mcp/validation.py`. The audit ids it answers: mcp-2, qa-4, cf-7, cf-2, mcp-1, rt-1, mcp-7, mcp-8, rt-6,
cf-10, mcp-10, mcp-11, triage-9 and cf-11.

## Done
- **Names as hints (mcp-2, qa-4, cf-7).** The server instructions, the `submit_job` and `check_text`
  descriptions, the `narrate_script` prompt and the Hint schema say it. Every invented or unusual name goes in
  as a hint, and the term alone is enough. QA then scores the name as one word: a close spelling, or an
  `asr_alias`, costs nothing. A name without a hint is scored as misheard words, which can fail the take
  (`WER_HIGH`) and use up its retakes. Checked against design §9.1 (lines 1372-1374), §11.1 step 4,
  `qa/textmatch.py` (a found term costs nothing; one not found costs exactly one substitution), and
  `text/hints.py` (a term-only hint is applied and recorded but leaves the engine text alone).
- **Result size (cf-2, mcp-1, rt-1).** "Keep a job to a scene, about 8 to 10 segments", "get_results
  with include_words false unless you need word times", and "use suggested_take_id: takes[] also lists
  failed and replaced attempts". `narration://jobs/{job_id}/report` exists (`contracts/names.py`,
  `backend/resources.py`). It is named in the instructions, in `get_results` and in the prompts. So is
  `narration://takes/{take_id}`.
- **Options by their real names (rt-6, cf-10).** `dry_run`, `strict_text` (it refuses digits, symbols and
  unit-like tokens, and a hinted term split across cues), `takes`, `max_retakes`, and `priority`.
  `interactive` is claimed first, and a running batch job gives way between pieces (`jobs/runner.py`,
  `jobs/admission.py`). Each option has a schema description.
- **The voice object.** Send the path, sha256 (lower-case hex) and the transcript copied character for
  character. A transcript that differs by one character is another voice, with no measurement yet:
  `keys.voice_hash` hashes the transcript as sent, in NFC. A test checks this.
- **Every input parameter has a description (mcp-8).** A test walks every input schema. `get_results`
  names its output fields: `segments[].suggested_take_id` and `takes[]`, the take's `qa` and its separate
  `flags` (RETAKEN). `get_job` says to call again with `wait_s` until the job is completed, failed or
  cancelled.
- **Tools not in this build (mcp-7).** Following the lead's update, only `audition_pronunciation` is marked,
  because `design_voice` and `profile_voice` come with WP34.
  - The mark comes from one set, `descriptions.NOT_IN_THIS_BUILD`. It leads the tool's description and is
    named in the instructions.
  - The `resolve_flags` and `add_pronunciation` prompts show how to hear a respelling meanwhile: a short
    segment with that respelling as its hint, one respelling per request, since a request takes each term
    once. What the recogniser heard is in `qa.terms`.
  - A test fails if a marked tool's kind joins `RUNNABLE_KINDS`, so WP35 must drop the mark when it lands.
- **design_voice and profile_voice texts (lead's update).** I checked these read-only against
  `worktrees/wp34-design` (`design/handler.py`, `design/profile.py`).
  - Every design candidate goes on the provenance list, so it is accepted with no allowlist edit.
  - A candidate flagged `WER_HIGH` would get `REF_TEXT_MISMATCH` from `measure_voice`.
  - `profile_voice` has no speaking rate, because it is sent no transcript.
  - WP34 does not touch `descriptions.py` or `schemas.py`, so there are no conflicts.
- **VOICE_NOT_SYNTHETIC's hint.** It says to add the sha256 to `[voices] allow_sha256` and to stop the
  daemon (`narration-admin daemon stop`; `daemon start` or the next submission starts it again). It also
  says to reconnect the client (`/mcp` in Claude Code). This is right because both processes load the
  config once: `daemon/__main__.py:104` and `mcp/__main__.py`. Both check the allowlist:
  `backend/service.py` at submit, and `jobs/engine.py:151`.
- **mcp-10 hint polish** (`mcp/validation.py`):
  - A `sha256` that fails its pattern gets "64 lower-case hex digits; lower-case it".
  - A field sent one level too high is pointed to the object that takes it, for example `takes` to
    `options.takes`. The object is found from the schema. `controls` is never pointed to, since every
    control is refused.
- **spec_revision (mcp-11, triage-9).** The `get_server_status` output schema and its description say
  that `spec_revision` is the MCP specification revision the server is built on, not the design's revision.
  No field was added. The follow-up at plan.md "Follow-up: get_server_status reports spec_revision …" can
  be closed.
- **README (cf-11).** `.mcp.json` runs `<venv_python> -m narration.mcp --config <service_root>/narration.toml`,
  and never the launcher executable. The README also covers:
  - `uv sync --frozen`;
  - starting the daemon from a terminal (`<venv_python> -m narration.admin --config … daemon start`) when a
    call answers `DAEMON_UNAVAILABLE`;
  - restarting the daemon and reconnecting the client after editing `narration.toml`.
  It uses placeholders only.
- CHANGELOG line under Unreleased / Changed.

## Tests
- `uv run python -m pytest` (the full default suite, run once before the lead's CPU pause): 3737 passed,
  15 skipped (platform and Developer Mode skips, and one "engine.admin is in this build"), 19 deselected.
- After the last text edits, following the lead's instruction to run only single files until 21:20:
  - `tests/mcp/test_descriptions.py` (new): 30 passed;
  - `tests/mcp/test_front_end.py`: 170 passed.
- `ruff check`, `ruff format --check`: clean. `basedpyright` on the five changed Python files: 0 errors.
  I did not run it over the whole tree after the last edit, because of the pause. Run the whole tree once
  more after 21:20 before merging.
- `py -3.12 tools/check_private.py --commits main..HEAD --base main`: exit 0.
- `py -3.12 tools/check_tracked.py`: exit 0.

## Decisions made (and why)
- **Length limit, 2048 characters.** KNOW: the installed Claude Code (2.1.258; its binary has `kP=2048`
  and "Server instructions truncated from N to 2048 chars") cuts server instructions longer than 2048
  characters. BELIEVE: it holds tool descriptions to the same length. Its telemetry slices tool
  descriptions at the same constant, but I did not find the cut itself.
  - My first draft of the instructions was 2865 characters.
  - The instructions are now 2009 characters, and the longest tool description (`submit_job`) is 1871.
  - A test holds every text to `CLIENT_TEXT_LIMIT`.
  - So the instructions leave the three per-tool rules (§7.1) and the backoff rule to the tool
    descriptions, which all carry them. "The service keeps no caller state" stays.
- **The negation lint does not apply to these texts.** The brief said `narration.lint` runs on them. KNOW: no
  test runs it on MCP texts. It lints voice descriptions (§3.5), and it finds 25 phrases in main's own
  tool texts ("never a guarantee" and the like). I kept to the design's wording style instead.
- **No key is affected.** KNOW: nothing under `src/narration/keys/` imports `narration.mcp`,
  `contracts.schemas` or `contracts.codes`, and a test checks it. The keys hash requests, text and pins
  only. `request_sha256` hashes the fixed string `narration.request/v1`, not a schema. No schema default,
  bound or field changed. So no measurement or cached take is invalidated.
- **Contracts edits were text only.** Descriptions in `contracts/schemas.py` and one default hint in
  `contracts/codes.py`, as the brief asked. No field, default or bound changed.
- **sha256 case: only the hint changed.** The validator does not lower-case the value itself, because the
  brief was text only. That behaviour change is proposed below.
- **Boilerplate stays.** mcp-8 notes that every tool description repeats the three §7.1 rules. The design
  and a test require them on every tool, so they stay.

## Contract change requests
- none made. Proposals, for the lead and owner:
  - `get_server_status`: add `design_revision` ("5.14") next to `spec_revision` (mcp-11). Not done here.
  - `get_results`: a `segment_ids` filter and/or a compact view (per segment: the suggested take's path,
    sha256, verdict and cue times), and perhaps `include_words` defaulting to false (cf-2, mcp-1). This is
    a design §7.5 change.
  - `mcp/validation.py`: lower-case `voice.sha256` and `audio.sha256` before validating. A hash's case
    carries no meaning, and the request's identity would hash the lower-case form.

## Dependency requests
- none

## Questions for the lead / owner
- When WP35 lands, it removes `audition_pronunciation` from `NOT_IN_THIS_BUILD`. A test forces this.
- After WP34 merges, `test_only_a_tool_the_backend_cannot_run_is_marked_not_in_this_build` can be tightened
  from a subset to an equality check: on `main` today, `design` and `profile` are not in `RUNNABLE_KINDS`
  yet.
- `wp/36-liveness` also edits `get_job`. My only `get_job` changes are its `ToolText` wording and its two
  parameter descriptions, so a conflict, if any, is textual.

## Next
- None. Ready for review. Rerun the whole suite and `basedpyright` after the CPU pause, before merging.
