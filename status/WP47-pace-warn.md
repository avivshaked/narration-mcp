# WP47-pace-warn PACE_FAST warns only (hotfix; the owner's decision, 2026-09-27)
State: blocked:paused-by-owner        Updated: 2026-09-27
## Done
- `QaProfile.pace_fail_tol_factor` is `float | None = None`: `default.v4` has no pace fail.
- `pace_flags`: with no factor, `PACE_FAST` only warns; `details.fast_fail_above` is null. A profile with a
  factor still fails above curve x (1 + f tol). `PACE_SLOW` unchanged (warn only).
- `names.QA_PROFILE = "default.v4"` (contracts 1.6.7, lead-approved; WP34's branch also claims 1.6.7 and is
  renumbered later). Docstrings (profile, checks, scorer, interfaces), `narration.example.toml`, CHANGELOG.
- Tests: pace edges under v4; +50% take warns (verdict warn, no retake); slow unchanged; a profile with a
  factor still fails; QA_PROFILE pinned; render key, seed and measurement key unchanged by the profile;
  a job whose take is far above the curve is warned and never retaken. Two store tests used the literal
  "default.v4" as "another profile"; they now use an invented name.
- Item 5 (a measured voice stays measured), checked in code:
  - the measurement key does not include the QA profile (`keys/__init__.py` `measurement_key_object`);
  - `require_measured` and `JobEngine.open` look a measurement up by (voice_hash, engine_profile_id) only;
  - `current_measurement` compares only the measurement key; `MeasurementRecord` has no QA field;
  - no code compares `versions.qa_profile`; `config.qa.profile` is read by nothing in `src/`;
  - the engine profile does not include the QA profile. Nothing forces a re-measure.
## Tests
`uv run python -m pytest tests/qa tests/keys tests/contracts tests/jobs tests/measure tests/backend tests/store
tests/material tests/mcp tests/engine tests/lint` → 1980 passed, 2 failed (the two store tests, since fixed);
then `tests/store tests/jobs/test_engine.py` → 262 passed.
## Decisions made (and why)
- `fast_fail_above` is null (not omitted), so a reader sees there is no fail line.
- `codes.FLAGS[PACE_FAST]` keeps severities (warn, fail): the code table lists what a profile may use, as
  for `PACE_SLOW`, which no profile fails.
## Contract change requests
- `names.QA_PROFILE` = "default.v4" (contracts 1.6.7), approved by the lead.
## Proposed design change DC-19 (for plan.md section 1.5; design section 11.1 table)
- PACE_FAST: warn outside curve × (1 ± tol); no fail (owner's decision 2026-09-27, until WP47's rate model is
  validated).
## Dependency requests
- none
## Questions for the lead / owner
- plan.md's WP47 says "with the same warn and fail factors"; this hotfix removes the fail. Reconcile when
  WP47 lands.
## Next
- ruff check, ruff format --check, basedpyright on the changed files; tools/check_tracked.py.
- Final commit (not wip), push, PR `fix(qa): PACE_FAST warns only, never fails (the owner's decision;
  contracts 1.6.7)`.
