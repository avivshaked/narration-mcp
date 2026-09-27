# The service's own material

The text the service ships and uses on its own behalf: the corpus it measures voices with, the benchmark it
measures its cue alignment on, its canary, the demo script of its acceptance test, and the fixtures its text
and QA tests run on (design sections 3.2, 10.1, 11.2, 15 and 20). **None of it is a caller's text**: it was
written for the service, about invented places, people and things, in a neutral documentary register.

## The sets

| Set | Path | Used by | What it holds |
|---|---|---|---|
| `narration-en.v1` | `calibration/narration-en.v1/` | `measure_voice` (design 3.2), `engine bridge` (10.1) | the design text (the service's default, design 16 and plan.md DC-13), 3 calibration paragraphs (150–300 spoken characters) and 9 ladder paragraphs, one per rung (80, 150, 250, 300, 350, 400, 450, 500, 560), each within 5 % of its rung |
| `alignment-en.v1` | `alignment/alignment-en.v1/` | `bench alignment` (11.2) | 12 paragraphs, 31 cues; every cue boundary marked with the pause expected |
| `canary.v1` | `canary/canary.v1/` | `engine pin` and the canary gate (10.1, plan.md DC-3) | the positive-only voice description, design text and seed the canary clip is designed from, and the gate's fixed text and seed |
| `demo-en.v1` | `demo/demo-en.v1/` | the Phase 4 acceptance test (20) | 20 paragraphs of 1–8 cues, with 3 pronunciation hints and 5 exact spans |
| `text-v1` | `fixtures/text-v1/` | the text pipeline and lint tests (9.3, 3.5) | named cases for canonical form, the join, hints, text warnings, exact spans and refused markup; and lint cases |
| `qa-faults-v1` | `fixtures/qa-faults-v1/` | the Phase 3 acceptance test (20) | specs of planted faults and planted non-faults, each with the transcript the ASR is pretended to return |

## Layout and formats

Each set is a folder `material/<kind>/<set-id>/` with its data files (JSON, UTF-8) and a `manifest.json`: the
set id, `kind`, `version`, `status`, a description, the design sections it follows, its licence, and each
file's `path`, `bytes` and `sha256`.

- **Paragraph sets** (`narration-en.v1`, `alignment-en.v1`, `demo-en.v1`). Each paragraph is a Segment as
  `submit_job` takes it (design 7.2): a `segment_id` and `cues`, each cue a `text` and optional `exact`
  spans `{start, end}` (code points into the cue, end exclusive, on whole words). `{"segment_id", "cues"}`
  can be sent as they are. The other fields are metadata, left out when sending:
  - `spoken_chars`: the length of the join in code points (design 7.2);
  - `target_spoken_chars`: a ladder paragraph's rung;
  - `boundaries`: in the alignment benchmark, one entry per cue boundary, `{"after_cue": k, "pause":
    true|false}`. `pause` is true when cue *k* ends a sentence and false when it ends mid-sentence with no
    punctuation, so the measured error can be split by boundary kind. No cue ends in a comma or a dash,
    where a pause is uncertain.

  The demo's `hints` are the request-level Hint fragments to send with it. Each paragraph set also lists its
  `invented_names`. The calibration corpus also has a top-level `design_text` paragraph (`segment_id`
  `cal-design`, one cue): section 3.2's calibration set is "the design text plus three corpus paragraphs",
  and `measure_voice` renders this one first. Carrying it in the corpus means the corpus version, and so the
  measurement key, covers the text the calibration renders.
- **`canary.json`**: `voice` {`description`, `design_text`, `seed`} for designing the canary clip with
  VoiceDesign; `gate` {`text`, `seed`} for the gate render. `voice.design_text` is the service's default
  design text, copied from design section 16; the calibration corpus's `design_text` paragraph is the same
  text.
- **Fixtures**: each file states its conventions at the top. Where the design left a point open, the code
  that consumes the fixture settled it (the text pipeline and lint in WP10, QA in WP14), and a case's `note`
  says which rule it pins. Invisible and combining characters are written as `\u` escapes.

## The rules the spoken texts follow

Everything except `text-v1` is in **spoken form** (design 9.1), so the service's own text checks raise
nothing on it:

- numbers, measures and units in words ("three hundred and twelve", "nineteen kilometres"); no digits;
- no symbols beyond the punctuation a reader voices (`. , ; : ! ? ' ’ ‘ " “ ” ( ) - – — …`);
- no unit abbreviations, and no lone letters other than "a", "A" and "I";
- cues already in canonical form (NFC, single spaces, trimmed); British spelling;
- in the calibration corpus every number word except "one" is marked as an exact span, and spans cover
  whole words only, never punctuation;
- the alignment benchmark uses only letters that fold to A–Z, the aligner's alphabet (design 11.2);
- the canary's description is positive-only: no word from the negation list of design 3.5.

`tests/material/` checks all of this with the service's own code: every segment goes through the text
pipeline (`narration.text`) as a caller's would, and the canary's description through the lint
(`narration.lint`). It also recomputes every hash, and checks the ladder's lengths against the rungs in
design section 16.

## Versions and the freeze

The four spoken sets, `narration-en.v1`, `alignment-en.v1`, `canary.v1` and `demo-en.v1`, are
**`"status": "frozen"`**. They were frozen as version 1 on 2026-09-27 after **gate H1** (plan.md section 5),
when the owner had listened to sample renders of them and approved them. A frozen set never changes:
- the measurement key includes the corpus version (design 10.2);
- the published alignment error names the benchmark's id and sha256 (11.2);
- every engine pin records the canary's material id (plan.md DC-3).

Each of those is made from the set's manifest, so a new text is a new set (`narration-en.v2`, and so on).
`tests/material` pins each frozen set's manifest sha256 (`FROZEN_MANIFESTS`).

The two fixture sets, `text-v1` and `qa-faults-v1`, stay **`"status": "draft"`**. They are test data: no key,
engine pin or published number names them, so they change with the tests that use them.

The hashes are over the files' exact bytes. `.gitattributes` keeps every file's line ends LF on every
platform, so a checkout on any machine hashes the same.

## Changing a text

A frozen set is never edited. Copy it to a new set id (`narration-en.v2`), change the copy, and point the
service at it: the configuration names the corpus and the benchmark (`[measurement] corpus`,
`[alignment] benchmark`), and the code names the canary and the demo. A draft set is edited in place:

1. Edit the JSON. Keep each cue in canonical form.
2. If a cue with exact spans changed, recompute their `start` and `end` in code points (in Python,
   `text.index("forty-eight")` and its `len`), and the paragraph's `spoken_chars`.
3. Update the file's `bytes` and `sha256` in the set's `manifest.json`:
   `uv run python -c "import hashlib,sys; d=open(sys.argv[1],'rb').read(); print(len(d), hashlib.sha256(d).hexdigest())" <file>`
4. Run `uv run pytest tests/material`. A failure names the value it expected.

## Licence

This material is part of the project's source and is licensed under PolyForm Noncommercial 1.0.0
(`LICENSE` at the repository root; plan.md section 1.4).
