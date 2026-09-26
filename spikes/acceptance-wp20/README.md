# WP20 acceptance: a render through the worker protocol against the bakeoff's take

plan.md, WP20:

> *Accept:* a render through the worker protocol, with the bakeoff's settings and seed, scores WavLM
> similarity ≥ 0.98 against the bakeoff take.

- `run.py`: the check. It starts the real worker (`python -m narration_worker --role qwen3`) and speaks
  the protocol to it. The steps are in its docstring.
- `results.json`: its output. It contains no paths and no GPU name. The audio stays under
  `.dev/spikes/acceptance/`.
- The GPU test `test_acceptance_render_matches_the_bakeoff_take_s10_1`, in
  `workers/qwen3tts/tests/test_gpu.py`, runs the same check (markers: `gpu`, `model`, `evidence` and
  `slow`).

## Method

1. **The worker:**
   - `hello`;
   - `load` of Base, with the section 10.1 switches, the snapshot's ten sampling values passed
     explicitly, and `non_streaming_mode` false;
   - `prepare_voice` for d2 and d4 (ICL: the allowlisted clip and its exact transcript);
   - `synthesize` for segments n07 and n08 of the bakeoff's names probe. These are the texts of the
     bakeoff's caller, read from the bakeoff at run time and never copied here. The seed follows the
     bakeoff's rule, `seed × 1000 + index` with seed 1, and each call sends the daemon's cap for its text
     (`max_new_tokens_for`, DC-4, ADR 0003).
   - The clips are cloned only if their sha256 is in `NARRATION_SPIKE_ALLOW_SHA256` (section 17.4); the
     values are set locally and are not in this repository.
2. **The scorer:** WavLM-SV `microsoft/wavlm-base-plus-sv` (the evidence's revision), on the CPU in the
   QA worker's venv (`spikes/wavlm_similarity.py`, ported from the bakeoff's method).
   - Each render is compared with the matching segment of the bakeoff's seed-1 take.
   - As a baseline, the seed-1 take is compared with the seed-2 and seed-3 takes of the same text.
3. **The scorer checks itself:** it must reproduce the bakeoff's recorded `spk_to_ref`, the mean over a
   take's eight segments against the clip, within 0.002. A scorer that cannot do that proves nothing.

## Results (KNOW, 2026-09-26: one 24 GB consumer NVIDIA GPU, shared)

| Render | Seed | Call cap | Frames | Similarity to the bakeoff take | Same length | Identical at 16 bits |
|---|---|---|---|---|---|---|
| d2 n07 | 1006 | 285 | 108 | **1.0000** | yes | **yes** |
| d2 n08 | 1007 | 203 | 82 | **1.0000** | yes | **yes** |
| d4 n07 | 1006 | 285 | 110 | **1.0000** | yes | **yes** |
| d4 n08 | 1007 | 203 | 89 | **1.0000** | yes | **yes** |

- **Accepted.** Every render is ≥ 0.98.
- **The baseline:** another seed of the same text and voice scored 0.967–0.993 against the seed-1 take.
  So 0.98 does not separate "the same take" from "another take of the same voice", but 1.0000 does.
- **The scorer check passed:** d2 0.9776 against the recorded 0.9775, and d4 0.9663 against 0.9663.
- **The renders are the bakeoff's takes, sample for sample.**
  - Converted to 16 bits the way the bakeoff stored its takes (`soundfile.write`, default WAV subtype
    PCM_16), every render equals its segment of the bakeoff's file (`identical_to_bakeoff_pcm16`).
  - The bakeoff ran a day earlier, in its own venv, with the same pins and no determinism switches.
  - That is a stronger result than the acceptance asks for, and it backs ADR 0002 (`bit_exact`).

## Caveats

- **The runs.** `results.json` is from the fourth run (20:40), on main's WP16, with the worker's own WAV
  writer and a cap per call. The three earlier runs (19:2x to 19:4x) rendered with the load's ceiling,
  8192, and the first two used WP16's `workers/common` from its branch. All four gave the same frames and
  the same similarities, and the last three (the first did not compare them) the same 16-bit samples. The
  call cap only truncates, and here it was well above each render's length (`spikes/dc4-max-new-tokens`).
- **Two voices, one script, two segments.** The tier and the other spikes cover more renders, but not
  more voices.
