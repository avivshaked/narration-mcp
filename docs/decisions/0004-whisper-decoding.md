# ADR 0004: Whisper decodes as the evidence did: five beams, not conditioned on the previous window

- Status: **proposed** by WP22, 2026-09-26; the worker pins option C until the lead (or the owner) decides.
  Changing the pin is one constant, `narration_worker_qa.asr.DECODING`.
- Date: 2026-09-26
- Design sections: 11.1 step 2 (ASR), 4 (the QA worker), 10.1 (every output-changing setting explicit),
  11.1 step 4 (`wer_raw` 5.5–8.6 % on name-dense text)
- Evidence: [spikes/acceptance-wp22](../../spikes/acceptance-wp22/README.md) (`decoding.py`, `decoding.json`,
  `run.py`, `results.json`), [spikes/h-i-qa-load](../../spikes/h-i-qa-load/README.md)

## Context

Section 11.1 step 2 pins Whisper-large-v3 "greedy", with sequential long-form transcription "each [window]
conditioned on the text before it". The bake-off's evidence, which the section's WER figures (5.5–8.6 %) come
from, was made by `eval/evaluate.py`, which called the transformers ASR pipeline without a generation config.
KNOW (transformers 5.17.0's source, the version of both the bake-off's eval venv and the QA worker): the pipeline
then fills `num_beams` with 5, and `condition_on_prev_tokens` defaults to off. So the evidence was made with
**five beams, not conditioned**, and no temperature fallback.

## Evidence (KNOW, 2026-09-26)

The six clone takes of the names probe (d2 and d4, seeds 1–3, about 119 s each), each transcribed whole through
the worker's own class (float16, eager attention, word times, long-form, the determinism switches), scored as
`eval/evaluate.py` scores (Whisper's English normaliser, word errors over the reference's 256 words):

| Decoding | WER per take (d2 s1, s2, s3, d4 s1, s2, s3) | mean | worst | peak VRAM allocated (119 s) | time for the six |
|---|---|---|---|---|---|
| bake-off (`eval/results.csv`) | 0.086, 0.055, 0.074, 0.059, 0.059, 0.055 | 0.064 | 0.086 | | |
| A. greedy, conditioned (section 11.1 as written) | 0.398, 0.090, 0.051, 0.070, **0.402**, 0.070 | 0.180 | 0.402 | 7.9 GB | 137 s |
| B. greedy, not conditioned | 0.074, 0.070, 0.055, 0.078, 0.051, 0.098 | 0.071 | 0.098 | 6.6 GB | 106 s |
| **C. five beams, not conditioned (the evidence's)** | **0.086, 0.055, 0.074, 0.059, 0.059, 0.055** | 0.064 | 0.086 | 9.5 GB | 151 s |
| D. greedy, conditioned, OpenAI's temperature fallback | 0.398, 0.090, 0.051, 0.070, 0.449, 0.070 | 0.188 | 0.449 | 7.9 GB | 149 s |

- **A loops.** On two of the six takes the transcript repeats a passage (341 and 339 normalised words for a
  256-word text). A take with a looping transcript fails the text match (section 11.1 step 4) though its audio
  is fine, and every retake of it can fail the same way.
- **D does not cure it.** The fallback's compression-ratio and log-probability thresholds did not catch these
  loops, and the fallback samples, so it would also cost determinism.
- **C reproduces the bake-off's WER exactly** on all six takes, with the worker's word times, eager attention
  and determinism switches.
- **B** is deterministic too, 30 % faster and 2.8 GB lighter, but its WER differs from the evidence by up to
  4.3 points on one take (d4 seed 3), so the section's figures and the thresholds set against them would not
  describe it.
- VRAM figures are the allocator's peak including the resident models (3.3 GB); see spike (h), QA half.

## Decision (proposed)

**Option C.** `transcribe` decodes with five beams, no sampling, temperature 0, no fallback thresholds, and
sequential long-form **not** conditioned on the previous window's text, all passed explicitly in the worker's
own generation config. Section 11.1 step 2 would read: "Whisper-large-v3 (fp16, English, five beams, not
conditioned on the previous window, no temperature fallback, word timestamps; revision pinned)".

## Alternatives

- **A** (the section as written): refused by the evidence above.
- **B** (greedy, not conditioned): the choice if VRAM or speed matters more than matching the evidence; it
  keeps the section's word "greedy". The QA group's need would drop from about 10 GB to about 9 GB for long
  takes.
- **D** (conditioned with fallback): refused; it loops and is not deterministic.

## What would reverse it

- The QA group's VRAM need (spike (h)) being too high for the GPUs the project supports: then B.
- A later measurement on other voices or texts where C loops or B is clearly better.
