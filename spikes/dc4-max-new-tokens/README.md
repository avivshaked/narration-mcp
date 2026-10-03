# DC-4 evidence: does `max_new_tokens` change a render that does not reach it?

plan.md section 1.3 item 1 and DC-4 (`docs/design-log.md`). The owner chose a cap per call derived from the text:
[ADR 0003](../../docs/decisions/0003-max-new-tokens.md) (accepted: option C).

- `run.py`: the spike, run in the Qwen worker's venv under the GPU lock (about 5 minutes).
- `results.json`: its output. The audio stays under `.dev/spikes/dc4/`.

## Method

- **Two of the service's paragraphs, in the d2 voice:** `narration-en.v1` ladder-150 (151 characters,
  seed 12) and ladder-350 (347 characters, seed 21).
- **Every render goes through the worker's engine**, with the section 10.1 switches and all ten sampling
  values explicit. The tier is `bit_exact` (spike d), so equal hashes mean equal renders.
- **Each paragraph is rendered at four caps, each with the same seed.** The run of 2026-09-26 set each
  cap as the `load`'s ceiling, before the worker took a cap per call (ADR 0003, accepted: option C).
  `run.py` now passes it as the call's own `max_new_tokens` under the ceiling of 8192; both reach qwen-tts's
  `generate` as the same value.
  - 8192, the pinned value;
  - 2048;
  - *N*, the talker steps the 8192 render took, the smallest cap that still lets the talker emit its end
    token;
  - *N* − 1.

## Results (KNOW, 2026-09-26: one 24 GB consumer NVIDIA GPU, shared)

| Paragraph | *N* (talker steps) | Frames decoded | 8192 | 2048 | *N* | *N* − 1 |
|---|---|---|---|---|---|---|
| ladder-150 | 113 | 112 (8.96 s) | reference | identical | identical | `hit_token_cap`, 111 frames, differs |
| ladder-350 | 250 | 249 (19.92 s) | reference | identical | identical | `hit_token_cap`, 248 frames, differs |

- **The cap only truncates** (`cap_only_truncates: true`). A render that ends under its cap is
  bit-identical whatever the cap, so a lower or length-derived cap keeps every existing take valid.
- **A cut take is not a prefix of the uncut one.**
  - The codec decodes the frames together, so dropping the last frame changed earlier samples too.
  - ladder-150 differed from the first sample. ladder-350 matched only up to sample 286,080 of 476,160.
  - A cut take fails `TOKEN_CAP_HIT` anyway.
- **Frames per character of engine text:** 0.742 (ladder-150) and 0.718 (ladder-350). With the other
  spikes' renders the range is 0.70–0.82 over 80–560 characters, in two voices (ADR 0003).
- **Generation speed:** about 6.2 talker steps per second at 250 steps (39–40 s for ladder-350 at the
  lower caps).
  - Two renders were slower: ladder-350 at 8192 (60 s) and ladder-150 at *N* − 1 (28 s, against 18 s for
    the rest).
  - Other jobs share this GPU, and the slow renders were at different caps. So no effect of the cap on
    speed is claimed.

## Not measured

- **A real runaway to 8192 steps.** It would hold the shared GPU for over 20 minutes. Its cost in ADR 0003
  is an extrapolation (BELIEVE).
- **Voices other than d2, and languages other than English.**
