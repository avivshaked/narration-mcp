# ADR 0003: `max_new_tokens`: a ceiling of 8192 at `load`, and a cap per call derived from the text (DC-4)

- Status: **accepted: option C** (the owner, 2026-09-26; design revision 5.5, section 10.1 and Appendix A;
  contracts 1.6). Proposed by WP20 with the options below.
- Date: 2026-09-26
- Design sections: 10.1 (audio-changing settings, pinned explicitly), 11.1 step 1 (`TOKEN_CAP_HIT`),
  6 (`EngineProfile`), 3.2 (an over-long segment is warned about, never refused), Appendix A
- Evidence: [spikes/dc4-max-new-tokens](../../spikes/dc4-max-new-tokens/README.md),
  [spikes/h-i-qwen-load](../../spikes/h-i-qwen-load/README.md),
  [spikes/d-repeat-test](../../spikes/d-repeat-test/README.md)

## Decision

1. **`load` passes the snapshot's `max_new_tokens`, 8192, as the ceiling**, with the other nine sampling
   values, all explicit (`narration_qwen3tts.settings.effective_generation` reads them from the snapshot;
   `parse_generation` refuses a `load` without any of them).
2. **Every `synthesize` and `design` call passes its own `max_new_tokens`**, which the daemon computes from
   the text the call speaks (the engine text, or the design text):
   `min(8192, max(floor, ceil(per_char × len(text))))`, with `per_char` = 2.5 and `floor` = 128, the
   `[engines.*]` keys `max_new_tokens_per_char` and `max_new_tokens_floor`, hashed into the engine profile
   (`narration.contracts.names.max_new_tokens_for`).
3. **The worker refuses, never clamps.** A call without a cap, or with a cap below 2 or above the loaded
   ceiling, is `INVALID_REQUEST` for `max_new_tokens`, checked with the other arguments (before
   `VOICE_NOT_PREPARED`). 2 is the floor because qwen-tts passes `min_new_tokens=2` to the talker; a `load`
   whose ceiling is below 2 is refused the same way.
4. **The worker passes the call's cap to qwen-tts as that call's `max_new_tokens`** and echoes it in the
   reply (`max_new_tokens`), so the daemon can check that its rule was applied. The engine checks that the
   talker was given exactly that cap, and reports `INTERNAL` if not.
5. **The worker reports the cap exactly.** `new_tokens` is the codec frames decoded: the talker's steps less
   one, so at most the cap less one. `hit_token_cap` is true exactly when the talker's last sampled token is
   not the end token; it is never guessed from the audio's length.
6. **Tests plant the fault with a low call cap.** The GPU test renders with a call cap of 24 and gets
   `hit_token_cap: true` with 23 frames, while the loaded ceiling stays 8192.

## Context

The design names `max_new_tokens` as an audio-changing setting that must be passed explicitly. It
assumed 2048, the library's hard-coded fallback. But `Qwen3TTSModel.from_pretrained` reads the pinned
snapshots' `generation_config.json`, which sets **8192**, and every piece of clone evidence ran with
8192 (plan.md section 1.3 item 1).

The cap has two jobs:

- **It stops a runaway render**, one that never emits its end token.
- **It makes `TOKEN_CAP_HIT` fire** (section 11.1 step 1), so that a take the model cut short never
  passes.

With 8192 alone, the first job is done late and the second practically never happens.

## The options the owner chose from


| | Option | Runaway cost (BELIEVE) | `TOKEN_CAP_HIT` | Evidence stays valid | Change needed |
|---|---|---|---|---|---|
| **A** | Keep 8192 for every segment | **about 22 min or more** of GPU per runaway | practically never | yes | none |
| **B** | A fixed lower cap, e.g. 2048 | about 5.5 min | only above about 2,600 characters | yes, for renders under 2048 | new engine profile (the cap is hashed) |
| **C** | **A cap per segment from its engine text:** `min(8192, max(floor, ceil(k × len(engine_text))))` | about 1 min for a 150-character segment | on any segment that runs well past its text | yes, for every render that ends under its cap | new engine profile (`k`, `floor` hashed); `synthesize` gains `max_new_tokens`; the daemon computes it |

WP20 recommended C, with `k` = 2.5 frames per character and `floor` = 128, and the owner chose it:

- the factor gives about three times the largest rate seen (0.82 frames per character);
- the floor is about 10 s of audio, so a very short cue always has room;
- option B would effectively refuse a very long segment, which section 3.2 forbids (warn, never refuse).

### Why the evidence stays valid (KNOW)

**The cap only truncates.** Two paragraphs were rendered with the same seed at caps of 8192, 2048 and
exactly *N* (the talker steps the full render took). All three renders were bit-identical.

At *N* − 1, `hit_token_cap` was true and the audio differed. So a lower cap changes only renders that
would have run past it. Every existing take and hash stays valid under A, B or C, as long as the render
ends under its cap.

## Evidence

**KNOW** means measured here, with the script and its output saved. The machine is one 24 GB consumer
NVIDIA GPU, shared, with the `bit_exact` tier (ADR 0002).

| Claim | Label | Where |
|---|---|---|
| The effective cap is 8192 (`generation_config.json`); 2048 is the library's fallback | KNOW (source + run) | `workers/qwen3tts/src/narration_qwen3tts/settings.py`; `spikes/h-i-qwen-load/results.json` |
| At caps 8192, 2048 and *N*, a render is identical to the 8192 render. ladder-150 (seed 12, *N* = 113) and ladder-350 (seed 21, *N* = 250) | KNOW | `spikes/dc4-max-new-tokens/results.json` (`cap_only_truncates: true`) |
| At *N* − 1, `hit_token_cap` is true, with *N* − 2 frames decoded | KNOW | same |
| **A cut take is not a prefix of the uncut take.** Decoding fewer frames changed earlier samples: ladder-150 differed from sample 0, and ladder-350 matched only up to sample 286,080 of 476,160. It fails either way, so this matters only to anyone comparing audio | KNOW | same (`identical_prefix_samples`) |
| The codec runs at 12.5 frames per second, so 8192 frames is about 655 s of audio | KNOW | `spikes/h-i-qwen-load/results.json` (`frames_per_audio_s`) |
| **Frames per character of engine text were 0.70–0.82 in 13 distinct renders** (`non_streaming_mode` false) of 80 to 560 characters, in two voices and several seeds. The longest paragraphs had the lowest rates (0.72 at 347 characters, 0.70 at 560) | KNOW | `spikes/dc4-max-new-tokens/results.json`; `spikes/d-repeat-test/renders.csv`; `spikes/h-i-qwen-load/results.json`; `spikes/e-non-streaming-ab/results.json` |
| Generation ran at about 6.2 talker steps per second (RTF about 2.0) at 250–390 steps | KNOW | `spikes/h-i-qwen-load/results.json` (391 steps in 62.7 s); `spikes/dc4-max-new-tokens/results.json` (250 steps in 39–40 s) |
| A runaway to 8192 steps would take at least 8192 / 6.2 ≈ 22 min, and likely more, because each step's attention grows with length | BELIEVE (extrapolated; not run: it would hold the shared GPU for over 20 minutes) | |
| Other voices, slower deliveries or other languages stay under 2.5 frames per character | ASSUME (only d2 and d4, English, were measured). WP33's measurement ladder will show each voice's rate | |

## Consequences

- **Contracts (done, 1.6):** `names.max_new_tokens_for`, and the `[engines.*]` keys
  `max_new_tokens_per_char` and `max_new_tokens_floor`, hashed into the engine profile. The `load`
  settings keep 8192 as the ceiling.
- **The protocol (WP16):** `synthesize` and `design` take a required `max_new_tokens`, and the audio reply
  echoes it.
- **The worker (WP20, done):** checks the cap (2 to the ceiling), passes it to `generate`, echoes it, and
  keeps the other nine values from `load`.
- **The daemon (WP31/WP33)** computes the cap from the text the call speaks. The text and the hints are
  already in the render key, and the rule is in the profile hash, so the key still comes only from the
  request and the pins (section 10.2).
- **WP33's measurement** may record each voice's frames per character. A voice near the rule's limit
  would then get a warning, not a cut.
- WP32's `vram_need_mb` need not allow for an 8192-frame render in practice, but the ceiling still permits
  one (BELIEVE: about 1.6 GB more KV cache than a 30 s render; not measured).

## Alternatives considered

- **2048, the value the design assumed.** It invalidates no evidence, because every render under 2048
  steps is unchanged. But the choice between it and a derived cap is the same decision as B versus C, so
  it is folded in as option B.
- **Stop on a silence or repetition detector instead of a cap.** That would be an audio-changing
  heuristic inside generation. Not proposed: the design keeps workers as model runners, and QA already
  judges the take.

## What would change this

- A voice or language measured above about 2 frames per character, which would raise `k`.
- A runaway actually observed, which would give its real cost and replace the BELIEVE row.
- A qwen-tts upgrade that changes `generation_config.json` or the codec's frame rate.
