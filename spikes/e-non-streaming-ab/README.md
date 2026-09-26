# Spike (e): the `non_streaming_mode` A/B for Base (optional)

Design sections 10.1 and 20 (e). Section 10.1 pins `non_streaming_mode=false` for Base, the value every
piece of clone evidence used. It says: "Change it only if a Phase 0 A/B justifies `true`, and that would
mean a new engine profile."

In qwen-tts 0.1.1 the flag decides how the text reaches the talker:

- **false** feeds the text alongside the codec stream;
- **true** gives the whole text up front.

- `render.py`: step 1, in the Qwen worker's venv under the GPU lock. It renders through the worker's
  engine with the section 10.1 switches. The audio and `renders.json` stay under `.dev/spikes/e/`.
- `score.py`: step 2, in the QA worker's venv under the GPU lock.
  - Whisper-large-v3 transcribes each render, as the bakeoff did.
  - WER is a word-level edit distance after Whisper's English normaliser.
  - WavLM-SV measures similarity to the voice's own clip (`spk_to_ref`, on the CPU).
  - Pace is in words per minute.
- `results.json`: the scores, with the transcripts. The texts are the service's own calibration
  paragraphs.

## Method

- **The renders:** the service's calibration paragraphs `narration-en.v1` cal-01, cal-02 and cal-03
  (196–235 characters), in d2 and d4, with seed 31.
- **Each render is made once per mode**, with everything else equal. That is 6 renders per mode, about
  82 s of audio each.

## Results (KNOW, 2026-09-26: one 24 GB consumer NVIDIA GPU, shared)

| `non_streaming_mode` | Renders | WER mean | `spk_to_ref` mean (min) | Pace (wpm) | Audio |
|---|---|---|---|---|---|
| **false** (pinned) | 6 | 0.0175 | 0.9813 (0.9781) | 176.9 | 82.3 s |
| true | 6 | 0.0219 | 0.9823 (0.9749) | 177.0 | 82.2 s |

- **No difference that matters.**
  - WER differs by one misheard word in about 230, the kind of variation one seed gives against another.
  - Similarity to the clip differs by 0.001 in the mean. `true` has the lowest single value.
  - Pace and length are the same.
- The two modes produce different audio for the same seed, with different hashes and lengths. It is a
  different delivery, not a better one.
- **Decision: keep `non_streaming_mode=false` for Base**, as section 10.1 pins it.
  - Nothing here justifies a new engine profile.
  - All the clone evidence (the bakeoff, spikes d and h+i, and the acceptance) used `false`.

## Limits

- **Small:** 6 renders per mode, one seed, two voices, English. It can show a large effect, not a small
  one.
- **Not measured:** the head and end of the take, which the mode might affect differently (the head
  insertion check is WP14's). Nor was it measured whether `true` behaves better on very long segments.
  The length ladder (spike c, WP39) is the place for that, if anyone wants it.
