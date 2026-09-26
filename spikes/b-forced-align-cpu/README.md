# Spike (b): forced alignment on the CPU

*Plan.md WP15. Run 2026-09-26 on the development machine (Windows 11, 32 logical CPUs, capped at 4
threads, no GPU used). Scripts: [`spike_b.py`](spike_b.py), [`calibrate.py`](calibrate.py),
[`wildcard.py`](wildcard.py) with [`wildcard_emit.py`](wildcard_emit.py). Outputs:
[`results.json`](results.json), [`calibration.json`](calibration.json), [`wildcard.json`](wildcard.json),
[`wildcard_boundaries.csv`](wildcard_boundaries.csv). No audio is saved; the bake-off's takes are read in
place.*

*The bake-off's text is private. The scripts read it at run time and write numbers only: timings, frame
counts, errors and scores, keyed by the bake-off's ids (take, paragraph `n01`–`n08`, cue and word
index). No text, word or transcript from it appears in these files.*

## The questions

1. Design section 11.2 step 3 BELIEVED that `torchaudio.functional.forced_align` and `merge_tokens` run
   on the CPU of the QA worker's build (torch and torchaudio 2.11.0+cu128). Section 1 says an earlier smoke
   test was never saved. Does it run?
2. `facebook/wav2vec2-large-960h-lv60-self` at the pinned revision has only `pytorch_model.bin`. Does it
   load under transformers 5.17.0 and torch 2.11, where `torch.load` defaults to `weights_only`?
3. (Added for plan.md DC-11.) Does a wildcard token for the words the aligner cannot spell fix the cue
   boundaries they displaced, and move no other?

## How to run it

With the QA worker's interpreter, from the repository root, and `NARRATION_MODELS_ROOT` set (plus
`NARRATION_BAKEOFF_ROOT` for the parts that align bake-off takes):

```sh
workers/qa/.venv/Scripts/python.exe spikes/b-forced-align-cpu/spike_b.py --out spikes/b-forced-align-cpu/results.json
workers/qa/.venv/Scripts/python.exe spikes/b-forced-align-cpu/calibrate.py --out spikes/b-forced-align-cpu/calibration.json
uv run python spikes/b-forced-align-cpu/wildcard.py --scratch .dev/<folder>
```

The first two need the worker's `src` on `PYTHONPATH`. `wildcard.py` runs with the server's venv (it uses
`narration.align`) and starts `wildcard_emit.py` with the worker's. They take about 30 s, 85 s and 75 s.
`spike_b.py` aligns four paragraphs (n08, n01, n05, n06) of the bake-off take
`outputs/qwen3-tts-1.7b-clone-d2-late-night_take1-seed1/r48_names_probe.wav`, through the worker's own code
(`workers/qa/src/narration_worker_qa/align.py`).

## Answers

**1. Yes: `forced_align` and `merge_tokens` run on this CPU (KNOW).**

- Both are present in torchaudio 2.11.0+cu128. Calling them raised no warning, and `forced_align`'s
  docstring carries no deprecation notice (`results.json` → `api`).
- They are deterministic: two runs on the same input gave identical spans, on a synthetic emission and
  on each real paragraph.
- With fewer frames than tokens plus repeats, `forced_align` **raises** `RuntimeError: targets length is
  too long for CTC. Found log_probs length: 4, targets length: 4, and number of repeats: 1`. The worker
  checks `T ≥ L + R` first and replies `ALIGNMENT_ERROR` with `{reason: "too_short", frames, tokens,
  repeats}` instead (`api.worker_guard`, `alignment.too_short_0_1_s`: 0.1 s of audio for 25 tokens of an
  invented sentence gives 4 frames, and no crash).
- Speed, 4 threads: emissions plus alignment take about 0.1 × the audio's length (19.8 s of audio in
  1.8 s; 6.6 s in 0.5 s). The frame count `(n − 400) // 320 + 1` at 16 kHz matched the model's output on
  every paragraph (987, 327, 623, 755).

**2. Yes: the pinned `pytorch_model.bin` loads (KNOW).** No safetensors conversion is needed.

- Revision `54074b1c16f4de6a5ad59affb4caa8f2ea03a119`. The file's sha256 is
  `00b604cf4d28e86559e8adaeb3a186daa89dc37f5ab216771a0a15a26db0de9f`, which matches the models manifest.
- `torch.load`'s `weights_only` default is `None` in torch 2.11, which means true. transformers 5.17.0
  calls it with `weights_only=True` explicitly (the spike recorded the call), and the state dict loads
  under it.
- Loading info: one missing key, `wav2vec2.masked_spec_embed`, SpecAugment's mask vector. It is used
  only in training, so it has no effect at inference. There were no unexpected or mismatched keys and no
  warnings. The model is float32 with `sdpa` attention and 315,471,520 parameters. It loaded in 2.4 s,
  offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`).
- The weights are the right ones. The model's unconstrained (greedy) reading spells 12 of paragraph n08's
  13 transcript words exactly as the text does, 41 of n01's 48, 12 of n05's 15 and 24 of n06's 27. The
  words it spells otherwise are the text's invented names, which it spells by ear. The paragraphs'
  numbers are written in digits, so they are not transcript words; the model reads them out as words.

**Decision: pin `facebook/wav2vec2-large-960h-lv60-self` at `54074b1c16f4de6a5ad59affb4caa8f2ea03a119`,
and load its `pytorch_model.bin` as it is.** What would reverse it: a transformers or torch upgrade that
refuses the file. The safetensors alternative is then the Hub's `refs/pr/*` conversion of the same
weights, if one exists, or a local conversion, each verified against these emissions. Neither is needed
now, so nothing was downloaded.

## A finding for the lead: words left out of the transcript

Design section 11.2 step 1 leaves a word with characters outside the alphabet (a digit, a symbol) out of
the transcript, "and the words around it still align". The bake-off's paragraphs are full of digits, so
the spike measured what happens to those neighbours.

**KNOW:** with the word dropped, forced alignment has to spend the unspoken number's audio on blanks, and
the neighbouring words' letters latch onto matching sounds inside it. The words next to a left-out word
are the ones that move: in the table below they are off by 11–30 frames on average, against 2–18 for
all words, and in n05 (15 transcript words, 41 in the greedy reading) by up to 89 frames. The fix measured here is one
wildcard token `*` per run of left-out words. It is an extra emission column with log-probability
log(1 − P(blank)) per frame, so it can absorb speech but not silence. The error is measured against the
model's own unconstrained (greedy) reading, for the words it spells exactly as the text does:

| Paragraph | Words | Dropped: mean / worst error | Dropped, next to a left-out word | Wildcard: mean / worst | Wildcard, next to one |
|---|---|---|---|---|---|
| n08 | 12 | 2.2 / 53 frames | 13.3 frames (n = 2) | 0.0 / 0 | 0.0 |
| n01 | 41 | 1.9 / 43 frames | 11.0 frames (n = 7) | 0.0 / 0 | 0.0 |
| n05 | 12 | 17.6 / 89 frames | 30.1 frames (n = 7) | 0.4 / 9 | 0.6 |
| n06 | 24 | 2.9 / 43 frames | 11.8 frames (n = 6) | 0.0 / 0 | 0.0 |

One frame is 20 ms, so 89 frames is 1.78 s. **Caveat:** the reference is the same model's argmax path, not
hand marks. It shows where the model itself hears each word, not where the word truly starts; WP38's
hand-marked benchmark measures that.

This matters only for text that already carries a `WRITTEN_FORM_TOKEN` warning (R6 asks for spoken text
without digits). But a mislocated neighbour can move a cue boundary by seconds. The lead approved the
wildcard as plan.md DC-11, on the conditions measured in part 3 below.

## Starting values for the thresholds (calibration)

Design section 11.2 sets the confidence and cross-check thresholds from the alignment benchmark (WP38),
which needs hand marks. Until then `narration.align` needs starting values. They are labelled ASSUME,
and [`calibrate.py`](calibrate.py) measured what they separate. Its output is
[`calibration.json`](calibration.json). The inputs were the six bake-off clone takes, 8 paragraphs each,
cut into sentences as cues, in about 85 s. The planted cases are another paragraph's text over a
paragraph's audio, and an invented sentence (`NEVER_SPOKEN`) added last or second.

**Confidence** is the mean posterior of a cue's letters (KNOW):

| Cues | n | Scores |
|---|---|---|
| sentences as spoken | 114 | min 0.595, p5 0.858, p50 0.920 |
| planted, not in the audio (another paragraph's text; the invented sentence, last or second) | 8 | 0.143, 0.152, 0.217, 0.258, 0.333, 0.393, 0.499, 0.704 |
| spoken cues beside a planted one | 4 | 0.235, 0.619, 0.725, 1.000 |

The lowest spoken cue (0.595) is n05's second sentence (take d4 seed 1); n05 is the paragraph with the
most words left out of the transcript. A sentence missing from the audio also drags its neighbour down (0.235),
because the aligner squeezes its letters into the neighbour's audio. The take is broken either way.

**Starting values (ASSUME):** `CUE_LOW_CONFIDENCE` below **0.75**, and a cue is unplaceable below **0.50**.
0.50 is under every spoken cue and over 7 of the 8 planted ones. The eighth, 0.704, is n08's second sentence
over n05's audio, which found sounds to match; it would get `CUE_LOW_CONFIDENCE`.

This calibration used the transcript as it was before DC-11, with the numbers dropped. A cue's confidence
counts only its letters, and the wildcard moves letters only where they were displaced, so the values
should hold (BELIEVE); the evidence tests, run with the wildcard, find every spoken cue at 0.6 or more and
the median at 0.85 or more. The benchmark (WP38) sets them for good.

**Pauses** use 20 ms energy frames, measured against the paragraph's 95th-percentile frame level. A gap
counts when it holds at least two consecutive silent frames (40 ms). KNOW:

| Gap between two aligned words | n | −30 dB | −35 dB | −40 dB | −45 dB |
|---|---|---|---|---|---|
| between sentences (a pause is expected) | 66 | 100 % | 100 % | 98.5 % | 92.4 % |
| inside a sentence (commas, stop closures) | 1266 | 20.0 % | 16.0 % | 11.8 % | 8.2 % |

The deepest frame between sentences was 68 dB below the reference at the median (p5 −77 dB).

**Starting value (ASSUME):** silence is below the 95th percentile **− 35 dB**, and a pause is at least **2
frames** (40 ms). That finds every sentence pause, while in-sentence gaps are found only where the audio
really has one.

## 3. The wildcard on six takes (plan.md DC-11)

The lead approved the wildcard on two conditions: that it fixes the boundary after n05's left-out numbers
on all six takes, and that it moves no other boundary beyond noise. [`wildcard.py`](wildcard.py) aligned
the evidence tests' 24 paragraph pairs (four per take, one cue per sentence) through `narration.align`
twice: *before*, the transcript with its wildcards taken out, which is exactly what the aligner built
before DC-11, and *after*, the transcript as it is built now. Every one of the 90 cue boundaries is in
[`wildcard_boundaries.csv`](wildcard_boundaries.csv); the summary is in [`wildcard.json`](wildcard.json).

**Condition 1, met (KNOW).** In all six takes the boundary between n05 and n06 now holds the 0.9 s of
silence the bake-off inserted between them; before, the next cue started 1.3–3.4 s early:

| Take | Next cue's start, before | After | End of the inserted silence | Silence outside the boundary, before → after |
|---|---|---|---|---|
| d2 seed 1 | 10.52 s | 13.46 s | 13.38 s | 2.86 s → 0 |
| d2 seed 2 | 11.02 s | 14.28 s | 14.18 s | 3.16 s → 0 |
| d2 seed 3 | 12.08 s | 13.46 s | 13.38 s | 1.30 s → 0 |
| d4 seed 1 | 11.36 s | 15.04 s | 14.58 s | 3.22 s → 0 |
| d4 seed 2 | 12.18 s | 15.64 s | 15.06 s | 2.88 s → 0 |
| d4 seed 3 | 10.42 s | 14.36 s | 13.86 s | 3.44 s → 0 |

(A start after the silence's end is the next paragraph's own lead-in silence, which the pause includes.)
Paragraph boundaries that hold the inserted silence: 18 of 24 before, 24 of 24 after.

**Condition 2, met (KNOW).** 84 of the 90 boundaries did not move at all (0 ms). The other six are the
six boundaries above. The 33 sentence boundaries that have a reference on both sides (the greedy reading
of the words on either side) keep the same error, mean 0.08 s, p95 0.13 s, max 0.21 s, before and after.

**Words (KNOW).** Against the greedy reading, over the 2,404 words it spells as the text does: mean error
3.51 → 0.19 frames, p95 21 → 0; for the 484 next to a wildcard, mean 16.1 → 0.5 frames, p95 79 → 0,
worst 200 → 28.

**What a wildcard's own span is worth (KNOW).** This decides what a cue whose words are all unspellable
gets.

- Its edges are not word edges. Against the greedy reading of the words it stands for (116 wildcards
  with both neighbours matched), its start is off by 6 frames at the median, 24.5 at p95 and 51 at worst,
  and its end by 9, 32 and 50. That is 0.5–0.6 s at p95, where a letter word is off by 0 at p95. A CTC
  path may leave part of the number's speech to the blank, so the span covers only some of it.
- It cannot tell spoken from unspoken. A wildcard planted where nothing was left out still takes a span:
  in a pause between sentences, 1 frame, scored 1.0 in 20 of 24 (it takes a frame of the neighbouring
  speech) and moving no neighbour; between two words inside a sentence, 3 frames at the median and 23
  at worst, scored 0.86 at the median, and pushing its neighbours by up to 54 frames. Real wildcards score
  0.63 at the median.

**Decision:** a cue whose only tokens are wildcards has no alignable words: `CUE_UNALIGNED` with
`details.reason` `no_alignable_words`, null times, and no retake (DC-12). Its span can neither say where
the cue's speech starts and ends within the aligner's error, nor that the cue was spoken at all. A
wildcard's span still bounds the snapping of the cue it belongs to: the boundary goes into a pause next
to the span, which is how condition 1 is met. A wildcard's score never counts in a cue's confidence.
