# Spike (f): the "Marufubens" check, re-run with silence prepended and saved

Design sections 1 and 20 (f). In the bakeoff's names probe, the invented name "Marufubens" opens segment
n05, and section 1 records that Whisper heard it split or misspelt in every take. Section 1 BELIEVED this
is not clipping. An earlier session had checked it with 0.5 s of silence prepended, but saved neither the
script nor its output. This spike repeats the check and saves both.

- `run.py`: the spike, run in the QA worker's venv under the GPU lock (about 1 minute of GPU).
- `results.json`: its output, **numbers only**.
  - The segments' sentences are the bakeoff caller's text, and Whisper's transcripts of them, the heard
    forms of the name included, are transcripts of the caller's takes. None of them is published.
  - For each transcription, the results give whether the name was heard as written, how many words it
    was heard as, and the character edit distance of the heard form to the written name. For each take,
    they give whether prepending silence changed the heard form.
  - The heard forms and the full transcripts stay in the gitignored `.dev/spikes/f/`. `run.py
    --publish-only` rebuilds `results.json` from them without the GPU.

## Method

- **Takes:** the bakeoff's six clone takes of the names probe (d2 and d4, seeds 1–3). Each segment is
  cut out by its recorded sample boundaries.
- **n05, where the name is the first word:**
  - the onset is measured: the leading silence by section 13's relative rule (the first 20 ms frame
    within 40 dB of the segment's p95 frame level), and the level of the first 20 ms;
  - the segment is transcribed as it is, and with 0.5 s and 1.0 s of digital silence prepended.
- **n04, the control:** the same name comes mid-sentence, where no clipping at a segment's start can touch
  it.
  - It is transcribed as it is and with 0.5 s of silence prepended.
- **The ASR is Whisper-large-v3**, called as the bakeoff's `eval/evaluate.py` called it:
  - the transformers pipeline in fp16, English, `return_timestamps=True`;
  - the pinned snapshot, offline, with 0 network attempts.
- **Whisper's English normaliser** is applied to both sides before the alignment. The *heard form* is the
  transcript's words that the word-level alignment places where the reference has the name.
- **The edit distance** is the Levenshtein distance in characters from the heard form's words, joined
  without spaces, to the written name. A split is counted by the number of words, not as an extra
  character.

## Results (KNOW, 2026-09-26)

Words heard and edit distance are given for each condition in turn: as is, then with 0.5 s of silence
prepended, then (n05 only) with 1.0 s.

| Take | n05 leading silence | n05 first 20 ms | n05 words heard | n05 edit distance | n04 words heard | n04 edit distance |
|---|---|---|---|---|---|---|
| d2 seed 1 | 0.08 s | −92.5 dB | 2 / 2 / 2 | 2 / 2 / 2 | 1 / 1 | 4 / 4 |
| d2 seed 2 | 0.54 s | −91.3 dB | 2 / 1 / 1 | 0 / 2 / 2 | 1 / 1 | 2 / 1 |
| d2 seed 3 | 0.02 s | −79.1 dB | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 1 | 1 / 1 |
| d4 seed 1 | 0.42 s | −90.5 dB | 2 / 2 / 2 | 1 / 1 / 1 | 1 / 1 | 2 / 2 |
| d4 seed 2 | 0.54 s | −92.0 dB | 2 / 2 / 2 | 2 / 2 / 2 | 1 / 1 | 1 / 1 |
| d4 seed 3 | 0.10 s | −91.6 dB | 2 / 2 / 2 | 2 / 2 / 2 | 1 / 1 | 1 / 1 |

- **The name was never heard as written, in any condition.**
  - That is 0 of 18 transcriptions of n05 and 0 of 12 of n04.
  - Where it was not, the edit distance to the written name was 0–2 characters for n05 (median 2), and
    1–4 for n04 (median 1).
  - The one n05 distance of 0 had exactly the name's letters, but heard as two words.
  - Mid-sentence (n04), where clipping at the segment's start cannot reach the name, Whisper still misspelt
    it every time.
- **Prepended silence did not fix it.**
  - It changed the heard form of n05 in 3 of the 6 takes, and never to the written name.
  - For n04 it changed the heard form in 1 of the 6 takes.
- **Every n05 segment starts in silence.** Its first 20 ms lie 21–38 dB below the speech threshold, and
  the speech begins 0.02–0.54 s in. No take starts mid-phoneme, which is where qwen-tts's proportional
  cut of the reference would have shown.
- **Conclusion: the split is how Whisper reads an invented word, not clipping.** Section 1's BELIEVE
  becomes KNOW for these six takes. The name was split into two words only at the start of a segment (14 of
  the 18 transcriptions of n05), and was always heard as one misspelt word mid-sentence (12 of 12 for n04).
  Whether the voice pauses inside the name there, or Whisper treats a sentence-initial unknown word
  differently, would take a listener to tell.
  - This bears on QA (WP14), and agrees with the design:
    - a term sent as a hint is collapsed in `wer_adj`, and its `asr_aliases` record how the ASR writes it
      (section 9.1);
    - Whisper "spells invented names unpredictably" (section 11.2).
  - The split form (two words for one) is one more alias shape to allow for.

## Limits

- **Two voices, one name, the bakeoff's takes only.** Nothing was rendered for this spike.
- **This is ASR evidence, not a listening test.**
