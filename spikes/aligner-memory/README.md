# The aligner's memory by audio length (WP15's follow-up F5)

Design section 11.2. The QA worker's `align` op computes wav2vec2's CTC emissions for a whole take in one pass
on the CPU, then runs torchaudio's `forced_align` over them. WP15 measured takes up to about 40 s. This measures
longer ones: is one pass safe for the lengths a segment can reach, or should the emissions be computed in
chunks?

- `run.py`: the spike, run in the QA worker's venv. No GPU, no audio file read or written.
- `results.json`: its output (`ran_at` is UTC). No paths, no text.

## Method

For each length, a fresh process (the worker's thread cap of 8, offline):

1. load the pinned wav2vec2 snapshot (`facebook/wav2vec2-large-960h-lv60-self`) through the worker's own class,
   `narration_worker_qa.align.Wav2Vec2Aligner`;
2. make synthetic audio of that length (a 150 Hz tone in low-level noise: the emissions' cost does not depend on
   what is said) and a token sequence of 15 letters and word separators per second, a fast narration's rate;
3. align it with the class's `emission` and `spans`, sampling the process's resident memory every 5 ms;
4. report the peak above the memory held after the model loaded, and the times.

The ladder stops before a length whose peak, extrapolated as the square of the length from the last run (the
worst case, self-attention over the whole clip), would exceed half the machine's available memory.

## Results (KNOW, 2026-09-27, 13:07–13:09 UTC; a 64 GB Windows machine; torch 2.11.0, transformers 5.17.0)

| Audio | Frames | Tokens | Peak above the loaded model | Emissions | `forced_align` | Real-time factor |
|---|---|---|---|---|---|---|
| 30 s | 1499 | 450 | 1576 MB | 3.0 s | 0.01 s | 0.10 |
| 60 s | 2999 | 900 | 1651 MB | 6.9 s | 0.02 s | 0.12 |
| 120 s | 5999 | 1800 | 2649 MB | 16.7 s | 0.08 s | 0.14 |
| 240 s | 11999 | 3600 | 5291 MB | 41.3 s | 0.23 s | 0.17 |
| 480 s | | | not run: the quadratic worst case (21.2 GB) exceeded half the available memory | | | |

- The process holds about 0.75 GB after loading the model.
- An earlier run the same day (13:05 UTC, which failed only when writing its results) measured the same peaks
  to within 1 MB.
- **The peak grows linearly, not with the square of the length:** flat to 60 s, then about 22 MB per second of
  audio (2649 MB at 120 s, 5291 MB at 240 s). BELIEVE: PyTorch's CPU attention kernel does not materialise the
  full attention matrix, so the feature encoder's and the layers' activations, linear in the length, dominate.
  Extrapolated (BELIEVE): about 10.6 GB at 480 s.
- `forced_align` itself costs next to nothing; the emissions are the cost.

## What this means

- **One pass is safe for the segments the service expects.** A segment of up to 2 minutes needs under 2.7 GB of
  RAM above the model, and alignment runs at 0.10–0.17 × real time.
- A segment is never refused for its length (section 3.2), so a very long one is possible: a take at the
  8192-token ceiling (about 11 minutes at 12 Hz) would need roughly 15 GB (BELIEVE, linear extrapolation).
- **Proposed, not built (the lead's call):** if the service must bound the aligner's memory on smaller machines,
  compute the emissions in windows of about 60 s with a few seconds' overlap and stitch them (the frames of each
  overlap taken from the window whose centre is nearer). That bounds the peak at about 1.7 GB for any length. It
  changes the alignment near window edges slightly, so it would need WP15's alignment benchmark re-run before it
  ships.
