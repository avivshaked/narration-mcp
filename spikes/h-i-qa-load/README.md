# Spikes (h) and (i), QA half: offline loading, VRAM and load times

Design section 20, Phase 0: **(h)** VRAM and load times; **(i)** offline loading from a snapshot directory
named by its commit SHA. This folder covers the QA group's GPU models, Whisper-large-v3 and
WavLM-base-plus-sv. The Qwen half is `spikes/h-i-qwen-load`. The CTC aligner runs on the CPU; its memory is
`spikes/aligner-memory`.

- `run.py`: the spike, run in the QA worker's venv under the GPU lock. It drives the worker's own classes
  (`narration_worker_qa.asr.WhisperAsr`, `narration_worker_qa.sv.WavLmSv`) with the worker's determinism
  switches, so it measures the code the service runs.
- `results.json`: its output from 2026-09-26. It contains no paths, no GPU name and no text: the audio is
  named by the bakeoff's ids (the d2 clone take, seed 1).

## Method

1. Set the worker's start-up environment before torch loads (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, thread caps of 8), and install a guard that refuses every socket
   connection and name lookup leaving the machine, recording each attempt.
2. Check that each snapshot folder is named by its 40-hex revision; apply the QA worker's switches (TF32 off,
   cuDNN deterministic, benchmark off, deterministic algorithms warn-only); load Whisper (float16, eager
   attention) and WavLM (float32) on `cuda:0`, both resident, as the QA group is.
3. Transcribe (English, word times, long-form: what the job engine sends) and embed a 40 s slice and the
   whole of the bakeoff's d2 clone take, seed 1 (119 s), recording the allocator's peaks; embed both again
   on the CPU (the canary's path) and compare the embeddings.
4. Measure how the peaks grow with length, from 10 s to 119 s cut from the start of the take: transcription
   with and without word times, and embedding, which is also compared with the mean of 30 s windows'
   embeddings.
5. Unload, load each model a second time, and re-hash every weight file against the models manifest.

## Results (KNOW, 2026-09-26: one 24 GB consumer NVIDIA GPU, shared; torch 2.11.0+cu128, CUDA 12.8)

| | Whisper-large-v3 | WavLM-base-plus-sv |
|---|---|---|
| Offline load from the SHA-named folder | yes, **0** network attempts | yes, **0** network attempts |
| Load time: first / second in the process | 2.5 s / 2.5 s | 0.3 s / 0.25 s |
| Reserved VRAM added by the load | 3184 MB | 420 MB |
| Weights re-hashed against the manifest | 11 files, 0 mismatched | 3 files, 0 mismatched |

- **The group resident:** 3331 MB allocated, 3606 MB reserved. After unload: 56 MB reserved.
- The first run of the spike (the same day, without step 4) measured the same loads, peaks and speeds within a
  few per cent; `results.json` is the second run.
- **CUDA context:** about 450 MB (the device-level change when the process started CUDA; an estimate, since
  other jobs share the device).
- **Load times were measured with a warm OS file cache**, as in the Qwen half.
- **Speed:** transcription with word times runs at 0.17–0.24 × real time (40 s in 6.8 s, 119 s in 28.6 s);
  embedding takes 0.3 s for 40 s on the GPU. On the CPU, embedding 40 s takes about 2.4 s, the first call
  included (it loads the CPU copy); the embeddings equal the GPU's to 6 decimal places (cosine 1.000000).

### What the peaks grow with (allocated above the resident models)

| Audio | transcribe, no word times | transcribe, word times | embed | embed: cosine of one pass to the mean of 30 s windows |
|---|---|---|---|---|
| 10 s | 251 MB | 3179 MB | 126 MB | 1 (one window) |
| 20 s | 259 MB | 3420 MB | 288 MB | 1 (one window) |
| 30 s | 274 MB | 3589 MB | 602 MB | 1 (one window) |
| 60 s | 307 MB | 4377 MB | 2234 MB | 0.99904 |
| 90 s | 326 MB | 4956 MB | 4894 MB | 0.99647 |
| 119 s | 309 MB | 4568 MB | 8446 MB (13.3 GB reserved) | 0.99396 |

- **Word times cost about 3 GB at any length.** BELIEVE: asking `generate` for token timestamps turns on
  `output_attentions`, and the encoder then keeps the self-attention maps of all 32 layers, 32 × 20 heads ×
  1500² × 2 bytes = 2.9 GB. Only the decoder's cross-attention is needed for word times. Not changed here:
  the job engine needs word times, and the cost is flat.
- **Embedding grows with the square of the length** (WavLM's self-attention over the whole clip): 0.6 GB at
  30 s, 2.2 GB at 60 s, 8.4 GB at 119 s. The bakeoff embedded segments of about 15 s, so its numbers never
  met this. A take of two minutes needs about 12 GB allocated and 13 GB reserved in one pass.

## What this gives the engine profile (`vram_need_mb`)

Section 4 checks that free VRAM ≥ need + 1 GB before it loads a group. The QA group's need is the resident
models plus the larger of the two peaks (the ops run one at a time) plus the CUDA context:

- **takes up to 60 s: about 9 GB** (3.4 GB resident + 4.4 GB for transcription with word times + 0.45 GB
  context, rounded up for the allocator's reserve). This is the value proposed for `vram_need_mb`, 9000,
  with a limit on how long a take may be for it to hold;
- takes of 90 s: about 9.5 GB; of 120 s: about 13–14 GB, because embedding in one pass dominates.

The plan's ASSUMEd ~5 GB was an underestimate, because of word times and long takes.

**Proposed (for the lead):** bound the embedding's memory by embedding audio longer than 30 s in windows of
at most 30 s and taking the mean of their L2-normalised embeddings, re-normalised. Its cosine to the one-pass
embedding is 0.999 at 60 s and 0.994 at 119 s (the table above); audio up to 30 s is unchanged, so every
number of the bakeoff's evidence is unchanged. The need would then stay about 9–9.5 GB for any take length
(transcription's peak grows slowly). It changes what the similarity of a long take means, slightly, so it is
not built without the lead's decision.
