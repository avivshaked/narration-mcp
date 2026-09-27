# Spikes (h) and (i), QA half: offline loading, VRAM and load times

Design section 20, Phase 0: **(h)** VRAM and load times; **(i)** offline loading from a snapshot directory
named by its commit SHA. This folder covers the QA group's GPU models, Whisper-large-v3 and
WavLM-base-plus-sv. The Qwen half is `spikes/h-i-qwen-load`. The CTC aligner runs on the CPU; its memory is
`spikes/aligner-memory`.

- `run.py`: the spike, run in the QA worker's venv under the GPU lock. It drives the worker's own classes
  (`narration_worker_qa.asr.WhisperAsr`, `narration_worker_qa.sv.WavLmSv`) with the worker's determinism
  switches, so it measures the code the service runs.
- `results.json`: its output (the run's time is in `ran_at`, UTC). It contains no paths, no GPU name and no
  text: the audio is named by the bakeoff's ids (the d2 clone take, seed 1).
- `wavlm_weights.py` and `wavlm_weights.json`: WavLM-base-plus-sv's two snapshots compared tensor by tensor on
  the CPU (which revision to pin).

## Method

1. Set the worker's start-up environment before torch loads (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, thread caps of 8), and install a guard that refuses every socket
   connection and name lookup leaving the machine, recording each attempt.
2. Check that each snapshot folder is named by its 40-hex revision; apply the QA worker's switches (TF32 off,
   cuDNN deterministic, benchmark off, deterministic algorithms warn-only); load Whisper (float16, eager
   attention) and WavLM (float32) on `cuda:0`, both resident, as the QA group is.
3. Transcribe (English, word times, long-form: what the job engine sends; five beams, not conditioned: ADR 0004)
   and embed (as the worker embeds: DC-15's windows over 30 s) a 40 s slice and the whole of the bakeoff's d2
   clone take, seed 1 (119 s), recording the allocator's peaks; embed both again on the CPU (the canary's
   path) and compare the embeddings.
4. Measure how the peaks grow with length, from 10 s to 119 s cut from the start of the take: transcription
   with and without word times; embedding as the worker embeds, and in one pass over the whole clip (what it
   did before DC-15), with the cosine of the two.
5. Unload, load each model a second time, and re-hash every weight file against the models manifest.

## Results (KNOW: one 24 GB consumer NVIDIA GPU, shared; torch 2.11.0+cu128, CUDA 12.8)

The run of 2026-09-27, 12:57 UTC (`results.json`), with the worker as built: five beams (DC-14) and windowed
embedding (DC-15).

| | Whisper-large-v3 | WavLM-base-plus-sv |
|---|---|---|
| Offline load from the SHA-named folder | yes, **0** network attempts | yes, **0** network attempts |
| Load time: first / second in the process | 3.2 s / 3.3 s | 0.33 s / 0.27 s |
| Reserved VRAM added by the load | 3184 MB | 420 MB |
| Weights re-hashed against the manifest | 11 files, 0 mismatched | 3 files, 0 mismatched |

- **The group resident:** 3331 MB allocated, 3606 MB reserved.
- **CUDA context:** about 440 MB (the device-level change when the process started CUDA; an estimate, since
  other jobs share the device).
- **Load times were measured with a warm OS file cache**, as in the Qwen half.
- **Speed:** transcription with word times runs at 0.20–0.23 × real time (40 s in 9.4 s, 119 s in 23.9 s);
  embedding takes 0.18 s for 40 s and 0.32 s for 119 s on the GPU. On the CPU, embedding 40 s takes about
  2 s, the first call included (it loads the CPU copy), and 119 s about 6 s; the CPU's embeddings equal the
  GPU's to 6 decimal places (cosine 1.000000), windows and all.

### What the peaks grow with (allocated above the resident models)

| Audio | transcribe, no word times | transcribe, word times | embed as built (DC-15) | embed in one pass | cosine, as built to one pass |
|---|---|---|---|---|---|
| 10 s | 1247 MB | 4851 MB | 126 MB (1 window) | 126 MB | 1 |
| 20 s | 1292 MB | 5851 MB | 288 MB (1 window) | 288 MB | 1 |
| 30 s | 1308 MB | 6450 MB | 602 MB (1 window) | 602 MB | 1 |
| 60 s | 1309 MB | 6450 MB | 602 MB (2 windows) | 2234 MB | 0.99904 |
| 90 s | 1310 MB | 6451 MB | 602 MB (3 windows) | 4894 MB | 0.99647 |
| 119 s | 1310 MB | 6452 MB | 595 MB (4 windows) | 8446 MB (13.3 GB reserved) | 0.99365 |

- **Transcription's peak stops growing at 30 s**, Whisper's window: sequential long-form decodes one 30 s
  window at a time. Five beams cost about 1.3 GB without word times.
- **Word times cost 3.6 to 5.1 GB more** (10 s to 30 s and over). BELIEVE: asking `generate` for token
  timestamps turns on `output_attentions`, and the encoder then keeps the self-attention maps of all 32
  layers (32 × 20 heads × 1500² × 2 bytes = 2.9 GB), while only the decoder's cross-attention is needed for
  word times; the rest grows with the beams' decoded tokens. Not changed here: the job engine needs word
  times, and the cost is bounded.
- **Embedding in one pass grows with the square of the length** (WavLM's self-attention over the whole
  clip): 8.4 GB at 119 s. **Windowed embedding (DC-15) keeps it at the 30 s figure, 0.6 GB, for any
  length.** Its cosine to the one-pass embedding is 0.999 at 60 s and 0.994 at 119 s. Audio of up to 30 s is
  embedded exactly as before (one window), and every segment of the bakeoff's evidence was shorter than
  that, so no evidence number changes (`spikes/acceptance-wp22`).

## What this gives the engine profile (`vram_need_mb`)

Section 4 checks that free VRAM ≥ need + 1 GB before it loads a group. The QA group's need is the resident
models plus the larger of the two ops' peaks (they run one at a time) plus the CUDA context. With DC-15 both
peaks are bounded, so **one need holds for a take of any length**:

- the largest peak is transcription with word times: 9814 MB allocated, the resident models included, and up
  to 10.6 GB reserved (step 3 of the method; the allocator's cache);
- plus the CUDA context, about 440 MB: **about 11.0 GB on the device**.

**Proposed: `vram_need_mb` = 11500** (the measured 11.0 GB and a margin of about 4 % for the allocator's
fragmentation). The plan's ASSUMEd ~5 GB was an underestimate: word times and five beams cost most of it.

Two ways to lower it, neither built (the lead's call, if VRAM matters more):

- greedy decoding not conditioned (ADR 0004's option B) peaks about 3 GB lower, but its WER differs from the
  evidence by up to 4.3 points on one take;
- keeping only the decoder's cross-attention for word times would save most of the encoder's 2.9 GB (BELIEVE;
  it needs a change to how the worker asks transformers for word times, and a re-run of the acceptance).
