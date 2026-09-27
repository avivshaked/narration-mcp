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
   and embed (as the worker embeds: one pass up to 60 s, else DC-15's windows) a 40 s slice and the whole of
   the bakeoff's d2 clone take, seed 1 (119 s), one after another as the worker runs them, recording the
   allocator's peaks and what stays reserved after each op; embed both again on the CPU (the canary's path)
   and compare the embeddings.
4. Measure how the peaks grow with length, from 10 s to 119 s cut from the start of the take: transcription
   with and without word times; embedding as the worker embeds, in one pass over the whole clip, and in windows
   of 30 s (DC-15 as first approved) and 60 s (as amended), with each one's cosine to the one-pass embedding.
5. Unload, load each model a second time, and re-hash every weight file against the models manifest.

## Results (KNOW: one 24 GB consumer NVIDIA GPU, shared; torch 2.11.0+cu128, CUDA 12.8)

The run of 2026-09-27, 13:42 UTC (`results.json`), with the worker as built: five beams (DC-14), 60 s windows
(DC-15 as amended), and the allocator's cache returned after each GPU op (`narration_worker_qa.gpu`). Earlier
runs the same day (12:57, 13:11 and 13:29 UTC) measured every peak below to the same megabyte.

| | Whisper-large-v3 | WavLM-base-plus-sv |
|---|---|---|
| Offline load from the SHA-named folder | yes, **0** network attempts | yes, **0** network attempts |
| Load time: first / second in the process | 2.6 s / 2.8 s | 0.41 s / 0.24 s |
| Reserved VRAM added by the load | 3184 MB | 420 MB |
| Weights re-hashed against the manifest | 11 files, 0 mismatched | 3 files, 0 mismatched |

- **The group resident:** 3331 MB allocated, 3606 MB reserved. After unload: 56 MB reserved.
- **CUDA context:** 436 MB in this run, 448 MB in the earlier ones (the device-level change when the process
  started CUDA; an estimate, since other jobs share the device).
- **Load times were measured with a warm OS file cache**, as in the Qwen half.
- **Speed:** transcription of 119 s with word times takes about 22 s (0.19 × real time) and embedding it 0.5 s
  on the GPU; on the CPU, embedding 40 s takes about 2.5 s, the first call included (it loads the CPU copy).
  The CPU's embeddings equal the GPU's to 6 decimal places (cosine 1.000000), windows and all. Step 3 of this
  run took longer, 38 s for the whole take, while a CPU test suite ran beside it; the 13:29 run took 23.5 s.

### What the peaks grow with (allocated above the resident models)

| Audio | transcribe, no word times | transcribe, word times | embed, one pass | embed, 30 s windows | embed, 60 s windows (as built) |
|---|---|---|---|---|---|
| 10 s | 1247 MB | 4851 MB | 126 MB | 126 MB (1), cosine 1 | 126 MB (1), cosine 1 |
| 20 s | 1292 MB | 5851 MB | 288 MB | 288 MB (1), cosine 1 | 288 MB (1), cosine 1 |
| 30 s | 1308 MB | 6450 MB | 602 MB | 602 MB (1), cosine 1 | 602 MB (1), cosine 1 |
| 60 s | 1309 MB | 6450 MB | 2234 MB | 602 MB (2), cosine 0.999039 | 2234 MB (1), cosine 1 |
| 90 s | 1310 MB | 6451 MB | 4894 MB | 602 MB (3), cosine 0.996473 | 1290 MB (2), cosine 0.998083 |
| 119 s | 1310 MB | 6452 MB | 8446 MB (13.3 GB reserved) | 595 MB (4), cosine 0.993649 | 2197 MB (2), cosine 0.997583 |

The windows' count is in brackets; each cosine is to the one-pass embedding of the same audio.

- **Transcription's peak stops growing at 30 s**, Whisper's window: sequential long-form decodes one 30 s
  window at a time. Five beams cost about 1.3 GB without word times.
- **Word times cost 3.6 to 5.1 GB more** (10 s to 30 s and over). BELIEVE: asking `generate` for token
  timestamps turns on `output_attentions`, and the encoder then keeps the self-attention maps of all 32
  layers (32 × 20 heads × 1500² × 2 bytes = 2.9 GB), while only the decoder's cross-attention is needed for
  word times; the rest grows with the beams' decoded tokens. Not changed here: the job engine needs word
  times, and the cost is bounded.
- **Embedding in one pass grows with the square of the length** (WavLM's self-attention over the whole
  clip): 8.4 GB at 119 s. **With 60 s windows (DC-15 as amended) it is bounded by the 60 s figure, 2.2 GB,
  for any length**, below transcription's peak. Its cosine to the one-pass embedding is 0.998 at 90 s and
  0.9976 at 119 s. Audio of up to 60 s is embedded exactly as before, in one pass.
- **Why 60 s, not 30 s** (`spikes/acceptance-wp22/windows.json`): the bakeoff's voicelock evidence compares
  clips of up to 37 s, embedded in one pass. With 30 s windows 3 of its 52 rows moved by more than the
  acceptance's 0.002 (at most 0.0057); with 60 s windows every row is reproduced, and the acceptance passes.
  Two runs show it: `spikes/acceptance-wp22/results.json` (2026-09-27 13:32–13:41 UTC) was made with 60 s
  windows but **before** the release after each op (next section) was added; the GPU test
  `test_acceptance_matches_the_bakeoff_eval_s11_1`, which runs the same script, passed on the final code,
  release included (13:46–13:53 UTC; its output was not kept).

### What stays reserved between ops

PyTorch's caching allocator keeps the blocks an op freed. In the 13:29 UTC run, before the worker returned
them (its `results.json` is in commit `05114c0`, "DC-15 at 60 s passes the acceptance; cosines measured":
`models.sv.embed_cuda.whole_take` and `models.asr.transcribe.whole_take`), the whole take transcribed and
then embedded left **14.2 GB reserved for 5.6 GB allocated**: the
allocator could not reuse Whisper's freed blocks for WavLM's shapes, so the process's footprint on the shared
GPU grew past the group's need. The worker now calls `torch.cuda.empty_cache()` after each GPU op
(`narration_worker_qa.gpu`). In this run:

| Op (step 3, in the worker's order) | peak allocated | peak reserved | reserved after the op |
|---|---|---|---|
| transcribe 40 s | 9814 MB | 10348 MB | 3640 MB |
| embed 40 s | 4395 MB | 4896 MB | 3640 MB |
| transcribe 119 s | 9816 MB | 10452 MB | 3640 MB |
| embed 119 s | 5567 MB | 6434 MB | 3640 MB |

So the footprint between ops is the resident models' (3.6 GB), and during an op that op's own peak: at most
10.5 GB reserved, in transcription. The 13:29 run (commit `05114c0`), without the release, reached 11.4 GB
reserved in transcription and 14.2 GB in the embedding after it. Results do not change: the kernels are deterministic.

## What this gives the engine profile (`vram_need_mb`)

Section 4 checks that free VRAM ≥ need + 1 GB before it loads a group. The QA group's need is the largest
footprint of any op (they run one at a time, and the cache is returned between them) plus the CUDA context.
Both ops' peaks are bounded (transcription by Whisper's 30 s window, embedding by DC-15's 60 s windows), so
**one need holds for a take of any length**:

- the largest footprint is transcription with word times: 9.8 GB allocated, 10.5 GB reserved, the resident
  models included;
- plus the CUDA context, 436–448 MB: **about 10.9 GB on the device**.

**`vram_need_mb` = 11500** (the lead's decision, on these numbers): the measured 10.9 GB and a margin of about
5 % for the allocator. The plan's ASSUMEd ~5 GB was an underestimate: word times and five beams cost most of it.

It holds for this configuration only. Re-run this spike before changing any of: the models or their
revisions, Whisper's dtype or attention implementation, the decoding (ADR 0004), word times, `WINDOW_S`, or
the release after each op.

Two ways to lower it, neither built (the lead's call, if VRAM matters more):

- greedy decoding not conditioned (ADR 0004's option B) peaks about 3 GB lower, but its WER differs from the
  evidence by up to 4.3 points on one take;
- keeping only the decoder's cross-attention for word times would save most of the encoder's 2.9 GB (BELIEVE;
  it needs a change to how the worker asks transformers for word times, and a re-run of the acceptance).
