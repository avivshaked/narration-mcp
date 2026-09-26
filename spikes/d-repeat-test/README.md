# Spike (d): the repeat test on the Base clone path, which sets the determinism tier

Design sections 10.1 and 20 (d). Bit-exact output was KNOW only for VoiceDesign within one process
(section 1, row B). For the clone path the service uses, it was BELIEVE, within one process and across
processes. This spike decides it. The decision is [ADR 0002](../../docs/decisions/0002-determinism-tier.md).

- `run.py`: the spike. The parent process starts one child process per run in the plan, then compares.
- `results.json`: the summary, the plan, the switches in force, and the tier.
- `renders.csv`: one row per render, with its sha256 over the float32 samples as generated.
- `summary.warnings` was added after the run, from the warnings each render recorded in the run's local
  files, with `run.py --compare-only` (no GPU; `summary_recomputed_at`). Nothing else in the summary
  changed.

## Method

- **Items (4):**
  - voice d2 × the service's `narration-en.v1` paragraphs ladder-080, ladder-150 and cal-02 (seeds 11, 12
    and 13);
  - voice d4 × ladder-080 (seed 11).
  - These are 5 to 12 s of speech, 62 to 148 codec frames.
- **Every render goes through the worker's engine** (`narration_qwen3tts.engine`), as the worker runs it:
  - the voice prompt is made once per process (with the documented suspension of deterministic
    algorithms during its encode);
  - then `seed_everything` with the seed;
  - then `generate_voice_clone` with every audio-changing setting explicit (`max_new_tokens` 8192, the
    ceiling, as the call's cap; `non_streaming_mode` false), in bf16 with `sdpa`.
- **Three environments, each in its own processes**, because `CUBLAS_WORKSPACE_CONFIG` can only be set
  before CUDA starts:
  - `pinned`: section 10.1's switches (TF32 off, cuDNN deterministic, benchmark off, deterministic
    algorithms warn-only, `CUBLAS_WORKSPACE_CONFIG=:4096:8`);
  - `no_det_algos`: the same, with `use_deterministic_algorithms` off;
  - `defaults`: torch's defaults and no `CUBLAS_WORKSPACE_CONFIG`, as the bakeoff ran.
- **Order changes:**
  - within each variant's first process, every item is rendered, then rendered again in reverse order;
  - further fresh processes render in another order and prepare the voices in another order (d4 before
    d2).
  - 7 processes and 40 renders in all (each item rendered 10 times).
- The GPU was shared with other jobs throughout, as it is in use.

## Results (KNOW, 2026-09-26: one 24 GB consumer NVIDIA GPU; torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19)

| Variant | Renders | Processes | Same bytes within a process | Same bytes across processes |
|---|---|---|---|---|
| `pinned` | 16 | 3 | **yes**, all 4 items | **yes**, all 4 items |
| `no_det_algos` | 12 | 2 | yes | yes |
| `defaults` | 12 | 2 | yes | yes |

- **Each item produced exactly one sha256 across all its 10 renders (40 in all).** The three variants
  produced the *same* bytes as each other, so on this stack the switches do not change the clone path's
  output at all.
- The order of the renders and of the voice preparation made no difference.
- `use_deterministic_algorithms(warn_only=True)` raised no warning during generation or decoding, so it
  reported no non-deterministic op on the path: 0 warnings in all 40 renders, 16 of them with it on
  (`summary.warnings`).
- **Cost (`renders.csv`, `gen_s`):**
  - with the switches, generation took about 5–13 % longer than with torch's defaults;
  - for example, ladder-150 took 18.0 s `pinned`, 16.7 s `no_det_algos` and 16.4 s `defaults` (means of
    3–4 renders);
  - the GPU was shared, so this is indicative only.
- **Therefore the tier is `bit_exact`** for the Base clone path on this machine (ADR 0002).

## Limits: what this does not show

- **Other GPUs, drivers, CUDA or cuDNN versions.** The design promises nothing across them (section 10.1),
  and the result is this machine's. `EngineProfile.tier` is an observation made on the installing machine,
  so ADR 0002 proposes that `engine pin` repeats a short version of this test there.
- **Long renders.** The longest item here is 12 s. A 31 s render (spike h+i) was made once, not repeated.
- **Batch size above 1.** The design fixes batch size at 1.

## Corroboration: the bakeoff's files (KNOW)

The same holds against renders made a day earlier, in another process tree, by the bakeoff's own venv.
That venv has the same pins and no determinism switches.

- Stored at the bakeoff's 16 bits, the worker's renders equal the bakeoff's files **sample for sample**:
  - the acceptance renders of n07 and n08 in d2 and d4 (`spikes/acceptance-wp20/`);
  - the VoiceDesign re-design of the d2 clip (`spikes/h-i-qwen-load/recheck_pcm16.json`).
- The 16-bit conversion is the bakeoff's own: `soundfile.write` with the default WAV subtype.
- The float renders underneath cannot be compared, because the bakeoff did not keep them.
