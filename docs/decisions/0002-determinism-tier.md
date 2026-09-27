# ADR 0002: The Qwen3-TTS clone path is `bit_exact` (spike d)

- Status: **accepted** (the lead, 2026-09-26); proposed by WP20 (spike d). The lead edits section 10.1 at
  merge (see Consequences).
- Date: 2026-09-26
- Design sections: 10.1 (the determinism layers, the tier and the canary gate), 6 (`EngineProfile.tier`),
  20 (d)
- Evidence: [spikes/d-repeat-test](../../spikes/d-repeat-test/README.md),
  [spikes/h-i-qwen-load](../../spikes/h-i-qwen-load/README.md),
  [spikes/acceptance-wp20](../../spikes/acceptance-wp20/README.md)

## Context

Section 10.1 has two tiers:

- **`bit_exact`:** an evicted take comes back with the same id, and the canary gate compares hashes.
- **`similar`:** a take id is stable only while cached, and the canary gate compares similarity only.

Bit-exact output within one process was KNOW only for VoiceDesign. For the Base clone path, which the
service uses for every take, it was BELIEVE, within one process and across processes. The contracts
already treat the tier as an observation. `EngineProfile.tier` is "the outcome of the repeat test on
this machine", is unhashed, and is set after the pin.

## Decision

1. **The tier is `bit_exact`** for `qwen3-base-1.7b.p1` (the clone path) and `qwen3-design-1.7b.p1`, on
   the stack pinned in `workers/qwen3tts`:
   - qwen-tts 0.1.1, transformers 4.57.3 and torch 2.11.0+cu128;
   - bf16 with `sdpa`, batch size 1.

   The service is built for `bit_exact` first. The `similar` path still exists (plan section 5) for a
   machine where the repeat test fails.
2. **Keep every section 10.1 switch**, although on this stack they changed nothing:
   - `pinned`, `no_det_algos` and torch's defaults gave the same bytes;
   - the switches cost about 5–13 % of generation time.

   They are cheap insurance for GPUs, drivers and kernels not tested here (BELIEVE: other GPUs can choose
   other cuBLAS and cuDNN kernels). Dropping them would be a design change.
3. **One exemption, which the design's switch list states.** The worker turns
   `torch.use_deterministic_algorithms` off for the whole `create_voice_clone_prompt` call (the voice
   prompt's encode and speaker embedding), then restores the previous mode, on every exit path.
   - Torch 2.11's deterministic replicate padding cannot take the tensor-valued pad of the Mimi encoder,
     so with the switch on, encoding a voice fails.
   - Padding's forward pass is a gather, which is deterministic either way.
   - Spike (d) prepared the voices afresh in every process, in two orders, and every render matched.
   - This is `narration_qwen3tts.engine._deterministic_algorithms_suspended`.
4. **The tier is decided on the installing machine, not shipped** (proposal for WP32).
   - `narration-admin engine pin` should repeat a short version of spike (d): render the canary twice in
     one worker process and once in a fresh one, then compare the float32 hashes.
     - All equal: set `tier: "bit_exact"`.
     - Otherwise: set `tier: "similar"`, and say so in `doctor`.
   - That takes a few seconds and one extra worker start.
   - This machine's result is evidence, not a promise: section 10.1 promises nothing across a GPU,
     driver or CUDA change.
5. **Identical audio must make an identical file.** The worker writes its float32 WAV itself
   (`narration_qwen3tts.wav`): the format, the `fact` chunk and the data, and nothing else.
   - `soundfile.write` adds a `PEAK` chunk holding the time of writing. Two renders with identical
     samples then have different bytes and different sha256s, which would defeat both the take id's
     stability and the canary gate's hash comparison.
   - Any other worker (and any file the service hashes) must avoid the same trap.
   - Note (WP16's follow-ups, lead-authorised): the writer is now `narration_worker.wav`, shared by every
     worker; it writes the same bytes as `narration_qwen3tts.wav` did, which is gone.

## Evidence

**KNOW** means measured here, with the script and its output saved. The machine is one 24 GB consumer
NVIDIA GPU, shared with other jobs, with CUDA 12.8 and cuDNN 9.19, on 2026-09-26.

| Claim | Label | Where |
|---|---|---|
| **Four clone items gave one hash each, 10 renders per item and 40 in all:** d2 × ladder-080, ladder-150 and cal-02, and d4 × ladder-080. The renders ran in 7 processes, in forward, reverse and shuffled order, with voices prepared in either order | KNOW | `spikes/d-repeat-test/results.json`, `renders.csv` |
| **The switches changed no output:** `pinned`, `no_det_algos` (deterministic algorithms off) and `defaults` (torch's defaults, no `CUBLAS_WORKSPACE_CONFIG`) produced identical bytes | KNOW | same (`pinned_equals_defaults`) |
| `warn_only` raised no non-determinism warning during generation or decoding: 0 warnings in 40 renders, 16 of them with deterministic algorithms warn-only; none in spike h+i's renders | KNOW | `spikes/d-repeat-test/results.json` (`summary.warnings`, from the per-render warnings the run recorded); `spikes/h-i-qwen-load/results.json` (`warnings` per render) |
| **The switches cost about 5–13 % of generation time.** Means of 3–4 renders, on a shared GPU, so indicative only | KNOW | `spikes/d-repeat-test/renders.csv` (`gen_s`) |
| Renders through the worker protocol equal the bakeoff's takes sample for sample, once converted to the bakeoff's 16 bits (d2 and d4, n07 and n08). The bakeoff ran a day earlier, in its own venv with the same pins and no switches | KNOW | `spikes/acceptance-wp20/results.json` (`identical_to_bakeoff_pcm16`) |
| The VoiceDesign re-design of the d2 clip (seed 2001) equals the bakeoff's clip sample for sample, with the switches off or on | KNOW | `spikes/h-i-qwen-load/recheck_pcm16.json` |
| **Through the protocol**, the canary gate text rendered twice with the same seed gave identical samples. With `soundfile.write` the two files differed only in the `PEAK` chunk's timestamp. With the worker's own writer they are identical files | KNOW | `workers/qwen3tts/tests/test_gpu.py` (run under the lock, 2026-09-26: failed, then passed after the fix); `tests/test_wav.py` |
| Deterministic algorithms make the voice-prompt encode fail (torch 2.11 `_replication_pad` with a tensor pad). It is reproduced without a model | KNOW | `spikes/h-i-qwen-load/repro_replicate_pad.json` |
| The same holds on other GPUs, drivers, or CUDA and cuDNN versions | not claimed | hence decision 4 |
| Renders longer than 12 s repeat too. A 31 s render was made once, not repeated | BELIEVE | nothing on the path depends on length except the KV cache |

## Consequences

- **WP32:**
  - the canary gate compares hashes first (section 10.1, layer 4);
  - `engine pin` runs the short repeat test and records `tier` (decision 4);
  - the hash is over the worker's file, which now repeats byte for byte (decision 5).
- **WP40:** "rendering twice and in a fresh process behaves as the tier says" is tested against
  `bit_exact` on this machine.
- **Design, section 10.1:**
  - replace "BELIEVE … the Phase 0 repeat test (d) decides" with this result;
  - add decision 3's exemption to the switch list: it covers the whole `create_voice_clone_prompt` call.

  That is a lead edit; nothing in the contracts changes.
- A caller can rely on an evicted take coming back with the same id, on a machine whose pin recorded
  `bit_exact`.

## Alternatives considered

- **Declare `similar` everywhere.** This is simpler and always safe. But it throws away a property that
  holds here: stable take ids after eviction, and a hash-only canary gate with no model swap.
  Rejected on the evidence.
- **Drop the switches to save 5–13 %.** They made no difference here, but that is one machine. The saving
  is small against a silent loss of reproducibility elsewhere. Not proposed. The owner may revisit it
  with the repeat test's results from other machines.
- **Keep deterministic algorithms on during the encode by patching Mimi's padding** (a Python int
  instead of a tensor). Rejected: it patches a pinned third-party library to work around a torch bug, for
  no gain in determinism, since the encode was reproducible anyway.

## What would reverse this

- A repeat test on the installing machine that differs, which sets `similar` there by decision 4.
- A torch, CUDA or qwen-tts upgrade whose repeat test differs. That upgrade is a new engine profile
  anyway.
- `warn_only` reporting a non-deterministic op on the generation or decode path.
- Batch size above 1, which the design does not allow.
