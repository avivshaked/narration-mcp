# Spikes (h) and (i), Qwen half: offline loading, VRAM and load times

Design section 20, Phase 0: **(h)** VRAM and load times; **(i)** offline loading from a snapshot directory
named by its commit SHA. This folder covers the two Qwen3-TTS models. The QA half of (h) (Whisper and
WavLM) belongs to WP22.

- `run.py`: the spike, run in the Qwen worker's venv under the GPU lock. It drives the worker's own engine
  (`narration_qwen3tts.engine`), so it measures the code the service runs.
- `results.json`: its output from 2026-09-26. It contains no paths and no GPU name. Machine facts are
  kept in a gitignored local file.
- `repro_replicate_pad.py` + `repro_replicate_pad.json`: a model-free reproduction of the failure the
  first run hit (below).
- `recheck_pcm16.py` + `recheck_pcm16.json`: the d2 re-design comparison, redone after a measuring
  error. It needs no GPU (see "Correction").

## Method

1. Set the worker's start-up environment before torch loads: `HF_HUB_OFFLINE=1`,
   `TRANSFORMERS_OFFLINE=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and thread caps of 8.
2. Install a guard that refuses every socket connection and name lookup leaving the machine, and records
   each attempt. Loopback is allowed and counted separately.
3. Check that each snapshot folder is named by its 40-hex revision, then load it with the determinism
   switches of section 10.1 on and every audio-changing setting explicit (`generation_config.json`'s
   values, `max_new_tokens` 8192).
4. **Base:**
   - prepare the d2 voice;
   - render two ladder paragraphs of the service's own corpus (`narration-en.v1` ladder-150 and
     ladder-560);
   - render once more with a test-only cap of 24 tokens.
5. **VoiceDesign:**
   - design the service's canary voice (`material/canary/canary.v1`);
   - re-design the bakeoff's d2 clip from its description and seed 2001, with the switches off and then
     on;
   - compare the 16-bit PCM with the clip.
6. Load each model a second time, then re-hash every file against the models manifest.

## Results (KNOW, 2026-09-26: one 24 GB consumer NVIDIA GPU, shared; torch 2.11.0+cu128, CUDA 12.8)

| | Base | VoiceDesign |
|---|---|---|
| Offline load from the SHA-named folder | yes, **0** network attempts (0 loopback) | yes, **0** network attempts |
| Load time: first / second in the process | 2.9 s / 2.3 s | 3.6 s / 2.3 s |
| Reserved VRAM after load (allocator) | 4134 MB | 4146 MB |
| Peak reserved while rendering | 4824 MB (9.2 s of audio), **5304 MB** (31.3 s) | 4508 MB (12.5 s) |
| After unload | 38 MB reserved | 38 MB reserved |
| Weights re-hashed against the manifest | 11 files, 0 mismatched | 11 files, 0 mismatched |

- **CUDA context:** about 450 MB, measured as the device-level change when the process started CUDA. That
  is an estimate, because other jobs share the device.
- **The VRAM need for the Qwen group is therefore about 5.8 GB for a 30 s render**: 5.3 GB peak plus the
  context. It grows slowly with length: the allocator's peak rose by about 0.2 MB per generated token
  between the two ladder renders. The plan's ASSUMEd 6–8 GB was an overestimate. Section 4's check
  (free VRAM ≥ need + 1 GB) now has a measured number, so `vram_need_mb` in the engine profile can be
  about 6000 (BELIEVE: an 8192-token render would add about 1.6 GB of cache, which has not been
  measured).
- **Load times were measured with a warm OS file cache.** The weights had been read shortly before, so
  2–4 s is not a cold start from disk. A cold load was not measured, because emptying the OS file cache
  needs administrator rights.
- **Speed:** RTF about 2.0, meaning generation took about twice the audio's length, on the shared GPU.
- **Token cap:** the codec runs at exactly **12.5 frames per second** of audio. Every render generated
  `new_tokens` = talker steps − 1, because the last sampled token (the end token, or the cut-off token
  at the cap) is never decoded. With `max_new_tokens` 24, the engine reported `hit_token_cap: true` and
  23 frames (1.84 s). So 8192 tokens is about **655 s** of audio (plan.md section 1.3 item 1 estimated
  680 s at 12 Hz).
- **The d2 re-design is bit-identical to the bakeoff's clip** (see the correction below;
  `recheck_pcm16.json`).
  - Converted to 16 bits the way the bakeoff stored it, all 289,920 samples (151 frames) are equal.
  - The renders with the switches off and on hashed identically, so the switches made no difference.
  - This holds although this process had `CUBLAS_WORKSPACE_CONFIG` set and the bakeoff's did not, and
    although the bakeoff rendered d2 after d1's two takes in the same process.
  - The acceptance check found the same for the Base clone path (`spikes/acceptance-wp20/`).

### Correction (2026-09-26): `results.json` reports a false mismatch

`results.json` says `pcm16_bit_identical_to_bakeoff_clip: false` with `pcm16_max_abs_diff` 23,935. That is
a measuring error in the first version of `run.py`, not a difference in the audio.

- **The error:** `run.py` read its own FLOAT WAV with `soundfile.read(..., dtype="int16")`.
  - libsndfile does not scale float to integer on read by default, so every sample came back as −1, 0
    or 1.
  - 23,935 is the clip's peak (23,936) less one.
- **The fix:** `run.py` now converts the float render with `gpu_spike_common.as_bakeoff_pcm16`. That
  helper repeats the bakeoff's own write (`soundfile.write` with the default WAV subtype, PCM_16) in
  memory.
- **The re-check:** `recheck_pcm16.py` repeats the comparison from the renders this run saved, without the
  GPU. Its output, `recheck_pcm16.json`, replaces the two `pcm16_*` fields of `results.json`.
  - Its float hashes equal those in `results.json`, so it compared the same renders.
  - Both renders are bit-identical to the clip.
- `results.json` is left as that run wrote it.
- **No warnings** were raised by `use_deterministic_algorithms(warn_only=True)` during generation or
  decoding.

## A finding: deterministic mode breaks the voice prompt's encode (KNOW)

The first run failed in `prepare_voice` (`create_voice_clone_prompt`) with
`RuntimeError: _unsafe_index found unexpected index type Float`.

**Cause:**
- With `torch.use_deterministic_algorithms(True)`, even warn-only, torch 2.11 sends CUDA `replicate`
  padding through `torch._decomp.decompositions._replication_pad`.
- That function's `pw_cast_for_opmath` wrapper casts every tensor argument to the computation dtype.
- The Mimi encoder inside Qwen's speech tokenizer (transformers 4.57.3, `MimiModel.downsample`, the only
  replicate-padded convolution on this path) passes its right padding as a 0-dim integer tensor.
- The index becomes float, and the lookup fails.

`repro_replicate_pad.py` shows it without a model: the call fails **exactly** when deterministic
algorithms are on and the pad amount is a tensor, in float32 and bfloat16 alike.

**What the engine does:** it turns deterministic algorithms off for the whole `create_voice_clone_prompt`
call (the voice prompt's encode), and restores the previous mode afterwards, on every exit path
(`narration_qwen3tts.engine._deterministic_algorithms_suspended`).
- Padding's forward pass is a gather, so it is deterministic either way.
- cuDNN's deterministic flag, TF32 and the cuBLAS workspace setting are not touched.
- Generation and decoding run with every switch the caller set.

Spike (d) showed renders are bit-exact with this in place (ADR 0002, accepted). The lead notes the
exemption in section 10.1's list of switches.

Also seen (KNOW, from the installed source):
- Importing qwen-tts imports its 25 Hz tokenizer (`qwen_tts/core/tokenizer_25hz/vq/speech_vq.py`), which
  imports the `sox` package. `sox/__init__.py` runs `os.popen('sox -h')`, a **shell** command, and logs
  "SoX could not be found!" when that prints nothing.
- The worker uses neither the 25 Hz tokenizer nor SoX.
- The popen reads the child's stdout through its own pipe, and the shell's error goes to stderr, so the
  protocol stream is safe.
- **But a worker's first `load` runs whatever `sox` is first on `PATH`.** The worker imports qwen-tts
  only in `load` (`narration_qwen3tts.engine`), not when it starts, so `hello` runs no shell. That is
  third-party behaviour the worker cannot turn off without patching the library (section 17.9's
  "argument lists, never a shell" holds for our code). The lead's ruling: WP30 starts workers with the
  current directory set to their own project folder and, on Windows, `NoDefaultCurrentDirectoryInExePath=1`,
  so a `sox` in the current directory is never run.
