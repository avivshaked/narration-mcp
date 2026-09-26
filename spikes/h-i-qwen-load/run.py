"""Spikes (h) and (i), Qwen half: load Base and VoiceDesign offline from SHA-named snapshots; VRAM and times.

Design section 20, Phase 0: (h) VRAM and load times; (i) offline loading from a snapshot directory named by
its commit SHA. The QA half of (h) (Whisper and WavLM) is WP22's.

What it does, in one process, with the GPU lock held (AGENTS.md section 5):

1. sets the worker's start-up environment (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, CUBLAS_WORKSPACE_CONFIG,
   thread caps) and installs a guard that refuses every network connection and name lookup, recording any
   attempt;
2. checks that each snapshot folder is named by its 40-hex revision, and loads it through the worker's
   engine (``narration_qwen3tts.engine``) with the determinism switches on and every audio-changing setting
   explicit (section 10.1);
3. Base: prepares the d2 voice and renders two of the service's ladder paragraphs (about 150 and 560 spoken
   characters), recording time, allocator peaks and the token-cap facts; then renders once with a test-only
   cap of 24 tokens, which must report ``hit_token_cap``;
4. VoiceDesign: designs the service's canary voice (``material/canary``), and re-designs the bakeoff's d2
   clip from its description and seed, with the switches off (the bakeoff's conditions) and on, comparing
   the 16-bit PCM with the clip;
5. loads each model a second time (warm file cache), and re-hashes the weights against the manifest.

Run from the checkout, in the Qwen worker's venv::

    workers/qwen3tts/.venv/Scripts/python.exe spikes/h-i-qwen-load/run.py [--material <dir>]

Writes ``spikes/h-i-qwen-load/results.json`` (published: no paths, no GPU name) and, under
``.dev/spikes/h-i/``, the audio and ``local-facts.json``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
OUT = common.DEV_SPIKES / "h-i"
DEVICE = "cuda:0"
SEED = 1234
TEST_CAP = 24


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--material", help="the service's material folder (default: <checkout>/material)")
    parser.add_argument("--no-verify", action="store_true", help="skip re-hashing the weights")
    args = parser.parse_args()

    common.prepare_process_env()
    guard = common.NetworkGuard().install()
    material = common.material_root(args.material)
    texts = common.calibration_paragraphs(material)
    canary = common.canary(material)

    import numpy as np
    import pynvml
    import soundfile as sf
    import torch
    from narration_qwen3tts.engine import QwenEngine
    from narration_qwen3tts.settings import effective_generation

    common.cap_torch_threads(torch)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device_used_before_cuda = common.mb(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
    torch.cuda.init()
    torch.zeros(1, device=DEVICE)
    device_used_after_cuda = common.mb(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))

    results: dict[str, Any] = {
        "spike": "h+i (Qwen half)",
        "design_sections": ["20 (h)", "20 (i)", "4", "10.1"],
        "ran_at": common.now(),
        "software": common.software_facts(torch),
        "cuda_context_mb_estimate": device_used_after_cuda - device_used_before_cuda,
        "cuda_context_note": "device-level used memory before and after this process started CUDA (NVML); "
        "other jobs on the shared GPU can move it, so it is an estimate",
        "models": {},
    }
    engine = QwenEngine(torch)

    def settings_for(snapshot: Path, non_streaming: bool) -> dict[str, Any]:
        return {"non_streaming_mode": non_streaming, "generation": dict(effective_generation(snapshot))}

    def load(repo: str, snapshot: Path, non_streaming: bool) -> dict[str, Any]:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        before = common.gpu_memory(torch, DEVICE)
        result = engine.load(
            snapshot_dir=snapshot,
            device=DEVICE,
            dtype="bfloat16",
            attn_implementation="sdpa",
            settings=settings_for(snapshot, non_streaming),  # type: ignore[arg-type]
        )
        after = common.gpu_memory(torch, DEVICE)
        return {
            "load_s": round(result.load_s, 2),
            "vram_mb_reply": result.vram_mb,
            "memory_after_load": after,
            "device_free_drop_mb": before["device_free_mb"] - after["device_free_mb"],
            "model_type": result.model_type,
        }

    def render(name: str, call: Any) -> dict[str, Any]:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with common.captured_warnings() as warned:
            rendered = call()
        memory = common.gpu_memory(torch, DEVICE)
        path = OUT / f"{name}.wav"
        common.write_wav(path, rendered.audio, rendered.sample_rate)
        audio_s = rendered.audio.size / rendered.sample_rate
        return {
            "audio_s": round(audio_s, 3),
            "gen_s": round(rendered.gen_s, 2),
            "rtf": round(rendered.gen_s / audio_s, 3) if audio_s else None,
            "sample_rate": rendered.sample_rate,
            "new_tokens": rendered.new_tokens,
            "talker_steps": rendered.talker_steps,
            "hit_token_cap": rendered.hit_token_cap,
            "frames_per_audio_s": round(rendered.new_tokens / audio_s, 3) if audio_s else None,
            "peak_max_allocated_mb": memory["max_allocated_mb"],
            "peak_max_reserved_mb": memory["max_reserved_mb"],
            "sha256_float32": common.sha256_array(rendered.audio),
            "warnings": warned,
        }

    base_snapshot, base_revision = common.snapshot(common.MODEL_BASE)
    design_snapshot, design_revision = common.snapshot(common.MODEL_DESIGN)
    for snapshot, revision in ((base_snapshot, base_revision), (design_snapshot, design_revision)):
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or snapshot.name != revision:
            sys.exit(f"{snapshot} is not named by its 40-hex revision {revision}")

    # ---------------------------------------------------------------- Base
    common.apply_switches(torch, on=True)
    base: dict[str, Any] = {"repo": common.MODEL_BASE, "revision": base_revision, "snapshot_named_by_sha": True}
    base["first_load"] = load(common.MODEL_BASE, base_snapshot, non_streaming=False)
    clip, transcript, _ = common.voice_clip("d2")
    started = time.perf_counter()
    engine.prepare_voice("d2", clip, transcript, x_vector_only_mode=False)
    base["prepare_voice_s"] = round(time.perf_counter() - started, 2)
    base["renders"] = {}
    for segment_id in ("ladder-150", "ladder-560"):
        common.seed_all(torch, SEED)
        text = texts[segment_id]
        base["renders"][segment_id] = {
            "spoken_chars": len(text),
            **render(
                f"base-d2-{segment_id}", lambda t=text: engine.synthesize("d2", t, common.LANGUAGE, common.CEILING)
            ),
        }
    common.seed_all(torch, SEED)  # the run of 2026-09-26 set TEST_CAP as the load's ceiling; now it is the call's
    probe = render(
        "base-d2-token-cap-probe", lambda: engine.synthesize("d2", texts["ladder-150"], common.LANGUAGE, TEST_CAP)
    )
    base["token_cap_probe"] = {"max_new_tokens": TEST_CAP, **probe}
    engine.unload()
    base["memory_after_unload"] = common.gpu_memory(torch, DEVICE)
    results["models"]["base"] = base

    # ---------------------------------------------------------------- VoiceDesign
    design: dict[str, Any] = {"repo": common.MODEL_DESIGN, "revision": design_revision, "snapshot_named_by_sha": True}
    design["first_load"] = load(common.MODEL_DESIGN, design_snapshot, non_streaming=True)
    voice = canary["voice"]
    common.seed_all(torch, int(voice["seed"]))
    design["canary_design"] = render(
        "design-canary",
        lambda: engine.design(voice["description"], voice["design_text"], common.LANGUAGE, common.CEILING),
    )
    d2_clip, d2_text, d2_sidecar = common.voice_clip("d2")
    reference = sf.read(str(d2_clip), dtype="int16")[0]
    redesign: dict[str, Any] = {"seed": d2_sidecar["seed"], "reference_samples": int(reference.size)}
    for switches in (False, True):
        common.apply_switches(torch, on=switches)
        common.seed_all(torch, int(d2_sidecar["seed"]))
        label = "switches_on" if switches else "switches_off"
        facts = render(
            f"design-d2-redesign-{label}",
            lambda: engine.design(d2_sidecar["voice_description"], d2_text, common.LANGUAGE, common.CEILING),
        )
        audio, rate = sf.read(str(OUT / f"design-d2-redesign-{label}.wav"), dtype="float32")
        pcm = common.as_bakeoff_pcm16(audio, rate)
        same_length = pcm.size == reference.size
        facts["pcm16_bit_identical_to_bakeoff_clip"] = bool(same_length and np.array_equal(pcm, reference))
        facts["pcm16_samples"] = int(pcm.size)
        if same_length:
            facts["pcm16_max_abs_diff"] = int(np.max(np.abs(pcm.astype(np.int32) - reference.astype(np.int32))))
        redesign[label] = facts
    common.apply_switches(torch, on=True)
    design["d2_redesign_vs_bakeoff"] = redesign
    engine.unload()
    design["memory_after_unload"] = common.gpu_memory(torch, DEVICE)
    results["models"]["voice_design"] = design

    # ---------------------------------------------------------------- warm loads, then the weights' hashes
    for key, repo, snapshot, non_streaming in (
        ("base", common.MODEL_BASE, base_snapshot, False),
        ("voice_design", common.MODEL_DESIGN, design_snapshot, True),
    ):
        results["models"][key]["second_load"] = load(repo, snapshot, non_streaming)
        engine.unload()
    if not args.no_verify:
        for key, repo, snapshot, revision in (
            ("base", common.MODEL_BASE, base_snapshot, base_revision),
            ("voice_design", common.MODEL_DESIGN, design_snapshot, design_revision),
        ):
            started = time.perf_counter()
            expected = common.manifest_hashes(repo, revision)
            mismatched = [name for name, digest in expected.items() if common.sha256_file(snapshot / name) != digest]
            results["models"][key]["weights_verified"] = {
                "files": len(expected),
                "mismatched": mismatched,
                "hash_s": round(time.perf_counter() - started, 1),
            }

    results["network_attempts"] = guard.attempts
    results["loopback_connections"] = len(guard.loopback)
    results["offline_env"] = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    common.write_json(HERE / "results.json", results)
    common.write_json(OUT / "local-facts.json", common.local_facts(torch))
    pynvml.nvmlShutdown()
    print(f"network attempts: {len(guard.attempts)}; results in {HERE / 'results.json'}")
    return 0 if not guard.attempts else 1


if __name__ == "__main__":
    sys.exit(main())
