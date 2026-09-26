"""Spikes (h) and (i), QA half: load Whisper-large-v3 and WavLM-base-plus-sv offline from SHA-named snapshots;
VRAM and times while transcribing and embedding a real take.

Design section 20, Phase 0: (h) VRAM and load times; (i) offline loading from a snapshot directory named by its
commit SHA. The Qwen half is ``spikes/h-i-qwen-load``.

What it does, in one process, with the GPU lock held (AGENTS.md section 5):

1. sets the worker's start-up environment (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, CUBLAS_WORKSPACE_CONFIG, thread
   caps) and installs a guard that refuses every network connection and name lookup, recording any attempt;
2. checks that each snapshot folder is named by its 40-hex revision, applies the QA worker's determinism switches
   (``narration_worker_qa.worker.QA_DETERMINISM``) and loads each model through the worker's own classes
   (``narration_worker_qa.asr.WhisperAsr``, ``narration_worker_qa.sv.WavLmSv``) on ``cuda:0``;
3. transcribes (word times, long-form) and embeds a 40 s slice and the whole of one of the bakeoff's clone takes
   (d2, seed 1, about 119 s), recording the allocator's peaks; embeds both again on the CPU (the canary's path)
   and compares the embeddings;
4. measures the peaks' growth with length (10 to 119 s from the start of the take): transcription with and
   without word times, and embedding, which it also compares with the mean of 30 s windows' embeddings;
5. unloads, loads each model a second time (warm file cache), and re-hashes the weights against the manifest.

Run from the checkout, in the QA worker's venv, with the GPU lock held::

    workers/qa/.venv/Scripts/python.exe spikes/h-i-qa-load/run.py

Writes ``spikes/h-i-qa-load/results.json`` (published: numbers keyed by the bakeoff's ids, no text, no paths, no
GPU name) and ``.dev/spikes/h-i-qa/local-facts.json``.
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
OUT = common.DEV_SPIKES / "h-i-qa"
DEVICE = "cuda:0"
TAKE = "outputs/qwen3-tts-1.7b-clone-d2-late-night_take1-seed1/r48_names_probe.wav"
"""The bakeoff's d2 clone take, seed 1 (plan.md 1.2): about 119 s at 24 kHz."""
SLICE_S = 40.0
LENGTHS_S = (10.0, 20.0, 30.0, 60.0, 90.0, 119.0)
"""The lengths the peaks are measured at, cut from the start of the take."""
WINDOW_S = 30
"""The window of the windowed embedding the one-pass embedding is compared with (a possible bound on VRAM)."""


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--no-verify", action="store_true", help="skip re-hashing the weights")
    args = parser.parse_args()

    common.prepare_process_env()
    guard = common.NetworkGuard().install()

    import numpy as np
    import pynvml
    import torch
    from narration_worker.determinism import apply_determinism
    from narration_worker_qa.align import read_audio, to_mono_16k
    from narration_worker_qa.asr import WhisperAsr
    from narration_worker_qa.sv import WavLmSv
    from narration_worker_qa.worker import QA_DETERMINISM

    common.cap_torch_threads(torch)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    used_before_cuda = common.mb(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
    torch.cuda.init()
    torch.zeros(1, device=DEVICE)
    used_after_cuda = common.mb(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))

    audio, rate = read_audio(common.bakeoff_root() / TAKE)
    whole = to_mono_16k(audio, rate)
    clips = {"slice_40s": whole[: int(SLICE_S * 16_000)], "whole_take": whole}

    results: dict[str, Any] = {
        "spike": "h+i (QA half)",
        "design_sections": ["20 (h)", "20 (i)", "4", "11.1"],
        "ran_at": common.now(),
        "software": common.software_facts(torch),
        "determinism": apply_determinism(QA_DETERMINISM, torch),
        "cuda_context_mb_estimate": used_after_cuda - used_before_cuda,
        "cuda_context_note": "device-level used memory before and after this process started CUDA (NVML); "
        "other jobs on the shared GPU can move it, so it is an estimate",
        "audio": {name: {"take": "d2 seed1", "seconds": round(x.shape[0] / 16_000, 2)} for name, x in clips.items()},
        "models": {},
    }

    asr_snapshot, asr_revision = common.snapshot(common.MODEL_ASR)
    sv_snapshot, sv_revision = common.snapshot(common.MODEL_SV, common.SV_REVISION_EVIDENCE)
    for snapshot, revision in ((asr_snapshot, asr_revision), (sv_snapshot, sv_revision)):
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or snapshot.name != revision:
            sys.exit(f"{snapshot} is not named by its 40-hex revision {revision}")

    def load(model: Any, repo: str, revision: str, snapshot: Path) -> dict[str, Any]:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        before = common.gpu_memory(torch, DEVICE)
        load_s = model.load(repo, revision, snapshot, DEVICE)
        after = common.gpu_memory(torch, DEVICE)
        return {
            "load_s": round(load_s, 2),
            "memory_after_load": after,
            "reserved_added_mb": after["reserved_mb"] - before["reserved_mb"],
            "device_free_drop_mb": before["device_free_mb"] - after["device_free_mb"],
        }

    def measured(call: Any) -> tuple[Any, dict[str, Any]]:
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with common.captured_warnings() as warned:
            started = time.perf_counter()
            value = call()
            torch.cuda.synchronize(DEVICE)
            seconds = time.perf_counter() - started
        memory = common.gpu_memory(torch, DEVICE)
        return value, {
            "seconds": round(seconds, 2),
            "peak_max_allocated_mb": memory["max_allocated_mb"],
            "peak_max_reserved_mb": memory["max_reserved_mb"],
            "warnings": warned,
        }

    # ---------------------------------------------------------------- both models resident, as the QA group is
    asr = WhisperAsr(torch)
    sv = WavLmSv(torch)
    asr_facts: dict[str, Any] = {"repo": common.MODEL_ASR, "revision": asr_revision, "snapshot_named_by_sha": True}
    sv_facts: dict[str, Any] = {"repo": common.MODEL_SV, "revision": sv_revision, "snapshot_named_by_sha": True}
    asr_facts["first_load"] = load(asr, common.MODEL_ASR, asr_revision, asr_snapshot)
    sv_facts["first_load"] = load(sv, common.MODEL_SV, sv_revision, sv_snapshot)
    results["group_memory_after_load"] = common.gpu_memory(torch, DEVICE)

    asr_facts["transcribe"] = {}
    sv_facts["embed_cuda"] = {}
    sv_facts["embed_cpu"] = {}
    for name, clip in clips.items():
        seconds = clip.shape[0] / 16_000
        reply, facts = measured(lambda c=clip: asr.transcribe(c, "English", word_timestamps=True, long_form=True))
        timed = [w for w in reply["words"] if w["start_s"] is not None and w["end_s"] is not None]
        facts |= {
            "rtf": round(facts["seconds"] / seconds, 4),
            "words": len(reply["words"]),
            "words_with_times": len(timed),
            "last_word_end_s": timed[-1]["end_s"] if timed else None,
        }
        asr_facts["transcribe"][name] = facts
        on_gpu, facts = measured(lambda c=clip: sv.embed(c, "cuda"))
        sv_facts["embed_cuda"][name] = facts
        started = time.perf_counter()
        on_cpu = sv.embed(clip, "cpu")  # the first call loads the CPU copy
        cpu_seconds = time.perf_counter() - started
        cosine = float(np.dot(on_gpu["embedding"], on_cpu["embedding"]))
        sv_facts["embed_cpu"][name] = {
            "seconds_including_any_copy_load": round(cpu_seconds, 2),
            "cosine_to_cuda": round(cosine, 6),
        }

    # ---------------------------------------------------------------- by length: what the peaks grow with
    resident = common.gpu_memory(torch, DEVICE)["allocated_mb"]
    by_length: dict[str, Any] = {"resident_allocated_mb": resident, "transcribe": [], "embed": []}
    for seconds in LENGTHS_S:
        clip = whole[: int(seconds * 16_000)]
        for word_timestamps in (False, True):
            torch.cuda.empty_cache()
            _, facts = measured(
                lambda c=clip, w=word_timestamps: asr.transcribe(c, "English", word_timestamps=w, long_form=True)
            )
            by_length["transcribe"].append(
                {
                    "seconds": seconds,
                    "word_timestamps": word_timestamps,
                    "peak_above_resident_mb": facts["peak_max_allocated_mb"] - resident,
                    "seconds_taken": facts["seconds"],
                }
            )
        torch.cuda.empty_cache()
        one_pass, facts = measured(lambda c=clip: sv.embed(c, "cuda"))
        windows = [clip[i : i + WINDOW_S * 16_000] for i in range(0, clip.shape[0], WINDOW_S * 16_000)]
        windows = [w for w in windows if w.shape[0] >= sv.min_samples]
        mean = np.mean([sv.embed(w, "cuda")["embedding"] for w in windows], axis=0)
        mean /= np.linalg.norm(mean)
        by_length["embed"].append(
            {
                "seconds": seconds,
                "peak_above_resident_mb": facts["peak_max_allocated_mb"] - resident,
                "peak_reserved_mb": facts["peak_max_reserved_mb"],
                "seconds_taken": facts["seconds"],
                "windows_of_30s": len(windows),
                "cosine_one_pass_to_window_mean": round(float(np.dot(one_pass["embedding"], mean)), 6),
            }
        )
    results["by_length"] = by_length
    asr.unload()
    sv.unload()
    torch.cuda.empty_cache()
    results["memory_after_unload"] = common.gpu_memory(torch, DEVICE)

    # ---------------------------------------------------------------- warm loads, then the weights' hashes
    for facts, model, repo, revision, snapshot in (
        (asr_facts, WhisperAsr(torch), common.MODEL_ASR, asr_revision, asr_snapshot),
        (sv_facts, WavLmSv(torch), common.MODEL_SV, sv_revision, sv_snapshot),
    ):
        facts["second_load_alone"] = load(model, repo, revision, snapshot)
        model.unload()
        torch.cuda.empty_cache()
    if not args.no_verify:
        for facts, repo, revision, snapshot in (
            (asr_facts, common.MODEL_ASR, asr_revision, asr_snapshot),
            (sv_facts, common.MODEL_SV, sv_revision, sv_snapshot),
        ):
            started = time.perf_counter()
            expected = common.manifest_hashes(repo, revision)
            mismatched = [name for name, digest in expected.items() if common.sha256_file(snapshot / name) != digest]
            facts["weights_verified"] = {
                "files": len(expected),
                "mismatched": mismatched,
                "hash_s": round(time.perf_counter() - started, 1),
            }
    results["models"] = {"asr": asr_facts, "sv": sv_facts}
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
