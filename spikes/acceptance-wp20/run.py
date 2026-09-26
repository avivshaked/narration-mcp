"""WP20's acceptance check: a render through the worker protocol matches the bakeoff's take (plan.md WP20).

*Accept:* a render through the worker protocol, with the bakeoff's settings and seed, scores WavLM similarity
≥ 0.98 against the bakeoff take for the same text and seed. It need not be bit-identical: the bakeoff set no
determinism switches.

What it does:

1. copies the allowlisted clips d2 and d4 (sha256-checked) into a local store's ``scratch/voices/``, as the
   daemon does before a worker sees a clip (Appendix A);
2. starts the real worker, ``python -m narration_worker --role qwen3``, and speaks the protocol to it:
   ``hello``, ``load`` (Base, the section 10.1 switches, the snapshot's sampling values passed explicitly,
   ``non_streaming_mode`` false), ``prepare_voice`` (ICL: the clip and its exact transcript),
   ``synthesize`` for bakeoff segments n07 and n08 of the names probe with the bakeoff's seed rule
   (``seed * 1000 + segment index``, seed 1) and the daemon's call cap for each text
   (``max_new_tokens_for``, DC-4), ``unload``, ``shutdown``;
3. scores each render against the bakeoff take's segment with WavLM-SV, the evidence's method
   (``spikes/wavlm_similarity.py``, in the QA worker's venv, on the CPU), and scores two other takes of the
   same text (seeds 2 and 3) against the seed-1 take as a baseline for what "the same voice, another
   delivery" scores;
4. checks the scorer itself: the mean similarity of each seed-1 take's eight segments to the voice's clip
   must reproduce the bakeoff's recorded ``spk_to_ref`` (d2 0.9775, d4 0.9663) within 0.002;
5. records, beyond the acceptance, whether each render converted to 16 bits as the bakeoff stored its takes
   equals the take's segment sample for sample (``identical_to_bakeoff_pcm16``).

The segment texts are the bakeoff caller's; they are read from the bakeoff at run time, never copied here.

Run from the checkout in the Qwen worker's venv, with the GPU lock held (about 3 minutes)::

    workers/qwen3tts/.venv/Scripts/python.exe spikes/acceptance-wp20/run.py [--out <json>]

Exit code 0 when every render scores ≥ 0.98. Writes ``spikes/acceptance-wp20/results.json`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
STORE = common.DEV_SPIKES / "acceptance" / "store"
THRESHOLD = 0.98
SCORER_TOLERANCE = 0.002
"""How close the scorer's mean must come to the bakeoff's recorded spk_to_ref (it ran on the GPU, this on
the CPU)."""
TAKE_DIRS = {"d2": "qwen3-tts-1.7b-clone-d2-late-night_take1", "d4": "qwen3-tts-1.7b-clone-d4-radio-drama_take2"}
SCRIPT = "r48_names_probe"
SEGMENTS = ("n07", "n08")
SEED = 1
DETERMINISM = {
    "tf32": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "deterministic_algorithms": "warn_only",
}
QA_PYTHON = common.CHECKOUT / "workers" / "qa" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def take(voice: str, seed: int) -> tuple[Path, dict[str, Any]]:
    folder = common.bakeoff_root() / "outputs" / f"{TAKE_DIRS[voice]}-seed{seed}"
    return folder / f"{SCRIPT}.wav", json.loads((folder / f"{SCRIPT}.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", default=str(HERE / "results.json"), help="where to write the results")
    args = parser.parse_args()

    from narration_qwen3tts.settings import effective_generation
    from narration_worker.testing.client import WorkerProcess

    if STORE.exists():
        shutil.rmtree(STORE)
    (STORE / "scratch" / "voices").mkdir(parents=True)
    snapshot, revision = common.snapshot(common.MODEL_BASE)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    argv = [sys.executable, "-m", "narration_worker", "--role", "qwen3", "--store", str(STORE), "--cpu-threads", "8"]
    renders: list[dict[str, Any]] = []
    with WorkerProcess(argv, env=env) as worker:
        hello = worker.request("hello", timeout_s=300)
        assert hello["ok"], hello
        loaded = worker.request(
            "load",
            timeout_s=600,
            device="cuda:0",
            model={"repo": common.MODEL_BASE, "revision": revision, "snapshot_dir": str(snapshot)},
            engine_profile_id="qwen3-base-1.7b.p1",
            dtype="bfloat16",
            attn_implementation="sdpa",
            determinism=DETERMINISM,
            settings={"non_streaming_mode": False, "generation": dict(effective_generation(snapshot))},
        )
        assert loaded["ok"], loaded
        for voice in ("d2", "d4"):
            clip, transcript, _ = common.voice_clip(voice)
            digest = common.sha256_file(clip)
            voice_hash = "sha256:" + digest
            stored = STORE / "scratch" / "voices" / f"{digest}.wav"
            shutil.copyfile(clip, stored)
            prepared = worker.request(
                "prepare_voice",
                timeout_s=300,
                voice_hash=voice_hash,
                ref_wav=str(stored),
                ref_text=transcript,
                x_vector_only_mode=False,
            )
            assert prepared["ok"], prepared
            _, meta = take(voice, SEED)
            for segment_id in SEGMENTS:
                index = next(i for i, s in enumerate(meta["segments"]) if s["id"] == segment_id)
                seed = SEED * 1000 + index
                out = STORE / "scratch" / "acceptance" / f"{voice}-{segment_id}-seed{seed}.wav"
                reply = worker.request(
                    "synthesize",
                    timeout_s=900,
                    voice_hash=voice_hash,
                    engine_text=meta["segments"][index]["text"],
                    language=common.LANGUAGE,
                    seed=seed,
                    max_new_tokens=common.call_cap(meta["segments"][index]["text"]),
                    out_path=str(out),
                )
                assert reply["ok"], reply
                segment = meta["segments"][index]
                renders.append(
                    {
                        "voice": voice,
                        "segment": segment_id,
                        "seed": seed,
                        "reply": {k: v for k, v in reply.items() if k not in ("id", "ok")},
                        "bakeoff_samples": segment["end_sample"] - segment["start_sample"],
                        "_wav": str(out),
                        "_span": [segment["start_sample"], segment["end_sample"]],
                    }
                )
                print(f"{voice} {segment_id} seed {seed}: {reply['samples']} samples in {reply['gen_s']} s")
        assert worker.request("unload", timeout_s=300)["ok"]
        assert worker.request("shutdown", timeout_s=300)["ok"]
        worker.wait(120)

    pairs: list[dict[str, Any]] = []
    for render in renders:
        wav, _ = take(render["voice"], SEED)
        pairs.append(
            {
                "label": f"render {render['voice']} {render['segment']} vs bakeoff seed {SEED}",
                "a": render["_wav"],
                "b": str(wav),
                "b_span": render["_span"],
            }
        )
    for voice in ("d2", "d4"):
        base_wav, base_meta = take(voice, SEED)
        for other in (2, 3):
            other_wav, other_meta = take(voice, other)
            for segment_id in SEGMENTS:
                a = next(s for s in base_meta["segments"] if s["id"] == segment_id)
                b = next(s for s in other_meta["segments"] if s["id"] == segment_id)
                pairs.append(
                    {
                        "label": f"baseline {voice} {segment_id}: bakeoff seed {SEED} vs seed {other}",
                        "a": str(base_wav),
                        "a_span": [a["start_sample"], a["end_sample"]],
                        "b": str(other_wav),
                        "b_span": [b["start_sample"], b["end_sample"]],
                    }
                )
    # The scorer's own check: each seed-1 take's segments against the voice's clip, whose mean the bakeoff
    # recorded as spk_to_ref (eval/results.csv). A scorer that does not reproduce it proves nothing.
    tool_pairs = 0
    for voice in ("d2", "d4"):
        base_wav, base_meta = take(voice, SEED)
        clip, _, _ = common.voice_clip(voice)
        for segment in base_meta["segments"]:
            pairs.append(
                {
                    "label": f"scorer check {voice} {segment['id']}: bakeoff seed {SEED} vs the clip",
                    "a": str(base_wav),
                    "a_span": [segment["start_sample"], segment["end_sample"]],
                    "b": str(clip),
                }
            )
            tool_pairs += 1
    pairs_file = STORE / "pairs.json"
    scored_file = STORE / "scored.json"
    common.write_json(pairs_file, pairs)
    subprocess.run(
        [
            str(QA_PYTHON),
            str(common.CHECKOUT / "spikes" / "wavlm_similarity.py"),
            "--pairs",
            str(pairs_file),
            "--out",
            str(scored_file),
        ],
        check=True,
    )
    scored = json.loads(scored_file.read_text(encoding="utf-8"))["pairs"]
    import numpy as np
    import soundfile as sf

    for render, pair in zip(renders, scored, strict=False):
        render["similarity_to_bakeoff_take"] = pair["similarity"]
        render["passes"] = pair["similarity"] >= THRESHOLD
        render["same_length_as_bakeoff"] = render["reply"]["samples"] == render["bakeoff_samples"]
        # Not part of the acceptance: is the render the bakeoff's take, sample for sample, once stored at the
        # bakeoff's 16 bits? (It was, 2026-09-26; the README says what that shows.)
        audio, rate = sf.read(render["_wav"], dtype="float32")
        start, end = render["_span"]
        theirs = sf.read(str(take(render["voice"], SEED)[0]), dtype="int16", start=start, stop=end)[0]
        render["identical_to_bakeoff_pcm16"] = bool(np.array_equal(common.as_bakeoff_pcm16(audio, rate), theirs))
        del render["_wav"], render["_span"]
    baselines = [{"label": p["label"], "similarity": p["similarity"]} for p in scored[len(renders) : -tool_pairs]]
    scorer_check: dict[str, Any] = {}
    for voice in ("d2", "d4"):
        mine = [p["similarity"] for p in scored[-tool_pairs:] if p["label"].startswith(f"scorer check {voice} ")]
        folder = take(voice, SEED)[0].parent
        golden = json.loads((folder / f"{SCRIPT}.eval.json").read_text(encoding="utf-8"))["spk_to_ref"]
        mean = round(sum(mine) / len(mine), 4)
        scorer_check[voice] = {"segments": len(mine), "mean": mean, "bakeoff_spk_to_ref": golden}
        scorer_check[voice]["reproduced"] = abs(mean - golden) <= SCORER_TOLERANCE
    fingerprint = {k: v for k, v in hello["fingerprint"].items() if k not in ("gpu", "driver", "platform")}
    results = {
        "check": "WP20 acceptance: a render through the worker protocol vs the bakeoff's take",
        "ran_at": common.now(),
        "threshold": THRESHOLD,
        "worker_fingerprint": fingerprint,
        "load": {"load_s": loaded["load_s"], "vram_mb": loaded["vram_mb"], "determinism": DETERMINISM},
        "sv": {"model": common.MODEL_SV, "revision_prefix": common.SV_REVISION_EVIDENCE, "device": "cpu"},
        "renders": renders,
        "baselines_other_seeds": baselines,
        "scorer_check": {"tolerance": SCORER_TOLERANCE, **scorer_check},
        "accepted": all(r["passes"] for r in renders) and all(c["reproduced"] for c in scorer_check.values()),
    }
    common.write_json(Path(args.out), results)
    print(json.dumps({r["voice"] + " " + r["segment"]: r["similarity_to_bakeoff_take"] for r in renders}))
    return 0 if results["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
