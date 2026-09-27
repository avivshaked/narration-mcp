"""WP22's acceptance check: the QA worker, through its protocol, reproduces the bakeoff's ``eval/`` numbers.

*Accept (plan.md WP22):* on the bakeoff's real takes, similarities and transcripts match the bakeoff's numbers
within noise.

What it does:

1. copies into a local store's ``scratch/``, as the daemon does before a worker sees audio (Appendix A): the six
   clone takes of the names probe (d2 and d4, seeds 1 to 3), each take's eight segments (cut by the take's
   recorded sample boundaries, as ``eval/evaluate.py`` cut them), the two voices' clips, and every clip that
   ``refs/auditions/voicelock.csv`` compares;
2. starts the real worker, ``python -m narration_worker --role qa``, and speaks the protocol to it: ``hello``,
   ``load`` (Whisper and WavLM on ``cuda:0``), then per take ``transcribe`` of the whole take and of each
   segment (English, word times, long-form: what the job engine sends) and ``embed`` of each segment and of the
   voice's clip; ``embed`` of every voicelock clip; ``embed`` on the CPU of the two voices' clips (the canary's
   path); ``shutdown``;
3. scores as ``eval/evaluate.py`` scores (ported): WER with Whisper's English normaliser, ``spk_consist`` (the
   lowest similarity of a segment to the segments' mean), ``spk_to_ref`` (the segments' mean similarity to the
   voice's clip), ``worst_seg_wer``; and each voicelock row's ``spk_sim``;
4. with ``--bakeoff-recipe``, transcribes the takes again in this process exactly as ``eval/evaluate.py``
   did (the ASR pipeline with its defaults: five beams, no conditioning on the previous text), which checks
   that this scoring reproduces the bakeoff's WER. ``decoding.py`` compares the worker's decoding with the
   others section 11.1 could mean.

Accepted when every similarity is within ``SIM_TOLERANCE`` of the bakeoff's and every WER within
``WER_TOLERANCE`` (see there). The segment texts and transcripts are the bakeoff caller's: they are read at run
time and never written into the results; the transcripts are kept only in the gitignored
``.dev/spikes/acceptance-wp22/transcripts.json``.

Run from the checkout in the QA worker's venv, with the GPU lock held (about 10 minutes)::

    workers/qa/.venv/Scripts/python.exe spikes/acceptance-wp22/run.py [--out <json>] [--bakeoff-recipe]

Exit code 0 when accepted. Writes ``spikes/acceptance-wp22/results.json`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
LOCAL = common.DEV_SPIKES / "acceptance-wp22"
STORE = LOCAL / "store"
TAKE_DIRS = {"d2": "qwen3-tts-1.7b-clone-d2-late-night_take1", "d4": "qwen3-tts-1.7b-clone-d4-radio-drama_take2"}
SCRIPT = "r48_names_probe"
SEEDS = (1, 2, 3)
DEVICE = "cuda:0"
SIM_TOLERANCE = 0.002
"""How far a similarity may be from the bakeoff's (as WP20's scorer check). The bakeoff ran with torch's defaults
(cuDNN's TF32 on, no deterministic switches) and rounded to 4 places; the worker runs with section 10.1's
switches."""
WER_TOLERANCE = 0.01
"""How far a take's WER may be from the bakeoff's: two words of the reference's 256. The worker decodes as the
bakeoff did (five beams, not conditioned; ``narration_worker_qa.asr``), but with eager attention, word times and
the determinism switches, any of which could change a word."""


def normaliser(snapshot: Path) -> Callable[[str], str]:
    """Whisper's English text normaliser from the pinned snapshot (``WhisperTokenizer.normalize``)."""
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained(str(snapshot), local_files_only=True).normalize


def word_errors(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Substitutions, deletions and insertions between two word lists (Levenshtein), as jiwer counts them."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def wer(normalise: Callable[[str], str], reference: str, hypothesis: str) -> float:
    """``eval/evaluate.py``'s ``Scorer.wer``: both normalised, then word errors over the reference's words."""
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    return word_errors(ref, hyp) / len(ref) if ref else float("nan")


def bakeoff_results() -> dict[str, dict[str, float]]:
    """``eval/results.csv``'s rows for the names probe, by take id (``d2 seed1``)."""
    rows = list(csv.DictReader((common.bakeoff_root() / "eval" / "results.csv").open(encoding="utf-8")))
    out: dict[str, dict[str, float]] = {}
    for voice, folder in TAKE_DIRS.items():
        for seed in SEEDS:
            row = next(r for r in rows if r["model"] == f"{folder}-seed{seed}" and r["script"] == SCRIPT)
            out[f"{voice} seed{seed}"] = {
                k: float(row[k]) for k in ("wer", "worst_seg_wer", "spk_consist", "spk_to_ref")
            }
    return out


def voicelock_rows() -> list[dict[str, str]]:
    return list(csv.DictReader((common.bakeoff_root() / "refs" / "auditions" / "voicelock.csv").open(encoding="utf-8")))


def stage(source: Path, name: str) -> Path:
    """Copy a file into the store's scratch folder (the worker reads only inside the store)."""
    target = STORE / "scratch" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", default=str(HERE / "results.json"), help="where to write the results")
    parser.add_argument("--bakeoff-recipe", action="store_true", help="also transcribe as eval/evaluate.py did")
    args = parser.parse_args()

    common.prepare_process_env()
    import numpy as np
    import soundfile as sf
    from narration_worker.testing.client import WorkerProcess

    bakeoff = common.bakeoff_root()
    asr_snapshot, asr_revision = common.snapshot(common.MODEL_ASR)
    sv_snapshot, sv_revision = common.snapshot(common.MODEL_SV, common.SV_REVISION_EVIDENCE)
    normalise = normaliser(asr_snapshot)
    expected = bakeoff_results()
    script = json.loads((bakeoff / "scripts" / f"{SCRIPT}.json").read_text(encoding="utf-8"))
    full_reference = " ".join(s["text"] for s in script["segments"])

    if STORE.exists():
        shutil.rmtree(STORE)
    (STORE / "scratch").mkdir(parents=True)

    # ---------------------------------------------------------------- stage the audio in the store
    takes: dict[str, dict[str, Any]] = {}
    for voice, folder in TAKE_DIRS.items():
        for seed in SEEDS:
            take_id = f"{voice} seed{seed}"
            source = bakeoff / "outputs" / f"{folder}-seed{seed}" / f"{SCRIPT}.wav"
            meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
            audio, rate = sf.read(str(source), dtype="float32")
            segments = []
            for segment in meta["segments"]:
                path = STORE / "scratch" / "segments" / f"{voice}-seed{seed}-{segment['id']}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), audio[segment["start_sample"] : segment["end_sample"]], rate, subtype="FLOAT")
                segments.append({"id": segment["id"], "text": segment["text"], "wav": path})
            takes[take_id] = {
                "wav": stage(source, f"takes/{voice}-seed{seed}.wav"),
                "ref": stage(bakeoff / meta["settings"]["ref"], f"refs/{voice}.wav"),
                "segments": segments,
            }
    rows = voicelock_rows()
    clips = sorted({row[k] for row in rows for k in ("a", "b")})
    staged = {clip: stage(bakeoff / "refs" / f"{clip}.wav", f"voicelock/{i:03d}.wav") for i, clip in enumerate(clips)}

    # ---------------------------------------------------------------- the worker
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    argv = [sys.executable, "-m", "narration_worker", "--role", "qa", "--store", str(STORE), "--cpu-threads", "8"]
    transcripts: dict[str, Any] = {}
    scored: dict[str, dict[str, Any]] = {}
    voicelock: list[dict[str, Any]] = []
    canary: dict[str, Any] = {}
    started = time.perf_counter()
    with WorkerProcess(argv, env=env) as worker:
        assert worker.request("hello", timeout_s=300)["ok"]
        loaded = worker.request(
            "load",
            timeout_s=900,
            device=DEVICE,
            models={
                "asr": {"repo": common.MODEL_ASR, "revision": asr_revision, "snapshot_dir": str(asr_snapshot)},
                "sv": {"repo": common.MODEL_SV, "revision": sv_revision, "snapshot_dir": str(sv_snapshot)},
            },
        )
        assert loaded["ok"], loaded

        def call(op: str, **fields: Any) -> dict[str, Any]:
            reply = worker.request(op, timeout_s=900, **fields)
            if not reply["ok"]:
                raise RuntimeError(f"{op} failed: {reply['error']['code']}: {worker.stderr_text()[-2000:]}")
            return reply

        def embed(path: Path, device: str = "cuda") -> Any:
            return np.asarray(call("embed", wav=str(path), device=device)["embedding"], dtype=np.float64)

        def transcribe(path: Path) -> dict[str, Any]:
            return call("transcribe", wav=str(path), language="English", word_timestamps=True, long_form=True)

        for take_id, take in takes.items():
            began = time.perf_counter()
            full = transcribe(take["wav"])
            segment_wers, embeddings, texts = [], [], {}
            for segment in take["segments"]:
                heard = transcribe(segment["wav"])
                segment_wers.append(wer(normalise, segment["text"], heard["text"]))
                texts[segment["id"]] = heard["text"]
                embeddings.append(embed(segment["wav"]))
            e = np.stack(embeddings)
            mean = e.mean(0)
            mean /= np.linalg.norm(mean)
            ref = embed(take["ref"])
            untimed = sum(1 for w in full["words"] if w["start_s"] is None or w["end_s"] is None)
            scored[take_id] = {
                "wer": round(wer(normalise, full_reference, full["text"]), 4),
                "worst_seg_wer": round(max(segment_wers), 4),
                "spk_consist": round(float((e @ mean).min()), 4),
                "spk_to_ref": round(float((e @ ref).mean()), 4),
                "words": len(full["words"]),
                "words_without_times": untimed,
                "seconds": round(time.perf_counter() - began, 1),
            }
            transcripts[take_id] = {"full": full["text"], "segments": texts}
        embedded = {clip: embed(path) for clip, path in staged.items()}
        for index, row in enumerate(rows, 1):
            ours = float(embedded[row["a"]] @ embedded[row["b"]])
            voicelock.append({"row": index, "spk_sim": round(ours, 4), "bakeoff": float(row["spk_sim"])})
        for voice in TAKE_DIRS:
            path = takes[f"{voice} seed1"]["ref"]
            canary[voice] = {"cosine_cpu_to_cuda": round(float(embed(path, "cpu") @ embed(path, "cuda")), 6)}
        assert worker.request("shutdown", timeout_s=120)["ok"]
    worker_seconds = time.perf_counter() - started

    # ---------------------------------------------------------------- the bakeoff's own recipe, for comparison
    recipe: dict[str, Any] | None = None
    if args.bakeoff_recipe:
        recipe = bakeoff_recipe(takes, full_reference, normalise, asr_snapshot, transcripts)

    # ---------------------------------------------------------------- compare
    comparison: dict[str, Any] = {}
    failures: list[str] = []
    for take_id, ours in scored.items():
        theirs = expected[take_id]
        deltas = {k: round(ours[k] - theirs[k], 4) for k in ("wer", "worst_seg_wer", "spk_consist", "spk_to_ref")}
        comparison[take_id] = {"ours": ours, "bakeoff": theirs, "delta": deltas}
        for key in ("spk_consist", "spk_to_ref"):
            if abs(deltas[key]) > SIM_TOLERANCE:
                failures.append(f"{take_id} {key} {deltas[key]:+.4f}")
        if abs(deltas["wer"]) > WER_TOLERANCE:
            failures.append(f"{take_id} wer {deltas['wer']:+.4f}")
    voicelock_deltas = [abs(r["spk_sim"] - r["bakeoff"]) for r in voicelock]
    failures += [f"voicelock row {r['row']}" for r in voicelock if abs(r["spk_sim"] - r["bakeoff"]) > SIM_TOLERANCE]
    results: dict[str, Any] = {
        "check": "WP22 acceptance: the QA worker reproduces the bakeoff's eval numbers",
        "ran_at": common.now(),
        "software": common.software_facts(__import__("torch")),
        "models": {"asr": [common.MODEL_ASR, asr_revision], "sv": [common.MODEL_SV, sv_revision]},
        "tolerances": {"similarity": SIM_TOLERANCE, "wer": WER_TOLERANCE},
        "worker": {"load_s": loaded["load_s"], "vram_mb": loaded["vram_mb"], "seconds": round(worker_seconds, 1)},
        "takes": comparison,
        "voicelock": {
            "rows": len(voicelock),
            "max_abs_delta": round(max(voicelock_deltas), 4),
            "mean_abs_delta": round(float(np.mean(voicelock_deltas)), 5),
            "by_row": voicelock,
        },
        "canary_cpu_path": canary,
        "bakeoff_recipe": recipe,
        "failures": failures,
        "accepted": not failures,
    }
    common.write_json(Path(args.out), results)
    common.write_json(LOCAL / "transcripts.json", transcripts)
    print(json.dumps({"accepted": not failures, "failures": failures}, indent=2))
    return 0 if not failures else 1


def bakeoff_recipe(
    takes: dict[str, dict[str, Any]],
    full_reference: str,
    normalise: Callable[[str], str],
    snapshot: Path,
    transcripts: dict[str, Any],
) -> dict[str, Any]:
    """Transcribe each whole take and its segments as ``eval/evaluate.py`` did (``Scorer.transcribe``): the ASR
    pipeline in float16 with its own defaults and ``generate_kwargs`` language and task only; torch's default
    switches, as the bakeoff ran."""
    import soundfile as sf
    import torch
    from narration_worker_qa.align import to_mono_16k
    from transformers import pipeline

    common.cap_torch_threads(torch)
    asr = pipeline("automatic-speech-recognition", model=str(snapshot), dtype=torch.float16, device=DEVICE)

    def heard(path: Path) -> str:
        audio, rate = sf.read(str(path), dtype="float32")
        out = asr(
            {"raw": to_mono_16k(audio, rate), "sampling_rate": 16000},
            return_timestamps=True,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        return str(out["text"]).strip()

    out: dict[str, Any] = {"decoding": "the ASR pipeline's defaults (num_beams 5), language and task only"}
    for take_id, take in takes.items():
        text = heard(take["wav"])
        segment_wers = [wer(normalise, s["text"], heard(s["wav"])) for s in take["segments"]]
        out[take_id] = {
            "wer": round(wer(normalise, full_reference, text), 4),
            "worst_seg_wer": round(max(segment_wers), 4),
        }
        transcripts[take_id]["bakeoff_recipe_full"] = text
    return out


if __name__ == "__main__":
    sys.exit(main())
