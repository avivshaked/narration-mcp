"""Spike (b), part 2: starting values for the aligner's confidence and pause thresholds (plan.md WP15).

Design section 11.2 leaves the thresholds of steps 4, 5 and 7 to the alignment benchmark (WP38), which
needs hand marks. Until then ``narration.align`` needs starting values, labelled ASSUME. This script
measures what those values would separate, on the bake-off's six clone takes (8 paragraphs each, cut into
sentences as cues), read in place and never copied:

- **confidence** (step 5: the mean posterior of a cue's letters): for every sentence as spoken, and for
  planted failures: a paragraph's audio aligned with another paragraph's text, a sentence that was never
  spoken added to a paragraph, and a sentence never spoken put in a paragraph's middle;
- **pauses** (step 4: 20 ms energy frames): the frame level relative to the take's 95th percentile, inside
  each gap between two aligned words, split into gaps between sentences (a pause is expected) and gaps
  between words inside a sentence. For each threshold, the share of gaps that hold a run of at least two
  silent frames.

Run it like ``spike_b.py``, with the QA worker's interpreter and both environment variables set::

    workers/qa/.venv/Scripts/python.exe spikes/b-forced-align-cpu/calibrate.py --out <calibration.json>

The transcript is ``spike_b.py``'s minimal one; words with digits are left out, as the design does.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402  (after the thread cap, as in spike_b.py)
import soundfile as sf  # noqa: E402
import spike_b as sb  # noqa: E402
import torch  # noqa: E402

THRESHOLDS_DB = (-30.0, -35.0, -40.0, -45.0)
MIN_RUN_FRAMES = 2
ENERGY_FRAME_S = 0.02
NEVER_SPOKEN = "Nobody knows where they came from."


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def align_cues(aligner: Any, piece: Any, rate: int, cues: list[str]) -> tuple[list[float | None], list[Any]]:
    """Per-cue mean letter score, and each word's (cue, first frame, end frame)."""
    tokens: list[str] = []
    owner: list[tuple[int, int] | None] = []
    word = 0
    for ci, cue in enumerate(cues):
        for w in sb._tokens(cue)[0]:
            if tokens:
                tokens.append("|")
                owner.append(None)
            tokens.extend(w)
            owner.extend([(ci, word)] * len(w))
            word += 1
    reply = aligner.align(piece, rate, tokens)
    spans = reply["spans"]
    scores: list[float | None] = []
    for ci in range(len(cues)):
        mine = [s["score"] for s, o in zip(spans, owner, strict=True) if o is not None and o[0] == ci]
        scores.append(round(float(np.mean(mine)), 4) if mine else None)
    words: dict[int, list[int]] = {}
    cue_of: dict[int, int] = {}
    for s, o in zip(spans, owner, strict=True):
        if o is not None:
            words.setdefault(o[1], []).extend([s["start_frame"], s["end_frame"]])
            cue_of[o[1]] = o[0]
    return scores, [(cue_of[w], min(f), max(f)) for w, f in sorted(words.items())]


def levels_db(piece: Any, rate: int) -> Any:
    mono = piece.mean(axis=1) if piece.ndim > 1 else piece
    n = round(ENERGY_FRAME_S * rate)
    count = -(-mono.shape[0] // n)
    padded = np.zeros(count * n, dtype=np.float64)
    padded[: mono.shape[0]] = mono
    rms = np.sqrt((padded.reshape(count, n) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    return db - np.percentile(db, 95)


def longest_run(mask: Any) -> int:
    best = run = 0
    for silent in mask:
        run = run + 1 if silent else 0
        best = max(best, run)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    models_root = Path(os.environ["NARRATION_MODELS_ROOT"])
    bakeoff = Path(os.environ["NARRATION_BAKEOFF_ROOT"])
    aligner = sb.qa_align.Wav2Vec2Aligner(torch)
    aligner.load(sb.MODEL_REPO, sb.MODEL_REVISION, models_root / sb.SNAPSHOT)

    spoken: list[float] = []
    planted: list[dict[str, Any]] = []
    gaps: dict[str, list[dict[str, float | int]]] = {"between_sentences": [], "inside_sentences": []}
    per_take: dict[str, Any] = {}
    for take in sorted((bakeoff / "outputs").iterdir()):
        meta = json.loads((take / "r48_names_probe.json").read_text(encoding="utf-8"))
        audio, rate = sf.read(str(take / "r48_names_probe.wav"), dtype="float32", always_2d=True)
        segs = {s["id"]: s for s in meta["segments"]}
        rows: dict[str, Any] = {}
        for sid, seg in segs.items():
            piece = audio[seg["start_sample"] : seg["end_sample"]]
            cues = sentences(seg["text"])
            scores, words = align_cues(aligner, piece, rate, cues)
            rows[sid] = scores
            spoken.extend(s for s in scores if s is not None)
            rel = levels_db(piece, rate)
            for (ca, _, a_end), (cb, b_start, _) in itertools.pairwise(words):
                lo, hi = max(0, a_end - 1), min(len(rel), b_start + 1)  # one frame of tolerance each side
                window = rel[lo:hi]
                gaps["between_sentences" if cb != ca else "inside_sentences"].append(
                    {str(t): longest_run(window < t) for t in THRESHOLDS_DB}
                    | {"min_db": round(float(window.min()), 1) if window.size else 0.0}
                )
        per_take[take.name] = rows
        if take.name.endswith("d2-late-night_take1-seed1"):
            n07_first, *n07_rest = sentences(segs["n07"]["text"])
            cases = (  # (audio, what the text is, the text, the indices of the cues that are not in the audio)
                ("n08", "n07", segs["n07"]["text"], (0, 1)),
                ("n05", "n08", segs["n08"]["text"], (0, 1)),
                ("n07", "n03", segs["n03"]["text"], (0, 1)),
                ("n08", "n08 + never_spoken", f"{segs['n08']['text']} {NEVER_SPOKEN}", (2,)),
                ("n07", "n07 with never_spoken second", " ".join([n07_first, NEVER_SPOKEN, *n07_rest]), (1,)),
            )
            for audio_id, text_of, text, wrong in cases:
                seg = segs[audio_id]
                piece = audio[seg["start_sample"] : seg["end_sample"]]
                scores, _ = align_cues(aligner, piece, rate, sentences(text))
                planted.append(
                    {"audio": audio_id, "text_of": text_of, "cue_scores": scores, "cues_not_spoken": list(wrong)}
                )

    def summary(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values)
        return {
            "n": int(arr.size),
            "min": round(float(arr.min()), 4),
            "p5": round(float(np.percentile(arr, 5)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
        }

    def detection(rows: list[dict[str, float | int]]) -> dict[str, Any]:
        return {"n": len(rows)} | {
            f"share_with_run_of_{MIN_RUN_FRAMES}_frames_below_{t:g}_db": round(
                sum(1 for r in rows if r[str(t)] >= MIN_RUN_FRAMES) / len(rows), 3
            )
            for t in THRESHOLDS_DB
        }

    results = {
        "script": "spikes/b-forced-align-cpu/calibrate.py",
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "model": f"{sb.MODEL_REPO}@{sb.MODEL_REVISION}",
        "takes": "<bakeoff>/outputs/qwen3-tts-1.7b-clone-{d2,d4}-*-seed{1,2,3}/r48_names_probe.wav",
        "confidence": {
            "spoken_sentences": summary(spoken),
            "planted": planted,
            "planted_cues_not_spoken": sorted(
                p["cue_scores"][i] for p in planted for i in p["cues_not_spoken"] if p["cue_scores"][i] is not None
            ),
            "planted_spoken_cues_beside_them": sorted(
                s
                for p in planted
                for i, s in enumerate(p["cue_scores"])
                if i not in p["cues_not_spoken"] and s is not None
            ),
            "per_take_sentence_scores": per_take,
        },
        "pauses": {
            "energy_frame_s": ENERGY_FRAME_S,
            "reference": "95th percentile of the paragraph's frame levels",
            "between_sentences": detection(gaps["between_sentences"]),
            "inside_sentences": detection(gaps["inside_sentences"]),
            "between_sentences_min_db": summary([float(r["min_db"]) for r in gaps["between_sentences"]]),
        },
    }
    text = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
