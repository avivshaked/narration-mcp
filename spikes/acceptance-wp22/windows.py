"""DC-15's window length against the bakeoff's voicelock evidence (WP22).

``refs/auditions/voicelock.csv`` compares clips of up to 37 s, which the bakeoff embedded in one pass. DC-15, as
approved, embeds audio longer than 30 s in equal windows, so the rows with a clip over 30 s no longer reproduce
exactly. For each window length (in seconds; ``0`` means one pass, what the bakeoff did), this embeds every
voicelock clip through the worker's own class (``narration_worker_qa.sv.WavLmSv``) on the CPU, with DC-15's rule
(one pass up to the length, else ``ceil(len / length)`` equal windows, the mean of their L2-normalised embeddings
normalised again), and compares each row's similarity with the bakeoff's.

The CPU is used because its embeddings equal the GPU's to 6 decimal places (``spikes/h-i-qa-load``); no GPU, no
lock. Run from the checkout in the QA worker's venv::

    workers/qa/.venv/Scripts/python.exe spikes/acceptance-wp22/windows.py [--lengths 0 30 40 60]

Writes ``spikes/acceptance-wp22/windows.json``: numbers keyed by the CSV's row numbers (1-based) and the clips'
positions in its sorted list of clips, never the clips' names.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
SIM_TOLERANCE = 0.002
"""The acceptance's tolerance for a similarity (``run.py``)."""


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--lengths", type=int, nargs="+", default=[0, 30, 40, 60])
    args = parser.parse_args()

    common.prepare_process_env()
    import numpy as np
    import torch
    from narration_worker_qa.align import read_audio, to_mono_16k
    from narration_worker_qa.sv import WavLmSv, windows

    common.cap_torch_threads(torch)
    root = common.bakeoff_root()
    rows = list(csv.DictReader((root / "refs" / "auditions" / "voicelock.csv").open(encoding="utf-8")))
    clips = sorted({row[k] for row in rows for k in ("a", "b")})
    position = {clip: i for i, clip in enumerate(clips)}
    audio = {clip: to_mono_16k(*read_audio(root / "refs" / f"{clip}.wav")) for clip in clips}
    seconds = {clip: x.shape[0] / 16_000 for clip, x in audio.items()}

    snapshot, revision = common.snapshot(common.MODEL_SV, common.SV_REVISION_EVIDENCE)
    sv = WavLmSv(torch)
    sv.load(common.MODEL_SV, revision, snapshot, "cpu")
    model, where = sv._model_for("cpu")  # pyright: ignore[reportPrivateUsage]

    def embedding(x: Any, length_s: int) -> Any:
        parts = [x] if length_s == 0 else windows(x, length_s * 16_000)
        vectors = [sv._one_pass(model, where, part) for part in parts]  # pyright: ignore[reportPrivateUsage]
        mean = np.mean(np.stack(vectors), axis=0)
        return mean / np.linalg.norm(mean)

    by_length: dict[str, Any] = {}
    long_rows = [
        i for i, row in enumerate(rows, 1) if max(seconds[row["a"]], seconds[row["b"]]) > min(args.lengths[1:] or [30])
    ]
    for length_s in args.lengths:
        vectors = {clip: embedding(x, length_s) for clip, x in audio.items()}
        deltas = []
        detail = []
        for i, row in enumerate(rows, 1):
            ours = round(float(np.dot(vectors[row["a"]], vectors[row["b"]])), 4)
            delta = round(abs(ours - float(row["spk_sim"])), 4)
            deltas.append(delta)
            if i in long_rows:
                detail.append({"row": i, "spk_sim": ours, "bakeoff": float(row["spk_sim"]), "abs_delta": delta})
        by_length["one_pass" if length_s == 0 else f"{length_s}s"] = {
            "max_abs_delta": max(deltas),
            "rows_over_tolerance": [i for i, d in enumerate(deltas, 1) if d > SIM_TOLERANCE],
            "rows_with_a_clip_over_the_shortest_length": detail,
        }
        print(length_s, by_length["one_pass" if length_s == 0 else f"{length_s}s"]["rows_over_tolerance"], flush=True)

    results = {
        "check": "DC-15's window length against the bakeoff's voicelock rows (CPU, the worker's WavLmSv)",
        "ran_at": common.now(),
        "model": [common.MODEL_SV, revision],
        "tolerance": SIM_TOLERANCE,
        "rows": len(rows),
        "clips": len(clips),
        "clips_over_30s": {str(position[c]): round(seconds[c], 1) for c in clips if seconds[c] > 30},
        "longest_clip_s": round(max(seconds.values()), 1),
        "by_window_length": by_length,
    }
    common.write_json(HERE / "windows.json", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
