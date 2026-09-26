"""The CTC aligner's memory and time by audio length, in one pass (WP15's follow-up F5; design section 11.2).

The aligner computes wav2vec2's emissions for a whole clip in one pass on the CPU, then runs torchaudio's
``forced_align`` over them. WP15 measured takes up to about 40 s. This measures longer ones, to say whether one
pass is safe for the lengths a segment can reach, or whether the emissions should be computed in chunks.

For each length, a fresh process (the QA worker's venv, the worker's thread cap of 8, offline):

1. loads the pinned wav2vec2 snapshot through the worker's own class (``narration_worker_qa.align``);
2. makes synthetic audio of that length (low-level noise with a 150 Hz tone: the emission's cost does not
   depend on what is said) and a token sequence of 15 letters and word separators per second, a fast
   narration's rate of characters;
3. aligns it with ``Wav2Vec2Aligner.align``, sampling the process's resident memory every 5 ms;
4. reports the peak above the memory held after the model loaded, and the time.

No audio file is read or written, and no GPU is used. Run from the checkout in the QA worker's venv::

    workers/qa/.venv/Scripts/python.exe spikes/aligner-memory/run.py [--lengths 30 60 120 240 480]

Writes ``spikes/aligner-memory/results.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
REPO = "facebook/wav2vec2-large-960h-lv60-self"
TOKENS_PER_S = 15.0
SAMPLE_S = 0.005


def one(seconds: float) -> dict[str, Any]:
    """Align ``seconds`` of synthetic audio in this process; the facts as a dict."""
    common.prepare_process_env()
    import numpy as np
    import psutil
    import torch
    from narration_worker_qa.align import Wav2Vec2Aligner

    common.cap_torch_threads(torch)
    process = psutil.Process()
    snapshot, revision = common.snapshot(REPO)
    aligner = Wav2Vec2Aligner(torch)
    load_s = aligner.load(REPO, revision, snapshot)
    rng = np.random.default_rng(0)
    rate = 16_000
    t = np.arange(int(seconds * rate)) / rate
    audio = (0.05 * np.sin(2 * np.pi * 150 * t) + 0.01 * rng.standard_normal(t.shape[0])).astype(np.float32)
    letters = list("ETAONIHSRDLUMWCFGYPBVK")
    tokens: list[str] = []
    while len(tokens) < int(seconds * TOKENS_PER_S):
        tokens += [letters[i] for i in rng.integers(0, len(letters), 4)] + ["|"]
    tokens = tokens[: int(seconds * TOKENS_PER_S)]
    if tokens[-1] == "|":
        tokens[-1] = "E"

    baseline = process.memory_info().rss
    peak = baseline
    running = True

    def sample() -> None:
        nonlocal peak
        while running:
            peak = max(peak, process.memory_info().rss)
            time.sleep(SAMPLE_S)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    emission = aligner.emission(audio)
    emission_s = time.perf_counter() - started
    peak_after_emission = max(peak, process.memory_info().rss)
    spans = aligner.spans(emission, tokens)
    total_s = time.perf_counter() - started
    running = False
    sampler.join()
    peak = max(peak, process.memory_info().rss)
    mib = 1024 * 1024
    return {
        "seconds": seconds,
        "frames": int(emission.shape[0]),
        "tokens": len(tokens),
        "spans": len(spans),
        "load_s": round(load_s, 2),
        "rss_after_load_mb": round(baseline / mib),
        "peak_above_loaded_mb_emission": round((peak_after_emission - baseline) / mib),
        "peak_above_loaded_mb_total": round((peak - baseline) / mib),
        "emission_s": round(emission_s, 2),
        "forced_align_s": round(total_s - emission_s, 2),
        "total_s": round(total_s, 2),
        "realtime_factor": round(total_s / seconds, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--lengths", type=float, nargs="+", default=[30.0, 60.0, 120.0, 240.0, 480.0])
    parser.add_argument("--one", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one is not None:
        print(json.dumps(one(args.one)))
        return 0

    import psutil

    runs: list[dict[str, Any]] = []
    for seconds in args.lengths:
        completed = subprocess.run(
            [sys.executable, __file__, "--one", str(seconds)], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            runs.append({"seconds": seconds, "error": completed.stderr.strip().splitlines()[-1][:500]})
            continue
        runs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        print(runs[-1], flush=True)
    import torch

    results = {
        "spike": "aligner memory by length (WP15 F5)",
        "design_sections": ["11.2"],
        "ran_at": common.now(),
        "software": common.software_facts(torch),
        "cpu_threads": common.CPU_THREADS,
        "machine_ram_gb": round(psutil.virtual_memory().total / 1024**3),
        "tokens_per_second": TOKENS_PER_S,
        "runs": runs,
    }
    common.write_json(HERE / "results.json", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
