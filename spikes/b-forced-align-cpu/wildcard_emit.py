"""Spike (b), part 3 (DC-11), the model side: emissions, the greedy reading, and forced alignments.

Run with the QA worker's interpreter, the worker's ``src`` on ``PYTHONPATH`` and the thread cap in the
environment (``wildcard.py`` starts it so)::

    workers/qa/.venv/Scripts/python.exe spikes/b-forced-align-cpu/wildcard_emit.py \\
        --snapshot <models_root>/models--facebook--wav2vec2-large-960h-lv60-self/snapshots/<revision> \\
        --jobs <jobs.json> --out <emitted.json>

Each job is ``{wav, start_s, end_s, variants: {name: tokens}}``. For each job the model runs once on the
CPU; the reply holds the frame count, the greedy (unconstrained argmax) reading as words with their first
and end frames, and each variant's token spans from ``Wav2Vec2Aligner.spans`` (the worker's own code, the
wildcard included). Nothing is written but ``--out``; no audio is saved.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from narration_worker_qa import align as qa

REPO = "facebook/wav2vec2-large-960h-lv60-self"


def greedy_words(emission: Any, labels: dict[int, str], blank: int) -> list[list[Any]]:
    """The argmax path's words, each ``[letters, first frame, end frame]`` (ported from ``spike_b.py``)."""
    best = emission.argmax(dim=-1).tolist()
    words: list[list[Any]] = []
    letters, first, last, prev = "", -1, -1, blank
    for frame, label in enumerate(best):
        if label != blank and label != prev:
            char = labels[label]
            if char == "|":
                if letters:
                    words.append([letters, first, last])
                letters = ""
            else:
                if not letters:
                    first = frame
                letters += char
                last = frame + 1
        elif label != blank and label == prev and labels[label] != "|":
            last = frame + 1
        prev = label
    if letters:
        words.append([letters, first, last])
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    aligner = qa.Wav2Vec2Aligner(torch)
    load_s = aligner.load(REPO, args.snapshot.name, args.snapshot)
    vocab = json.loads((args.snapshot / "vocab.json").read_text(encoding="utf-8"))
    labels = {int(v): str(k) for k, v in vocab.items()}
    blank = int(vocab[qa.BLANK])
    out: list[dict[str, Any]] = []
    began = time.perf_counter()
    for job in json.loads(args.jobs.read_text(encoding="utf-8")):
        info = sf.info(job["wav"])
        start = round(float(job["start_s"]) * info.samplerate)
        stop = round(float(job["end_s"]) * info.samplerate)
        audio, rate = sf.read(job["wav"], dtype="float32", start=start, stop=stop, always_2d=True)
        emission = aligner.emission(qa.to_mono_16k(audio, rate))
        replies: dict[str, Any] = {}
        for name, tokens in job["variants"].items():
            try:
                replies[name] = {"ok": True, "spans": aligner.spans(emission, tokens)}
            except qa.AlignerError as exc:
                replies[name] = {"ok": False, "code": exc.code, "details": exc.details}
        out.append(
            {
                "num_frames": int(emission.shape[0]),
                "greedy": greedy_words(emission, labels, blank),
                "replies": replies,
            }
        )
    result = {"load_s": round(load_s, 2), "align_s": round(time.perf_counter() - began, 1), "jobs": out}
    partial = args.out.with_name(args.out.name + ".tmp")
    partial.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8", newline="\n")
    os.replace(partial, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
