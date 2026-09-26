"""Spike (b), part 4 (the WP15 review's F1): how far a cue's aligned edge lies from the pause next to it.

``narration.align`` snaps a cue's edge into a pause. Next to speech the cue does not own (an unplaceable
cue's), it snaps only when at most ``snap_reach_start_s`` / ``snap_reach_end_s`` of speech lies between
the pause and the cue's first / last aligned letter; more than that, and the pause belongs to the far side
of the other cue's speech. This script measures that distance where every cue is placed and every edge is
the cue's own, so it shows how much of it is the aligner's own imprecision at a word's edge.

The inputs are the evidence tests' 24 paragraph pairs (``tests/align/test_bakeoff_evidence_s11_2.py``): the
bake-off's six clone takes, one cue per sentence. For each placed cue, before any snapping:

- ``start``: the start of its first aligned span minus the end of the nearest pause before it;
- ``end``: the start of the nearest pause after its last aligned span minus that span's end.

Each is split by the edge's kind (the file's edge, a sentence boundary, the paragraph boundary) and by
whether the span is a letter word or a wildcard (DC-11). Run it from the checkout with the server's venv,
``NARRATION_BAKEOFF_ROOT`` and ``NARRATION_MODELS_ROOT`` set, and a scratch folder inside the project::

    uv run python spikes/b-forced-align-cpu/reach.py --scratch .dev/<folder>

It starts ``python -m narration_worker_qa.align`` with the QA worker's venv, reads the takes in place (never
copied), and writes ``reach.json`` next to itself: numbers only, keyed by take, pair and cue index. The
bake-off's text is private; none of it is written. It reads ``narration.align``'s internal steps
(``_place_words``, ``_decide_placement``) to see the edges before snapping.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf

from narration.align import CtcAligner, find_pauses
from narration.align import core as align_core
from narration.contracts.models import CueIn, SegmentIn
from narration.contracts.worker import AlignReply
from narration.text import TextPipeline

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1]
REPO = "facebook/wav2vec2-large-960h-lv60-self"
REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
SNAPSHOT = Path("models--facebook--wav2vec2-large-960h-lv60-self") / "snapshots" / REVISION
SCRIPT = "r48_names_probe"
PAIRS = (("n01", "n02"), ("n03", "n04"), ("n05", "n06"), ("n07", "n08"))
THREADS = "4"


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def qa_python() -> Path:
    venv = CHECKOUT / "workers" / "qa" / ".venv"
    for candidate in (venv / "Scripts" / "python.exe", venv / "bin" / "python"):
        if candidate.is_file():
            return candidate
    raise SystemExit("the QA worker's venv is not synced (workers/qa/.venv)")


def run_worker(jobs: list[dict[str, Any]], scratch: Path, snapshot: Path) -> list[dict[str, Any]]:
    jobs_path, out_path = scratch / "reach_jobs.json", scratch / "reach_replies.json"
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(CHECKOUT / "workers" / "qa" / "src"),
        OMP_NUM_THREADS=THREADS,
        MKL_NUM_THREADS=THREADS,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    argv = [str(qa_python()), "-m", "narration_worker_qa.align", "--snapshot", str(snapshot), "--repo", REPO]
    argv += ["--revision", REVISION, "--jobs", str(jobs_path), "--out", str(out_path), "--threads", THREADS]
    subprocess.run(argv, env=env, check=True, timeout=1800)
    return json.loads(out_path.read_text(encoding="utf-8"))["replies"]


def stats(values: list[float]) -> dict[str, Any]:
    xs = sorted(values)
    if not xs:
        return {"n": 0}

    def q(p: float) -> float:
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    return {"n": len(xs), "min": xs[0], "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "max": xs[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True, help="a folder inside the project for the worker's files")
    args = parser.parse_args()
    bakeoff = Path(os.environ["NARRATION_BAKEOFF_ROOT"])
    snapshot = Path(os.environ["NARRATION_MODELS_ROOT"]) / SNAPSHOT
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    aligner = CtcAligner(revision=REVISION)
    pipeline = TextPipeline()
    jobs: list[dict[str, Any]] = []
    todo: list[tuple[Path, str, str, Any, int, int, int]] = []
    takes = sorted(p for p in (bakeoff / "outputs").iterdir() if (p / f"{SCRIPT}.json").is_file())
    for take in takes:
        meta = json.loads((take / f"{SCRIPT}.json").read_text(encoding="utf-8"))
        rate = int(meta["sample_rate"])
        paragraphs = {p["id"]: p for p in meta["segments"]}
        for first, second in PAIRS:
            a, b = paragraphs[first], paragraphs[second]
            first_cues, second_cues = sentences(a["text"]), sentences(b["text"])
            cues = tuple(CueIn(text=c) for c in (*first_cues, *second_cues))
            segment = pipeline.plan_segment(SegmentIn(segment_id=f"{first}-{second}", cues=cues), [])
            transcript = aligner.build_transcript(segment, [])
            start, end = int(a["start_sample"]), int(b["end_sample"])
            wav = take / f"{SCRIPT}.wav"
            jobs.append(
                {"wav": str(wav), "tokens": list(transcript.tokens), "start_s": start / rate, "end_s": end / rate}
            )
            todo.append((take, first, second, transcript, len(first_cues), start, end))

    replies = run_worker(jobs, scratch, snapshot)
    rows: list[dict[str, Any]] = []
    tol = aligner.params.snap_tolerance_s
    for (take, first, second, transcript, first_count, start, end), reply in zip(todo, replies, strict=True):
        assert reply.get("ok") is True, (take.name, first, second)
        data, rate = sf.read(str(take / f"{SCRIPT}.wav"), dtype="float32", start=start, stop=end, always_2d=True)
        pauses = find_pauses(np.asarray(data, dtype=np.float32), rate, aligner.params.pauses)
        cues = align_core._cues(transcript)
        aligner._place_words(transcript, cast(AlignReply, reply), cues)
        for cue in cues:
            aligner._decide_placement(cue)
        last = len(cues) - 1
        for cue in (c for c in cues if c.placed):
            head, tail = cue.spans[0], cue.spans[-1]
            before = [p.end_s for p in pauses if p.start_s < head.start_s + tol]
            after = [p.start_s for p in pauses if p.end_s > tail.end_s - tol]
            rows.append(
                {
                    "take": take.name,
                    "pair": f"{first}+{second}",
                    "cue": cue.index,
                    "start_kind": "file" if cue.index == 0 else "paragraph" if cue.index == first_count else "sentence",
                    "start_span": "letters" if head.word is not None else "wildcard",
                    "start_s": round(head.start_s - max(before), 3) if before else None,
                    "end_kind": "file"
                    if cue.index == last
                    else "paragraph"
                    if cue.index == first_count - 1
                    else "sentence",
                    "end_span": "letters" if tail.word is not None else "wildcard",
                    "end_s": round(min(after) - tail.end_s, 3) if after else None,
                }
            )

    summary: dict[str, Any] = {}
    for side in ("start", "end"):
        for span in ("letters", "wildcard"):
            for kind in ("file", "sentence", "paragraph", "all"):
                values = [
                    r[f"{side}_s"]
                    for r in rows
                    if r[f"{side}_s"] is not None
                    and r[f"{side}_span"] == span
                    and (kind == "all" or r[f"{side}_kind"] == kind)
                ]
                summary[f"{side}/{span}/{kind}"] = stats(values)
    out = {
        "spike": "b, part 4: the reach of a snap (WP15 review F1)",
        "date": dt.date.today().isoformat(),
        "model": f"{REPO}@{REVISION}",
        "pauses": {
            "frame_s": aligner.params.pauses.frame_s,
            "silence_below_db": aligner.params.pauses.silence_below_db,
        },
        "takes": len(takes),
        "pairs": len(todo),
        "summary": summary,
        "rows": rows,
    }
    (HERE / "reach.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
