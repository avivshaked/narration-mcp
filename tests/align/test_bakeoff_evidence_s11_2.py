"""The aligner end to end on the bake-off's real clone takes (design section 11.2; plan.md WP15 acceptance).

A sanity check that cue times look right, not an accuracy measurement: that needs hand marks, and is the
alignment benchmark's (WP38). Each take has eight paragraphs, rendered one by one and joined with 0.9 s of
inserted silence. Two neighbouring paragraphs are aligned as one segment, one cue per sentence, so the
boundary between them has a known place: it must hold the inserted silence.

The takes and their text are read in place from ``NARRATION_BAKEOFF_ROOT`` at run time and never copied.
The text is private: nothing here quotes it, and no assertion message prints it (pairs, cues and words
are named by id and index). The emissions come from the QA worker's venv (``workers/qa/.venv``), run as a
subprocess with the development entry point ``python -m narration_worker_qa.align``; the rest is
``narration.align`` in this process. The tests skip cleanly without the bake-off, the pinned wav2vec2
snapshot under ``NARRATION_MODELS_ROOT``, or the QA worker's venv. All 24 pairs take about 90 s on 4 CPU
threads.

The bake-off's text holds numbers in digits (callers now send numbers in words, R6), which the aligner
cannot spell. Each run of them is a wildcard (plan.md DC-11); before it, a paragraph ending in such numbers
put the next cue 1.3-3.4 s early in all six takes (``spikes/b-forced-align-cpu/README.md``).
"""

from __future__ import annotations

import itertools
import json
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import soundfile as sf

from narration.align import CtcAligner
from narration.contracts import codes
from narration.contracts.models import Alignment, CueIn, SegmentIn, SegmentText
from narration.contracts.worker import AlignReply
from narration.text import TextPipeline, words

pytestmark = [pytest.mark.model, pytest.mark.evidence, pytest.mark.slow]

REPO = "facebook/wav2vec2-large-960h-lv60-self"
REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
SNAPSHOT = Path("models--facebook--wav2vec2-large-960h-lv60-self") / "snapshots" / REVISION
CHECKOUT = Path(__file__).resolve().parents[2]
SCRIPT = "r48_names_probe"
PAIRS = (("n01", "n02"), ("n03", "n04"), ("n05", "n06"), ("n07", "n08"))
FRAME_S = 0.02
THREADS = "4"
WORKER_TIMEOUT_S = 900


@dataclass(frozen=True, slots=True)
class Pair:
    """Two neighbouring paragraphs of one take, aligned as one segment."""

    take: str
    first: str
    second: str
    segment: SegmentText
    alignment: Alignment
    duration_s: float
    first_cues: int
    gap_s: tuple[float, float]
    """The inserted silence between the paragraphs, in seconds from the pair's start."""
    ends_unspellable: bool
    """Whether the first paragraph's last word cannot be spelled in the aligner's alphabet (a wildcard)."""

    @property
    def name(self) -> str:
        return f"{self.take} {self.first}+{self.second}"


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _qa_python() -> Path | None:
    venv = CHECKOUT / "workers" / "qa" / ".venv"
    for candidate in (venv / "Scripts" / "python.exe", venv / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


def _resources() -> tuple[Path, Path, Path]:
    bakeoff = os.environ.get("NARRATION_BAKEOFF_ROOT")
    if not bakeoff:
        pytest.skip("NARRATION_BAKEOFF_ROOT is not set: the bake-off's takes are needed")
    models = os.environ.get("NARRATION_MODELS_ROOT")
    if not models:
        pytest.skip("NARRATION_MODELS_ROOT is not set: the pinned wav2vec2 snapshot is needed")
    snapshot = Path(models) / SNAPSHOT
    if not snapshot.is_dir():
        pytest.skip(f"the pinned snapshot is not installed under NARRATION_MODELS_ROOT: {SNAPSHOT.as_posix()}")
    python = _qa_python()
    if python is None:
        pytest.skip("the QA worker's venv is not synced (workers/qa/.venv): it computes the emissions")
    return Path(bakeoff), snapshot, python


def _takes(bakeoff: Path) -> list[Path]:
    outputs = bakeoff / "outputs"
    takes = sorted(p for p in outputs.iterdir() if (p / f"{SCRIPT}.json").is_file()) if outputs.is_dir() else []
    if not takes:
        pytest.skip(f"no bake-off take of {SCRIPT} under NARRATION_BAKEOFF_ROOT/outputs")
    return takes


def _run_worker(python: Path, snapshot: Path, jobs: list[dict[str, object]], scratch: Path) -> list[dict[str, object]]:
    jobs_path, out_path = scratch / "jobs.json", scratch / "replies.json"
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(CHECKOUT / "workers" / "qa" / "src"),  # the QA worker is not an installed package yet
        OMP_NUM_THREADS=THREADS,
        MKL_NUM_THREADS=THREADS,
        OPENBLAS_NUM_THREADS=THREADS,
        NUMEXPR_NUM_THREADS=THREADS,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    argv = [
        str(python),
        "-m",
        "narration_worker_qa.align",
        "--snapshot",
        str(snapshot),
        "--repo",
        REPO,
        "--revision",
        REVISION,
        "--jobs",
        str(jobs_path),
        "--out",
        str(out_path),
        "--threads",
        THREADS,
    ]
    done = subprocess.run(argv, env=env, capture_output=True, timeout=WORKER_TIMEOUT_S, check=False)
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", errors="replace")[-3000:]
        pytest.fail(f"the aligner exited with {done.returncode}:\n{tail}")
    replies = json.loads(out_path.read_text(encoding="utf-8"))["replies"]
    assert len(replies) == len(jobs)
    return replies


@pytest.fixture(scope="module")
def pairs(tmp_path_factory: pytest.TempPathFactory) -> list[Pair]:
    bakeoff, snapshot, python = _resources()
    aligner = CtcAligner(revision=REVISION)
    pipeline = TextPipeline()
    jobs: list[dict[str, object]] = []
    todo: list[tuple[Path, str, str, SegmentText, int, int, int, int, int]] = []
    for take in _takes(bakeoff):
        meta = json.loads((take / f"{SCRIPT}.json").read_text(encoding="utf-8"))
        rate = int(meta["sample_rate"])
        paragraphs = {p["id"]: p for p in meta["segments"]}
        for first, second in PAIRS:
            a, b = paragraphs[first], paragraphs[second]
            first_cues, second_cues = _sentences(a["text"]), _sentences(b["text"])
            cues = tuple(CueIn(text=c) for c in (*first_cues, *second_cues))
            segment = pipeline.plan_segment(SegmentIn(segment_id=f"{first}-{second}", cues=cues), [])
            transcript = aligner.build_transcript(segment, [])
            start, end = int(a["start_sample"]), int(b["end_sample"])
            wav = take / f"{SCRIPT}.wav"
            jobs.append(
                {"wav": str(wav), "tokens": list(transcript.tokens), "start_s": start / rate, "end_s": end / rate}
            )
            todo.append(
                (
                    take,
                    first,
                    second,
                    segment,
                    len(first_cues),
                    start,
                    end,
                    int(a["end_sample"]),
                    int(b["start_sample"]),
                )
            )
    replies = _run_worker(python, snapshot, jobs, tmp_path_factory.mktemp("wp15-evidence"))
    out: list[Pair] = []
    for (take, first, second, segment, first_cues, start, end, gap_lo, gap_hi), reply in zip(
        todo, replies, strict=True
    ):
        data, rate = sf.read(str(take / f"{SCRIPT}.wav"), dtype="float32", start=start, stop=end, always_2d=True)
        audio = np.asarray(data, dtype=np.float32)
        transcript = aligner.build_transcript(segment, [])
        ok = reply.get("ok") is True
        error = None if ok else reply.get("error")
        alignment = aligner.resolve(
            transcript,
            cast(AlignReply, reply) if ok else None,  # JSON from the worker; resolve checks it fits
            audio,
            rate,
            [],
            None,
            error=error if isinstance(error, dict) else None,
        )
        last_cue = first_cues - 1
        last_word = max(w for c, w, _ in transcript.words if c == last_cue)
        out.append(
            Pair(
                take=take.name,
                first=first,
                second=second,
                segment=segment,
                alignment=alignment,
                duration_s=(end - start) / rate,
                first_cues=first_cues,
                gap_s=((gap_lo - start) / rate, (gap_hi - start) / rate),
                ends_unspellable=any(c == last_cue and w == last_word for c, w, _ in transcript.dropped),
            )
        )
    return out


def _codes(pair: Pair) -> list[str]:
    return [f.code for f in pair.alignment.flags]


def test_every_cue_of_a_real_take_is_placed_in_order_s11_2(pairs: list[Pair]) -> None:
    for pair in pairs:
        cues = pair.alignment.cues
        assert codes.ALIGNMENT_ERROR not in _codes(pair) and codes.CUE_UNALIGNED not in _codes(pair), pair.name
        assert len(cues) == len(pair.segment.cues), pair.name
        for cue in cues:
            assert cue.start_s is not None and cue.end_s is not None, (pair.name, cue.index)
            assert 0.0 <= cue.start_s < cue.end_s <= pair.duration_s + FRAME_S, (pair.name, cue.index)
            placed = [(i, w) for i, w in enumerate(cue.words) if w.start_s is not None and w.end_s is not None]
            assert placed, (pair.name, cue.index)
            (i0, w0), (i1, w1) = placed[0], placed[-1]
            assert w0.start_s is not None and w1.end_s is not None
            # the edge words take the cue's snapped times; an unspellable edge word (a number) may lie beyond them
            assert (w0.start_s == cue.start_s) if i0 == 0 else (w0.start_s >= cue.start_s), (pair.name, cue.index)
            last = len(cue.words) - 1
            assert (w1.end_s == cue.end_s) if i1 == last else (w1.end_s <= cue.end_s), (pair.name, cue.index)
            for (i, word), (_, after) in itertools.pairwise(placed):
                assert word.start_s is not None and word.end_s is not None and after.start_s is not None
                assert word.start_s < word.end_s <= after.start_s, (pair.name, cue.index, i)
        for cue, after in itertools.pairwise(cues):
            assert cue.end_s is not None and after.start_s is not None
            assert cue.end_s <= after.start_s, (pair.name, cue.index)


def _boundary(pair: Pair) -> tuple[float, float]:
    end = pair.alignment.cues[pair.first_cues - 1].end_s
    start = pair.alignment.cues[pair.first_cues].start_s
    assert end is not None and start is not None
    return end, start


def test_paragraph_boundary_holds_the_inserted_silence_s11_2(pairs: list[Pair]) -> None:
    for pair in pairs:
        end, start = _boundary(pair)
        gap_lo, gap_hi = pair.gap_s
        # the boundary is the whole pause: the first paragraph's tail silence, the inserted 0.9 s, the
        # second's lead-in; each side a paragraph's own silence away at most
        assert gap_lo - 1.0 <= end <= gap_lo + FRAME_S, (pair.name, end, pair.gap_s)
        assert gap_hi - FRAME_S <= start <= gap_hi + 1.0, (pair.name, start, pair.gap_s)


def test_boundary_after_unspellable_words_holds_the_inserted_silence_s11_2_dc11(pairs: list[Pair]) -> None:
    # Before the wildcard, the next cue started 1.3-3.4 s early here, in all six takes (spike (b), part 3).
    after_wildcard = [p for p in pairs if p.ends_unspellable]
    assert len(after_wildcard) >= 6, "the bake-off has a paragraph that ends in numbers written in digits"
    for pair in after_wildcard:
        end, start = _boundary(pair)
        assert end <= pair.gap_s[0] + FRAME_S and start >= pair.gap_s[1] - FRAME_S, (pair.name, end, start)


def test_sentence_boundaries_of_a_real_take_are_in_pauses_s11_2(pairs: list[Pair]) -> None:
    for pair in pairs:
        assert codes.CUE_BOUNDARY_NO_PAUSE not in _codes(pair), pair.name


def test_cue_times_give_a_plausible_speaking_rate_s11_2(pairs: list[Pair]) -> None:
    for pair in pairs:
        for cue, text in zip(pair.alignment.cues, pair.segment.cues, strict=True):
            assert cue.start_s is not None and cue.end_s is not None
            rate = len(words(text.spoken)) / (cue.end_s - cue.start_s)
            assert 1.0 <= rate <= 7.0, (pair.name, cue.index, round(rate, 2))


def test_confidence_on_real_speech_is_high_s11_2(pairs: list[Pair]) -> None:
    scores = [c.confidence for p in pairs for c in p.alignment.cues if c.confidence is not None]
    assert len(scores) == sum(len(p.segment.cues) for p in pairs)
    assert statistics.median(scores) >= 0.85
    assert min(scores) >= 0.6
