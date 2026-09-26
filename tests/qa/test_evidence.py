"""The bake-off's evidence, reproduced (plan.md WP14 acceptance; design section 1, "Names probe").

Marker ``evidence``: these tests read the bake-off's files through ``NARRATION_BAKEOFF_ROOT`` and skip cleanly
without it. Nothing is copied from there: the probe's texts, transcripts, names and numbers are read at run
time. The probe's text is a caller's script, used here as evidence only (design section 1), never as the
service's test material.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from narration.config import MeasurementConfig
from narration.contracts import codes
from narration.contracts.models import Hint, SimilarityBaseline
from narration.qa.checks import speaker_check
from narration.qa.normaliser import NumberReader
from narration.qa.profile import DEFAULT_PROFILE
from narration.qa.textmatch import TextMatch, match_text
from narration.text import words

from .builders import ANCHOR, VOICE_TRANSCRIPT, Cue, segment, with_similarity

pytestmark = pytest.mark.evidence

READER = NumberReader()
SCRIPT = "r48_names_probe"
PREFIX = "qwen3-tts-1.7b-clone-"


@dataclass(frozen=True)
class Take:
    name: str  # e.g. "d2-late-night_take1-seed1"
    segments: tuple[dict[str, Any], ...]  # the rendered segments: id and text
    evaluation: dict[str, Any]  # the bake-off's eval.json: transcripts and WERs


@pytest.fixture(scope="module")
def takes(bakeoff_root: Path) -> list[Take]:
    found = []
    for path in sorted((bakeoff_root / "outputs").glob(f"{PREFIX}*/{SCRIPT}.eval.json")):
        meta = json.loads(path.with_name(f"{SCRIPT}.json").read_text(encoding="utf-8"))
        found.append(
            Take(
                name=path.parent.name.removeprefix(PREFIX),
                segments=tuple(meta["segments"]),
                evaluation=json.loads(path.read_text(encoding="utf-8")),
            )
        )
    assert len(found) == 6, "the probe has 6 clone takes: d2 and d4, seeds 1-3"
    return found


@pytest.fixture(scope="module")
def results_csv(bakeoff_root: Path) -> dict[str, dict[str, str]]:
    with (bakeoff_root / "eval" / "results.csv").open(encoding="utf-8", newline="") as f:
        return {row["model"].removeprefix(PREFIX): row for row in csv.DictReader(f)}


@pytest.fixture(scope="module")
def names_table(bakeoff_root: Path) -> tuple[list[str], list[list[str]]]:
    """The bake-off's names report: its take columns, and its rows (segment, name, one cell per take, same?)."""
    lines = (bakeoff_root / "eval" / f"{SCRIPT}.names.md").read_text(encoding="utf-8").splitlines()
    table = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines if line.startswith("|")]
    header, rows = table[0], table[2:]
    return header[2:-1], rows


def numbers_as_exact_spans(seg: dict[str, Any]) -> Cue:
    """The segment as one cue, with every word that holds a digit marked as an exact span."""
    text: str = seg["text"]
    spans = tuple((w.index, w.index + 1) for w in words(text) if any(c.isdigit() for c in w.text))
    return Cue(text, exact=spans)


def score_segment(seg: dict[str, Any], transcript: str, hints: tuple[Hint, ...] = ()) -> TextMatch:
    return match_text(
        segment(numbers_as_exact_spans(seg), segment_id=seg["id"]),
        hints,
        transcript,
        VOICE_TRANSCRIPT,
        READER,
        DEFAULT_PROFILE,
    )


def test_names_probe_numbers_have_zero_mismatches_s11_3(takes: list[Take]) -> None:
    """Design section 1: "Numbers: zero mismatches across all 48 paragraph-takes", though Whisper writes some
    numbers as words and some as digits."""
    paragraph_takes = spans = 0
    for take in takes:
        for seg, heard in zip(take.segments, take.evaluation["segments"], strict=True):
            assert seg["id"] == heard["id"]
            match = score_segment(seg, heard["transcript"])
            paragraph_takes += 1
            spans += len(match.exact)
            assert [e for e in match.exact if e.match != "same"] == [], (take.name, seg["id"])
    assert paragraph_takes == 48
    assert spans == 6 * 24


def test_names_probe_raw_wer_reproduced_s11_1(takes: list[Take], results_csv: dict[str, dict[str, str]]) -> None:
    """``wer_raw`` reproduces the bake-off's WER to its 4 decimals, per take (results.csv) and per segment
    (eval.json): 5.5-8.6 % per take, worst segment 16-21 %, all from name spelling."""
    per_take = []
    for take in takes:
        full_ref = " ".join(s["text"] for s in take.segments)
        whole = match_text(
            segment(full_ref), (), take.evaluation["full_transcript"], VOICE_TRANSCRIPT, READER, DEFAULT_PROFILE
        )
        assert whole.wer_raw is not None
        assert round(whole.wer_raw, 4) == float(results_csv[take.name]["wer"]), take.name
        per_take.append(whole.wer_raw)
        worst = 0.0
        for seg, heard in zip(take.segments, take.evaluation["segments"], strict=True):
            raw = score_segment(seg, heard["transcript"]).wer_raw
            assert raw is not None
            assert round(raw, 4) == heard["wer"], (take.name, seg["id"])
            worst = max(worst, raw)
        assert round(worst, 4) == float(results_csv[take.name]["worst_seg_wer"])
        assert 0.15 <= worst <= 0.22
    assert min(per_take) >= 0.054 and max(per_take) <= 0.086


def test_names_probe_wer_adj_collapses_the_names_s11_1(
    takes: list[Take], names_table: tuple[list[str], list[list[str]]]
) -> None:
    """With the probe's names sent as hints, what raw WER counted against them no longer fails a segment.

    A name found costs nothing and a name not found costs one substitution, however many words it was heard
    as. Over the 48 paragraph-takes: no ``WER_HIGH`` fail, two warnings (the independent review's count), and
    ``wer_adj`` at most ``wer_raw`` in every segment.
    """
    _, rows = names_table
    hints = tuple(Hint(term=name) for name in sorted({row[1] for row in rows}))
    severities: list[str] = []
    for take in takes:
        for seg, heard in zip(take.segments, take.evaluation["segments"], strict=True):
            match = score_segment(seg, heard["transcript"], hints)
            assert match.wer_adj is not None and match.wer_raw is not None
            assert match.wer_adj <= match.wer_raw, (take.name, seg["id"])
            severities += [f.severity for f in match.flags if f.code == codes.WER_HIGH]
    assert sorted(severities) == ["warn", "warn"]


def test_names_probe_names_heard_exactly_are_verified_s11_1(
    takes: list[Take], names_table: tuple[list[str], list[list[str]]]
) -> None:
    """Every cell the bake-off's names report marks as heard exactly (✓) is a verified term here; so the name
    that design section 1 reports heard exactly in 12/12 occurrences is verified 12 times. The names are read
    from the report at run time."""
    columns, rows = names_table
    hints = tuple(Hint(term=name) for name in sorted({row[1] for row in rows}))
    by_take = {t.name: t for t in takes}
    exact_cells = 0
    cells_of: dict[str, int] = {}
    exact_of: dict[str, int] = {}
    verified: dict[str, int] = {}
    for row in rows:
        seg_id, name, cells = row[0], row[1], row[2:-1]
        for column, cell in zip(columns, cells, strict=True):
            cells_of[name] = cells_of.get(name, 0) + 1
            if cell != "✓":
                continue
            take = by_take[column]
            index = next(i for i, s in enumerate(take.segments) if s["id"] == seg_id)
            match = score_segment(take.segments[index], take.evaluation["segments"][index]["transcript"], hints)
            results = [t for t in match.terms if t.term == name]
            assert results and all(t.ok for t in results), (column, seg_id, name)
            exact_cells += 1
            exact_of[name] = exact_of.get(name, 0) + 1
            verified[name] = verified.get(name, 0) + len(results)
    assert exact_cells > 0
    assert any(cells_of[n] == exact_of.get(n, 0) == verified.get(n, 0) == 12 for n in cells_of)


def test_d4_similarities_not_flagged_under_a_d4_calibrated_threshold_s11_1(
    results_csv: dict[str, dict[str, str]],
) -> None:
    """Design section 11.1: the probe's d4 scored 0.966-0.969 against its clip and d2 0.978-0.984, so a fixed
    0.975 would flag every d4 take. Calibrated on d4's own similarities (the baseline's p5, minus the margin),
    none is flagged.

    The probe recorded one similarity per take (a mean over its 8 segments), not a measurement's calibration
    set, so the calibration here uses those three per-take values as the baseline sample.
    """
    config = MeasurementConfig()
    d4 = [float(row["spk_to_ref"]) for name, row in results_csv.items() if name.startswith("d4-")]
    d2 = [float(row["spk_to_ref"]) for name, row in results_csv.items() if name.startswith("d2-")]
    assert len(d4) == len(d2) == 3
    assert min(d4) >= 0.966 and max(d4) <= 0.970
    assert min(d2) >= 0.977 and max(d2) <= 0.984

    p5 = float(np.percentile(d4, 5))
    baseline = SimilarityBaseline(anchor_p5=p5, anchor_p50=float(np.median(d4)), consistency_p5=p5)
    for sim in d4:
        check = speaker_check(with_similarity(sim), ANCHOR, baseline, config)
        assert check.flags == (), sim
    # Against a fixed 0.975 (what a d2-like voice would need), every d4 take would warn.
    fixed = SimilarityBaseline(anchor_p5=0.975 + config.sim_warn_margin, anchor_p50=0.99, consistency_p5=0.99)
    for sim in d4:
        check = speaker_check(with_similarity(sim), ANCHOR, fixed, config)
        assert [f.severity for f in check.flags] == ["warn"], sim
