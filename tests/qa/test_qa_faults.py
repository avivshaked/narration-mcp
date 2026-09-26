"""The service's planted-fault QA fixtures (``material/fixtures/qa-faults-v1``, design section 20 Phase 3),
run through the scorer as pure QA: the spec's ``asr.text`` is the transcript, and every check outside the
spec (speaker, pace, signal, the other cues' alignment) is clean.

Faults must be caught and non-faults must pass. The spec's segment is as sent (section 7.2); the text pipeline
(WP10) plans it, so its exact spans and hints are what a real request would give QA.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest

from narration.contracts.models import CueTiming, Hint, QaResult, SegmentText
from narration.contracts.serial import from_json
from narration.qa import Scorer

from .builders import VOICE_TRANSCRIPT, alignment, inputs, placed_cues, planned

SPECS = Path(__file__).resolve().parents[2] / "material" / "fixtures" / "qa-faults-v1" / "specs.json"


@functools.cache
def specs() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(SPECS.read_text(encoding="utf-8"))["specs"])


def hints(spec: dict[str, Any]) -> list[Hint]:
    return [from_json(Hint, h) for h in spec["hints"]]


def as_segment(spec: dict[str, Any]) -> SegmentText:
    """The spec's segment as the text pipeline plans it."""
    cues = [(c["text"], [(e["start"], e["end"]) for e in c.get("exact", ())]) for c in spec["segment"]["cues"]]
    return planned(*cues, hints=hints(spec), segment_id=spec["segment"]["segment_id"])


def score(spec: dict[str, Any]) -> QaResult:
    seg = as_segment(spec)
    unplaced = {c["index"] for c in spec["expect"].get("cues", ()) if c["start_s"] is None}
    cues = tuple(
        CueTiming(index=c.index, start_s=None, end_s=None, confidence=None) if c.index in unplaced else c
        for c in placed_cues(seg)
    )
    render = spec["render"]
    return Scorer().score(
        inputs(
            seg,
            spec["asr"]["text"],
            hints=hints(spec),
            align=alignment(seg, cues=cues),
            hit_token_cap="max_new_tokens" in render and render["generated_tokens"] >= render["max_new_tokens"],
            voice_transcript=spec.get("voice_transcript", VOICE_TRANSCRIPT),
        )
    )


def test_the_fixture_set_is_there_s20() -> None:
    kinds = [s["kind"] for s in specs()]
    assert kinds.count("fault") >= 4 and kinds.count("non-fault") >= 3


@pytest.mark.parametrize("spec", specs(), ids=lambda s: s["name"])
def test_planted_faults_are_caught_and_non_faults_pass_s20(spec: dict[str, Any]) -> None:
    result = score(spec)
    expect = spec["expect"]
    raised = [(f.code, f.severity, f.cue) for f in result.flags]
    for want in expect.get("flags", ()):
        assert any(
            (code, severity) == (want["code"], want["severity"]) and ("cue" not in want or cue == want["cue"])
            for code, severity, cue in raised
        ), (want, raised)
    for code in expect.get("absent", ()):
        assert code not in {c for c, _, _ in raised}, raised
    for pair in expect.get("absent_severity", ()):
        assert (pair["code"], pair["severity"]) not in {(c, s) for c, s, _ in raised}, raised
    for want, got in zip(expect.get("exact", ()), result.exact, strict=True):
        assert got.cue == want["cue"] and got.match == want["match"]
        assert (got.expected, got.heard) == (want.get("expected", got.expected), want.get("heard", got.heard))
    if "wer_adj" in expect:
        assert result.metrics.wer_adj == pytest.approx(expect["wer_adj"]["value"], abs=5e-5)
        assert result.metrics.word_errors == expect["wer_adj"]["word_errors"]
    if "retake_trigger" in expect:
        assert any(f.retake_trigger for f in result.flags) == expect["retake_trigger"]
    if "verdict" in expect:
        assert result.verdict == expect["verdict"]
    if spec["kind"] == "non-fault":
        assert result.verdict != "fail"
