"""The service's own text fixtures (``material/fixtures/text-v1/cases.json``, WP18) through the planner.

Each case is design section 9.3 material: an input segment (with request hints and ``strict_text``) and
the outcome the design fixes. Only the fields a case lists under ``expect`` are asserted; a cue's
``warnings`` list, when present, is the complete list of findings for that cue. The fixture set is read
from the checkout; the test skips, saying so, when the set is not there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from narration.contracts.errors import NarrationError
from narration.contracts.models import Hint, SegmentIn, SegmentText
from narration.contracts.serial import from_json, to_json
from narration.text import TextPipeline, words

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "material" / "fixtures" / "text-v1" / "cases.json"


def _cases() -> list[dict[str, Any]]:
    if not CASES.is_file():
        return []
    return json.loads(CASES.read_text(encoding="utf-8"))["cases"]


def _run(case: dict[str, Any]) -> SegmentText | NarrationError:
    given = case["input"]
    segment = from_json(SegmentIn, given["segment"])
    hints = [from_json(Hint, h) for h in given.get("hints", [])]
    try:
        return TextPipeline().plan_request([segment], hints, strict_text=given.get("strict_text", False))[0]
    except NarrationError as exc:
        return exc


def _check_cue(cue: Any, expected: dict[str, Any]) -> None:
    for key in ("spoken", "engine"):
        if key in expected:
            assert getattr(cue, key) == expected[key], key
    for key in ("spoken_span", "engine_span"):
        if key in expected:
            assert list(getattr(cue, key)) == expected[key], key
    if "hints_applied" in expected:
        got = [{"term": h.term, "respell": h.respell, "offset": h.offset} for h in cue.hints_applied]
        assert got == expected["hints_applied"]
    if "warnings" in expected:
        got = [{**(w.details or {}), "severity": w.severity} for w in cue.warnings]
        for g, e in zip(got, expected["warnings"], strict=False):
            assert {k: g.get(k) for k in e} == e
        assert len(got) == len(expected["warnings"])
    if "exact" in expected:
        cue_words = words(cue.spoken)
        assert len(cue.exact) == len(expected["exact"])
        for got, want in zip(cue.exact, expected["exact"], strict=True):
            first, end = got.words
            if "word_range" in want:
                assert [first, end] == want["word_range"]
            if "words" in want:
                assert [w.text for w in cue_words[first:end]] == want["words"]


@pytest.mark.skipif(not CASES.is_file(), reason="material/fixtures/text-v1 is not in this checkout (WP18)")
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_text_v1_fixture_case_s9_3(case: dict[str, Any]) -> None:
    expect = case["expect"]
    result = _run(case)
    if expect["outcome"] == "refused":
        assert isinstance(result, NarrationError), f"expected a refusal, got {result!r}"
        for key, value in expect["error"].items():
            assert getattr(result, key) == value, key
        offenders = (result.details or {}).get("offenders", [])
        for got, want in zip(offenders, expect.get("offenders", []), strict=False):
            assert {k: got.get(k) for k in want} == want
        assert len(offenders) >= len(expect.get("offenders", []))
        return
    assert isinstance(result, SegmentText), f"expected ok, got {result!r}"
    for key in ("spoken", "engine", "spoken_chars"):
        if key in expect:
            value = {"spoken": result.spoken_text, "engine": result.engine_text, "spoken_chars": result.spoken_chars}
            assert value[key] == expect[key], key
    if "cues" in expect:
        assert len(expect["cues"]) <= len(result.cues)
        for cue, expected in zip(result.cues, expect["cues"], strict=False):
            _check_cue(cue, expected)
    if "flags" in expect:
        got = [to_json(f) for f in result.warnings]
        assert len(got) == len(expect["flags"])
        for flag, want in zip(got, expect["flags"], strict=True):
            assert {k: flag.get(k) for k in want} == want
