"""The service's own spoken-form material (``material/``, WP18) raises no text warning (plan.md WP18's
acceptance: "WP10's checks raise no text warning on any of it"), and its recorded spoken lengths are the
planner's. The canary's voice description passes the positive-only lint (section 3.5).

The sets are read from the checkout; each test skips, saying so, when its set is not there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from narration.contracts.models import SegmentIn
from narration.contracts.serial import from_json
from narration.lint import lint
from narration.text import TextPipeline

MATERIAL = Path(__file__).resolve().parents[2] / "material"
SEGMENT_KEYS = {"segment_id", "cues", "text", "attempts", "scene_seconds", "fit", "controls"}


def _load(relative: str) -> Any:
    path = MATERIAL / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _segments() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    calibration = _load("calibration/narration-en.v1/paragraphs.json") or {}
    design = [calibration["design_text"]] if "design_text" in calibration else []
    out += [("narration-en.v1", p) for p in design + calibration.get("calibration", []) + calibration.get("ladder", [])]
    for name, relative in (
        ("alignment-en.v1", "alignment/alignment-en.v1/paragraphs.json"),
        ("demo-en.v1", "demo/demo-en.v1/paragraphs.json"),
    ):
        out += [(name, p) for p in (_load(relative) or {}).get("paragraphs", [])]
    specs = (_load("fixtures/qa-faults-v1/specs.json") or {}).get("specs", [])
    out += [("qa-faults-v1", s["segment"]) for s in specs]
    return out


SEGMENTS = _segments()


@pytest.mark.skipif(not SEGMENTS, reason="the service's spoken-form material is not in this checkout (WP18)")
@pytest.mark.parametrize("item", SEGMENTS, ids=lambda item: f"{item[0]}/{item[1]['segment_id']}")
def test_service_material_raises_no_text_warning_s9_1(item: tuple[str, dict[str, Any]]) -> None:
    _, raw = item
    segment = from_json(SegmentIn, {k: v for k, v in raw.items() if k in SEGMENT_KEYS})
    planned = TextPipeline().plan_segment(segment, [])
    assert [w.details for c in planned.cues for w in c.warnings] == []
    assert planned.warnings == ()
    if "spoken_chars" in raw:
        assert planned.spoken_chars == raw["spoken_chars"]


@pytest.mark.skipif(_load("canary/canary.v1/canary.json") is None, reason="the canary set is not in this checkout")
def test_canary_description_passes_the_lint_s3_5() -> None:
    canary = _load("canary/canary.v1/canary.json")
    assert lint(canary["voice"]["description"]).findings == ()
