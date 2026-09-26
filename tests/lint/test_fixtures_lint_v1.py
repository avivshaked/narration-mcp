"""The service's own lint fixtures (``material/fixtures/text-v1/lint.json``, WP18) through the lint.

Each case is a voice description and the findings design section 3.5 fixes for it. The fixture set is read
from the checkout; the test skips, saying so, when the set is not there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from narration.lint import lint

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "material" / "fixtures" / "text-v1" / "lint.json"


def _cases() -> list[dict[str, Any]]:
    if not CASES.is_file():
        return []
    return json.loads(CASES.read_text(encoding="utf-8"))["cases"]


@pytest.mark.skipif(not CASES.is_file(), reason="material/fixtures/text-v1 is not in this checkout (WP18)")
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_lint_v1_fixture_case_s3_5(case: dict[str, Any]) -> None:
    description = case["description"]
    got = lint(description).findings
    expected = case["expect"]["findings"]
    assert len(got) == len(expected)
    for finding, want in zip(got, expected, strict=True):
        assert finding.offset == want["offset"]
        assert description[finding.offset :].startswith(want["trigger"])
        assert finding.phrase.startswith(want["trigger"])
        assert (finding.phrase, finding.suggestion) == (want["phrase"], want["suggestion"])
