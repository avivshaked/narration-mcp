"""The id shapes of names.ID_PATTERNS (section 6): one definition for the store, the front end and the schemas."""

from __future__ import annotations

import re

import pytest

from narration.contracts import names


@pytest.mark.parametrize(
    ("kind", "good"),
    [
        ("job_id", "job_01JBXQ7Z3M8V4T2R9K6N5P0W1C"),
        ("design_id", "01JBXQ7Z3M8V4T2R9K6N5P0W1C"),
        ("render_id", "rn_0123456789abcdef"),
        ("take_id", "tk_8c41d2e07a9b3f55"),
        ("analysis_id", "an_0123456789abcdef"),
        ("voice_hash", "sha256:" + "a" * 64),
    ],
)
def test_ids_match_their_whole_shape_s6(kind: names.IdKind, good: str) -> None:
    assert names.is_id(kind, good)
    assert re.search(names.id_schema_pattern(kind), good)
    for bad in (
        good + "\n",
        good + "x",
        " " + good,
        good.swapcase(),
        "",
        "../..",
        7,
    ):
        assert not names.is_id(kind, bad), (kind, bad)
