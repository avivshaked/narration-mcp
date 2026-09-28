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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hi.", 128),  # the floor: about 10 s of audio for the shortest text
        ("a" * 51, 128),  # 127.5 rounds up to 128, the floor anyway
        ("a" * 52, 130),
        ("a" * 150, 375),
        ("a" * 3276, 8190),
        ("a" * 3277, 8192),  # 8192.5 is capped at the ceiling
        ("a" * 4000, 8192),
    ],
)
def test_max_new_tokens_is_the_per_call_cap_dc4_s10_1(text: str, expected: int) -> None:
    assert names.max_new_tokens_for(text, per_char=2.5, floor=128, ceiling=8192) == expected


def test_max_new_tokens_counts_code_points_and_honours_a_lower_ceiling_dc4() -> None:
    assert names.max_new_tokens_for("é" * 100, per_char=2.5, floor=2, ceiling=8192) == 250
    assert names.max_new_tokens_for("a" * 1000, per_char=2.5, floor=128, ceiling=2048) == 2048
    assert names.MAX_NEW_TOKENS_CEILING == 8192


def test_max_new_tokens_is_exact_decimal_arithmetic_dc4() -> None:
    # A float ceil(1.1 * 50) is 56 (1.1 * 50 = 55.00000000000001); the cap reads 1.1 as 11/10, so it is 55.
    assert names.max_new_tokens_for("a" * 50, per_char=1.1, floor=2, ceiling=8192) == 55
    assert names.max_new_tokens_for("a" * 5, per_char=2.4, floor=2, ceiling=8192) == 12
    assert names.max_new_tokens_for("a" * 10, per_char=0.7, floor=2, ceiling=8192) == 7
    assert names.max_new_tokens_for("a", per_char=0.1, floor=2, ceiling=8192) == 2


@pytest.mark.parametrize(
    ("per_char", "floor", "ceiling"), [(0.0, 128, 8192), (2.5, 1, 8192), (2.5, 128, 1), (-1.0, 128, 8192)]
)
def test_max_new_tokens_refuses_a_meaningless_rule_dc4(per_char: float, floor: int, ceiling: int) -> None:
    with pytest.raises(ValueError, match="per_char"):
        names.max_new_tokens_for("text", per_char=per_char, floor=floor, ceiling=ceiling)


def test_the_qa_profile_is_default_v4_dc19() -> None:
    """Contracts 1.6.7 (DC-19, the owner's decision of 2026-09-27): PACE_FAST warns only. The profile's version
    is in the analysis key, so takes scored under default.v3 are scored again."""
    from narration.qa.profile import DEFAULT_PROFILE

    assert names.QA_PROFILE == "default.v4"
    assert DEFAULT_PROFILE.name == names.QA_PROFILE
    assert DEFAULT_PROFILE.pace_fail_tol_factor is None
