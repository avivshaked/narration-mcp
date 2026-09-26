"""The front-end's exact validator: ECMA-262 anchors in ``pattern`` (design section 14)."""

from __future__ import annotations

import re

import pytest

from narration.mcp.validation import ExactValidator, ecma_anchors


@pytest.mark.parametrize(
    ("pattern", "rewritten"),
    [
        ("a$|b$", r"a\Z|b\Z"),
        ("(a$)", r"(a\Z)"),
        ("^a$", r"^a\Z"),
        ("[$]", "[$]"),
        ("[^$]$", r"[^$]\Z"),
        ("[]$]", "[]$]"),
        ("[^]$]x$", r"[^]$]x\Z"),
        (r"\$", r"\$"),
        (r"\\$", r"\\\Z"),
        (r"a\\\$", r"a\\\$"),
        ("no anchor", "no anchor"),
    ],
)
def test_every_unescaped_dollar_outside_a_class_becomes_end_of_string_s14(pattern: str, rewritten: str) -> None:
    assert ecma_anchors(pattern) == rewritten


@pytest.mark.parametrize(
    ("pattern", "matches", "refuses"),
    [
        ("a$|b$", ["a", "b"], ["a\n", "b\n"]),
        ("(a$)", ["a"], ["a\n"]),
        ("[$]", ["$", "a$b"], ["a"]),
        (r"\$", ["$"], ["a"]),
        (r"\\$", ["x\\"], ["x\\\n", "x"]),
    ],
)
def test_a_trailing_newline_never_satisfies_an_anchor_s14(pattern: str, matches: list[str], refuses: list[str]) -> None:
    compiled = re.compile(ecma_anchors(pattern))
    for text in matches:
        assert compiled.search(text), (pattern, text)
    for text in refuses:
        assert not compiled.search(text), (pattern, text)
    assert re.search(pattern, refuses[0]) or pattern in {"[$]", r"\$"}, "Python's own $ is the laxer one"


def test_the_exact_validator_applies_the_anchors_in_any_position_s14() -> None:
    validator = ExactValidator({"type": "string", "pattern": "^(a$|b$)"})
    assert validator.is_valid("a")
    assert not validator.is_valid("a\n")
    assert not validator.is_valid("b\n")
