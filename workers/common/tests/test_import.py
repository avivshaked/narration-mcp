"""Smoke test: the shared worker package imports with the standard library alone."""

import narration_worker


def test_version_is_a_string() -> None:
    assert isinstance(narration_worker.__version__, str)
