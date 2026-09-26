"""Smoke tests of the package skeleton."""

import narration


def test_version_is_a_string() -> None:
    assert isinstance(narration.__version__, str)
