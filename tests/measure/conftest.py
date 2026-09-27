"""Fixtures for the ``measure_voice`` tests: a world with an unmeasured voice and the ``measure`` handler."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from .support import MeasureWorld, make_world


@pytest.fixture(autouse=True)
def _no_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    """A measurement publishes some forty files; these tests check what is published, not that it survives a
    power cut, so the store's fsyncs are skipped (they are most of a measurement's time on the fake)."""
    monkeypatch.setattr(os, "fsync", lambda fd: None)


@pytest.fixture
def world(tmp_path: Path) -> Iterator[MeasureWorld]:
    w = make_world(tmp_path)
    try:
        yield w
    finally:
        w.close()
