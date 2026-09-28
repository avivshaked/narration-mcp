"""Fixtures for the ``audition_pronunciation`` tests: a world with an unmeasured designed voice and the fake
workers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from .support import AuditionWorld, make_world


@pytest.fixture(autouse=True)
def _no_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests check what is published, not that it survives a power cut: the store's fsyncs are skipped."""
    monkeypatch.setattr(os, "fsync", lambda fd: None)


@pytest.fixture
def world(tmp_path: Path) -> Iterator[AuditionWorld]:
    w = make_world(tmp_path)
    try:
        yield w
    finally:
        w.close()
