"""Fixtures for the store's tests: a store under pytest's tmp_path, a platform stand-in, a clock to move."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from narration.store import NarrationStore

from .standin import StandInPlatform


class FakeClock:
    """Unix seconds that only move when a test says so."""

    def __init__(self, start: float = 1_790_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path: Path, clock: FakeClock) -> Iterator[NarrationStore]:
    with NarrationStore(tmp_path / "store", StandInPlatform(), clock=clock) as s:
        yield s
