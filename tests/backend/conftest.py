"""Fixtures for the backend's tests: the job engine's test world (a store with a pinned engine, a measured
synthetic voice, fake workers) and the backend over it."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from narration.backend import NarrationBackend
from tests.jobs.conftest import World, anchor, make_world
from tests.jobs.support import VOICE_TRANSCRIPT

from .support import FakeLauncher, TestPlatform, make_backend

__all__ = ["anchor"]


@dataclass
class Service:
    """The backend, its platform and launcher, and the job engine's world beneath them."""

    world: World
    backend: NarrationBackend
    platform: TestPlatform
    launcher: FakeLauncher

    def voice(self, **changes: Any) -> dict[str, Any]:
        """The test voice as a request sends it."""
        return {
            "path": str(self.world.clip),
            "sha256": self.world.clip_sha256,
            "transcript": VOICE_TRANSCRIPT,
            **changes,
        }

    def request(self, *texts: str, **options: Any) -> dict[str, Any]:
        """A ``submit_job`` request with one segment per text."""
        return self.world.request(*texts, **options)


@pytest.fixture
def service(tmp_path: Path, anchor: tuple[float, ...]) -> Iterator[Service]:
    world = make_world(tmp_path, anchor)
    backend, platform, launcher = make_backend(world)
    try:
        yield Service(world=world, backend=backend, platform=platform, launcher=launcher)
    finally:
        world.pool.close()
        world.store.close()
