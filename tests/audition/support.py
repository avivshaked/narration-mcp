"""Support for the ``audition_pronunciation`` tests: the measure tests' world (a store with a pinned Base engine,
a designed clip that is not measured, the fake workers), with the runner over the job engine and the ``measure``
and ``pronunciation`` handlers, as the daemon has them, and the front-end's backend over the same store.

Nothing here needs a GPU or a model: every worker is ``narration_worker``'s ``fake`` role, which renders what the
engine text says, hears it back word for word, and embeds a take close to the clip it cloned. Every term,
respelling and carrier is invented for these tests (design section 9.3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

from narration.audition import audition_registry
from narration.backend.service import NarrationBackend
from narration.contracts.models import JobRecord
from narration.jobs.engine import JobEngine
from narration.jobs.runner import EngineRunner
from narration.measure import build_measure_handler, measure_registry
from tests.backend.support import make_backend
from tests.design.support import queue
from tests.jobs.support import VOICE_TRANSCRIPT
from tests.measure.support import MeasureWorld
from tests.measure.support import make_world as make_measure_world

TERM: Final = "Thorvyn"
"""An invented name."""
CARRIER: Final = "The ferry to Thorvyn leaves at dawn."
"""An invented carrier sentence that holds the term."""
VARIANTS: Final = ({"label": "thor", "respell": "Thor-vin"}, {"label": "tor", "respell": "Tor-veen"})
"""Two invented respellings of the term."""


@dataclass
class AuditionWorld:
    """A measure world with the ``pronunciation`` handler, and the backend over its store."""

    inner: MeasureWorld
    backend: NarrationBackend

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def request(self, **changes: Any) -> dict[str, Any]:
        """An ``audition_pronunciation`` request for the world's clip, as its input schema has it."""
        body: dict[str, Any] = {
            "voice": {"path": str(self.inner.clip), "sha256": self.inner.clip_sha256, "transcript": VOICE_TRANSCRIPT},
            "term": TERM,
            "variants": [dict(v) for v in VARIANTS],
            "carrier": CARRIER,
            **changes,
        }
        return {k: v for k, v in body.items() if v is not None}

    def audition(self, **changes: Any) -> JobRecord:
        """Queue a ``pronunciation`` job as the front-end keeps it (the request by value)."""
        return queue(self.inner.store, "pronunciation", self.request(**changes))

    def results(self, job_id: str, **args: Any) -> dict[str, Any]:
        """``get_results`` for the job, from the front-end."""
        return self.backend.get_results_sync({"job_id": job_id, **args})

    def requests(self, op: str) -> list[Mapping[str, Any]]:
        """Every request of ``op`` the fake workers were sent, in order."""
        return [payload for _, o, payload in self.inner.pool.requests if o == op]

    def restart(self) -> None:
        """A new daemon over the same store and workers: nothing in memory."""
        inner = self.inner
        inner.host.stop_mode = None
        inner.runner.registry.close()
        inner.engine = JobEngine(inner.config, inner.engine.parts)
        inner.handler = build_measure_handler(inner.engine, material_root=inner.material)
        inner.runner = runner(inner)


def runner(world: MeasureWorld) -> EngineRunner:
    """The runner over the world's job engine with the ``measure`` and ``pronunciation`` handlers."""
    return EngineRunner(audition_registry(world.engine, measure_registry(world.engine, world.handler)))


def make_world(root: Path) -> AuditionWorld:
    """The measure tests' world with the ``pronunciation`` handler registered, and the backend over its store."""
    inner = make_measure_world(root)
    inner.runner.registry.close()
    inner.runner = runner(inner)
    backend, _, _ = make_backend(SimpleNamespace(store=inner.store, config=inner.config))
    return AuditionWorld(inner=inner, backend=backend)
