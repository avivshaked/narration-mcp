"""Job handlers by kind (plan.md WP31; WP33 to WP35 add theirs): what the runner dispatches a claimed job to.

A handler runs the jobs of some kinds a piece at a time, as ``JobEngine`` runs ``generate`` and ``analyse``:
``open`` plans a claimed job into a run, ``advance`` does one piece of it, ``save`` writes its state
between pieces, and ``cancel``, ``fail`` and ``release`` end it. The runner does everything else: claiming,
preemption, cancels, stops and giving a job back.

The ``Registry`` maps each kind to its handler, with what every handler shares: the GPU residency (one
resident model group across every kind, section 4) and the throughput estimate for the queue's drain. A
kind with no handler fails its job with ``INTERNAL`` (``details.kind``): this build does not run it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from narration.contracts.errors import NarrationError
from narration.contracts.models import JobRecord

from .admission import Throughput
from .engine import KINDS, JobEngine
from .gpu import Residency
from .host import RunnerHost
from .state import Outcome

R = TypeVar("R")


class JobHandler(Protocol[R]):
    """Runs the jobs of some kinds, a piece at a time (see the module docstring). ``R`` is its run type."""

    def open(self, host: RunnerHost, job: JobRecord) -> R:
        """Plan a claimed job. Raises ``NarrationError`` for a job-level problem."""
        ...

    def advance(self, host: RunnerHost, run: R) -> Outcome:
        """Do one piece of the job's work or one bounded wait; ``finished`` once its record is written."""
        ...

    def save(self, host: RunnerHost, run: R) -> bool:
        """Write the job's state by compare-and-set; False when it is no longer ``running``."""
        ...

    def cancel(self, host: RunnerHost, run: R) -> JobRecord | None:
        """Finish a job being cancelled as ``cancelled``, keeping what it made."""
        ...

    def fail(self, host: RunnerHost, job: JobRecord, error: NarrationError, run: R | None) -> JobRecord | None:
        """Record a job-level failure (a job being cancelled is cancelled instead)."""
        ...

    def release(self, run: R) -> None:
        """Remove the job's scratch files."""
        ...

    def remaining_audio_s(self, run: R) -> float:
        """Audio seconds of work left, for the queue's drain estimate."""
        ...


@dataclass(frozen=True, slots=True)
class Registry:
    """The handler of each job kind, and what the handlers share (see the module docstring)."""

    handlers: Mapping[str, JobHandler[Any]]
    residency: Residency
    throughput: Throughput

    def handler(self, kind: str) -> JobHandler[Any] | None:
        """The handler for ``kind``, or None when this build does not run it."""
        return self.handlers.get(kind)

    def close(self) -> None:
        """Close each handler that has a ``close`` (the job engine stops the thread that renews its leases),
        once each. The runner calls it at shutdown, after the last step."""
        seen: set[int] = set()
        for handler in self.handlers.values():
            close = getattr(handler, "close", None)
            if id(handler) not in seen and callable(close):
                seen.add(id(handler))
                close()

    @classmethod
    def of(cls, engine: JobEngine) -> Registry:
        """The registry of this build: the job engine runs ``generate`` and ``analyse``."""
        return cls(handlers=dict.fromkeys(KINDS, engine), residency=engine.residency, throughput=engine.throughput)


__all__ = ["JobHandler", "Registry"]
