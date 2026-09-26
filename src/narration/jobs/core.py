"""What the job engine's pieces share (plan.md WP31): the parts it is built from, and what it keeps between
jobs (the GPU residency, the throughput estimate, the canary outcome of the Qwen load in use).

The engine is split by concern: ``state`` (a job's state), ``stages`` (render, post-process, score),
``failures`` (a worker's failure turned into a retry, a flag or a job error), ``record`` (the job record and
the job's endings) and ``engine`` (planning and advancing). Each piece holds this core.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Protocol

from narration.config import Config
from narration.contracts.interfaces import AlignerCore, DeliveryProcessor, QaScorer, TextPlanner
from narration.contracts.names import CanaryStatus, JobPhase

from .admission import Throughput
from .gpu import NoProbe, Residency, VramProbe
from .hooks import EngineGuard, NoGuard
from .host import RunnerHost
from .pins import QaPins
from .state import JobRun
from .voice import PathCheck

DEFER_S: Final = 2.0
"""How long work that another holder is producing is left before it is looked at again."""


class Scorer(QaScorer, Protocol):
    """The QA scorer, with the version of its number reader, which the analysis key names."""

    @property
    def number_reader(self) -> str:
        """The number reader's version (``narration.qa``), part of the analysis key (section 10.2)."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineParts:
    """What the engine is built from: the service's text pipeline, post-processing, QA scorer and aligner; the
    QA group's pins; the engine guard (WP32's canary gate); the VRAM probe; the path check for a caller's
    clip; a monotonic clock; and how long to leave work another holder is producing (``defer_s``)."""

    text: TextPlanner
    delivery: DeliveryProcessor
    scorer: Scorer
    aligner: AlignerCore
    qa_pins: QaPins
    guard: EngineGuard = field(default_factory=NoGuard)
    probe: VramProbe = field(default_factory=NoProbe)
    check_path: PathCheck | None = None
    clock: Callable[[], float] = time.monotonic
    defer_s: float = DEFER_S


class EngineCore:
    """The config and parts, and the state one engine keeps across the jobs it runs: which model group is
    resident (``residency``), how fast work goes (``throughput``), and the canary outcome of the Qwen load in
    use (``canary``), which every render made on that load records."""

    def __init__(self, config: Config, parts: EngineParts) -> None:
        self.config = config
        self.parts = parts
        self.residency = Residency(gpu=config.gpu, probe=parts.probe, clock=parts.clock)
        self.residency.need_mb["qa"] = parts.qa_pins.vram_need_mb
        self.throughput = Throughput()
        self.canary: CanaryStatus = "not_run"

    def phase(self, host: RunnerHost, run: JobRun, phase: JobPhase) -> None:
        """Set the job's phase, and tell the daemon when it changes."""
        if run.phase != phase:
            run.phase = phase
            host.job_phase(phase)


__all__ = ["DEFER_S", "EngineCore", "EngineParts", "Scorer"]
