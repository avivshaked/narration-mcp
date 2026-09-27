"""The job engine as this installation provides it (plan.md WP31, WP32): what ``narration.jobs.runner``'s
``installed_engine`` builds at the daemon's first job.

It assembles the job engine from the daemon's configuration and the service's pins:

- the service's text pipeline, delivery post-processing and QA scorer, configured by ``[text]``,
  ``[delivery]`` and ``[measurement]``;
- the cue aligner (WP15's ``CtcAligner``) at its pinned revision, with ``[alignment]``'s thresholds;
- the QA group's pinned models (Whisper, WavLM-SV, the CTC aligner) and the VRAM it needs;
- the engine guard: the worker fingerprint check and the canary gate (``canary.CanaryGuard``);
- the free-VRAM probe (NVML on a ``cuda`` device).

A caller's clip is checked by the daemon's platform (``host.platform.check_readable_path``, section 17.3): the
parts leave ``check_path`` unset, which the job engine takes to mean that.

A snapshot that is not installed raises ``BACKEND_NOT_INSTALLED``; the runner fails that job with it and builds
again at the next, so an installation completed meanwhile is picked up. The engine profile itself is the
store's (``engine pin``), read by the job engine at each job.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from narration.jobs.core import EngineParts
from narration.jobs.engine import JobEngine
from narration.jobs.gpu import NoProbe, NvmlProbe, VramProbe
from narration.jobs.handlers import JobHandler, Registry
from narration.jobs.host import RunnerHost
from narration.post import DeliveryPipeline
from narration.qa import Scorer
from narration.text import TextPipeline

from .canary import CanaryGuard
from .qa import aligner, qa_pins


def probe_for(device: str) -> VramProbe:
    """NVML's free-VRAM reading for a ``cuda`` device (section 4 item 2); none for the CPU."""
    return NvmlProbe(device) if device.startswith("cuda") else NoProbe()


def engine(host: RunnerHost, *, probe: VramProbe | None = None) -> JobEngine:
    """The job engine for ``generate`` and ``analyse`` (see the module docstring)."""
    config = host.config
    pins = qa_pins(config)
    parts = EngineParts(
        text=TextPipeline(config.text),
        delivery=DeliveryPipeline(fade_s=config.delivery.fade_s),
        scorer=Scorer(config.measurement),
        aligner=aligner(config),
        qa_pins=pins,
        guard=CanaryGuard(sv=pins.sv),
        probe=probe if probe is not None else probe_for(config.gpu.device),
    )
    return JobEngine(config, parts)


def more_handlers(host: RunnerHost, engine: JobEngine) -> Mapping[str, JobHandler[Any]]:
    """The handlers of the other job kinds, by kind, built on the daemon's job ``engine`` so that they share
    its residency, throughput and canary.

    **WP33 registers ``measure`` here**: ``{"measure": narration.measure.build_measure_handler(engine)}``;
    WP34 and WP35 add their kinds after it. None is built yet, so a job of those kinds fails with
    ``INTERNAL`` and ``details.kind``.
    """
    return {}


def registry(host: RunnerHost, *, probe: VramProbe | None = None) -> Registry:
    """Every job kind this installation runs, by handler: ``generate`` and ``analyse`` on the job engine, and
    whatever ``more_handlers`` adds, all sharing the engine's residency and throughput."""
    built = engine(host, probe=probe)
    base = Registry.of(built)
    extra = more_handlers(host, built)
    if not extra:
        return base
    return Registry(handlers={**base.handlers, **extra}, residency=base.residency, throughput=base.throughput)


__all__ = ["engine", "more_handlers", "probe_for", "registry"]
