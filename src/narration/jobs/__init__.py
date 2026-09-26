"""The job engine (design sections 4, 7.3, 7.4, 8, 10 and 11; plan.md WP31).

The daemon (WP30) drives a ``JobRunner`` one step at a time; ``EngineRunner`` (``runner``) is that runner.
It claims jobs and runs each through ``JobEngine`` (``engine``): per round, render on Qwen, post-process on
the CPU, score on the QA group, then retakes of the take slots that failed QA, up to ``max_retakes``. Every
layer is looked up in the cache first and produced at most once, under a lease (section 4 item 6).

- ``host``: the daemon's seam, mirrored until WP30 merges.
- ``gpu``: one resident model group, the free-VRAM check and the wait (section 4).
- ``admission``: DC-2's numbers: ``poll_after_s``, ``retry_after_s``, ``est_drain_s``, ``admission``.
- ``plan``, ``pins``, ``voice``, ``hooks``: what the engine reads from a request, the pinned models, the clip
  a worker clones from, and where WP32's engine guard plugs in.
"""

from __future__ import annotations

from .engine import EngineParts, JobEngine, JobRun
from .gpu import NoProbe, NvmlProbe, VramProbe, VramReading
from .hooks import EngineGuard, NoGuard
from .pins import ModelPin, QaPins
from .runner import EngineRunner, build_runner

__all__ = [
    "EngineGuard",
    "EngineParts",
    "EngineRunner",
    "JobEngine",
    "JobRun",
    "ModelPin",
    "NoGuard",
    "NoProbe",
    "NvmlProbe",
    "QaPins",
    "VramProbe",
    "VramReading",
    "build_runner",
]
