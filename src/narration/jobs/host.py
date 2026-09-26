"""The daemon seam the job engine runs behind (plan.md WP30 and WP31), mirrored here until WP30 merges.

WP30's ``narration.daemon.seam`` defines the contract between the daemon and the job engine: the daemon owns
the process (the singleton, the workers, the commands, the idle timers, ``run/daemon.json``) and drives a
``JobRunner`` one ``step`` at a time on a thread of its own; everything the runner needs comes through the
``RunnerHost`` it is handed. This module repeats those definitions member for member, so the engine can be
built and tested before WP30 is on ``main``. Once it is, this module re-exports ``narration.daemon.seam``
instead, and nothing else in ``narration.jobs`` changes.

The contract, as the engine keeps it:

- ``step(host)`` does at most one segment's worth of work (here: one render, one post-processing or one
  scoring, or one bounded wait) and says whether it did any. It claims a job itself when it holds none.
- While it holds a job it says so: ``job_started``, ``job_phase`` on every phase change, ``job_finished``.
- Every wait inside a step goes through ``host.sleep``, which returns False as soon as a stop is asked for.
- ``stop_mode`` ``"now"`` means the daemon killed every worker: the request in flight fails with
  ``WorkerCrashed``, and the step must not retry it.
- ``has_work(host)`` says whether a ``step`` would find work, without claiming anything.
- ``shutdown(host, reason)`` gives back the job held (``return_job``), removes the scratch files of the
  work it abandoned, and releases its leases.
- Between steps in which no job is held, the daemon may stop every worker (``release_gpu``, the idle
  unload). Whatever a worker holds (models, a prepared voice) is keyed to that worker process and
  established again in a new one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol, runtime_checkable

from narration.config import Config
from narration.contracts.interfaces import Store, WorkerClient
from narration.contracts.models import JobRecord
from narration.contracts.names import GpuHolder, JobPhase, WorkerRole

log = logging.getLogger(__name__)

StopMode = Literal["segment", "now"]
"""How the daemon stops: ``segment`` (finish the step in flight, then stop) or ``now`` (workers killed)."""

ShutdownReason = Literal["segment", "now", "idle"]
"""Why the runner is shut down: one of the ``StopMode``s, or ``idle`` (the idle exit, nothing in flight)."""

DAEMON_HOLDER: Final = "narrationd"
"""The daemon's holder name for jobs and leases. One daemon runs per store."""

GROUP_ROLES: Final[dict[GpuHolder, WorkerRole]] = {"qwen": "qwen3", "qa": "qa"}
"""The worker role that serves each model group (section 4: Qwen, or Whisper + WavLM)."""


class ResidencyError(RuntimeError):
    """A GPU load for one group while another group's models are on the GPU (``[gpu] one_group_at_a_time``,
    section 4): the caller must unload the resident group first."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuFacts:
    """What the job engine knows of the GPU (section 7.6; DC-2's ``admission.gpu``): NVML's readings, the
    VRAM each group needs, and since when a job has waited for free VRAM (ISO time, or None)."""

    name: str | None = None
    total_mb: int | None = None
    free_mb: int | None = None
    need_mb: dict[str, int] = field(default_factory=dict)
    waiting_since: str | None = None


@runtime_checkable
class WorkerPool(Protocol):
    """The daemon's workers, one per model group (section 4), started on demand.

    Every call may raise ``WorkerFailure`` (``BACKEND_NOT_INSTALLED`` for a worker that cannot start),
    ``WorkerCrashed`` and ``WorkerTimeout``.
    """

    @property
    def gpu_holder(self) -> GpuHolder | None:
        """The group whose models are on the GPU, or None."""
        ...

    def loaded(self) -> frozenset[GpuHolder]:
        """The groups with models loaded, on the GPU or on the CPU."""
        ...

    def client(self, group: GpuHolder, *, cublas_workspace_config: str | None = None) -> WorkerClient:
        """The group's worker, started (or started again after a crash) if need be."""
        ...

    def load(
        self,
        group: GpuHolder,
        payload: Mapping[str, Any],
        *,
        gpu: bool = True,
        timeout_s: float,
        cublas_workspace_config: str | None = None,
    ) -> dict[str, Any]:
        """Send ``load`` to the group's worker; returns the reply. A GPU load while another group is on the GPU
        is refused (``[gpu] one_group_at_a_time``): unloading first is the caller's choice."""
        ...

    def unload(self, group: GpuHolder, *, timeout_s: float) -> None:
        """Send ``unload`` to the group's worker if it runs; its models are no longer loaded."""
        ...

    def stop(self, group: GpuHolder) -> None:
        """Unload and close the group's worker process."""
        ...


@runtime_checkable
class RunnerHost(Protocol):
    """What the daemon gives the ``JobRunner`` (see the module docstring)."""

    @property
    def store(self) -> Store:
        """The daemon's store."""
        ...

    @property
    def config(self) -> Config:
        """The daemon's config."""
        ...

    @property
    def workers(self) -> WorkerPool:
        """The daemon's workers, one per model group."""
        ...

    @property
    def holder(self) -> str:
        """The name to claim jobs and leases with (``DAEMON_HOLDER``)."""
        ...

    @property
    def stop_mode(self) -> StopMode | None:
        """None while running; ``segment`` or ``now`` once a stop was asked for."""
        ...

    def should_stop(self) -> bool:
        """Whether any stop was asked for."""
        ...

    def sleep(self, seconds: float) -> bool:
        """Wait up to ``seconds``; False as soon as a stop is asked for, True if the time passed."""
        ...

    def job_started(self, job: JobRecord) -> None:
        """The runner now holds ``job``: the daemon is ``busy`` with it (``current_job``)."""
        ...

    def job_phase(self, phase: JobPhase | None) -> None:
        """The held job's phase changed (``current_job.phase`` in ``run/daemon.json``)."""
        ...

    def job_finished(self) -> None:
        """The runner holds no job any more: it is terminal, or given back to the queue."""
        ...

    def set_gpu_facts(self, facts: GpuFacts) -> None:
        """Replace what ``run/daemon.json`` says of the GPU besides ``in_use``, ``holder``, ``unload_in_s``."""
        ...

    def set_est_drain(self, seconds: float | None) -> None:
        """The queue's drain estimate (``est_drain_s``), or None when unknown."""
        ...


@runtime_checkable
class JobRunner(Protocol):
    """The job engine the daemon drives."""

    def step(self, host: RunnerHost) -> bool:
        """Do at most one segment's worth of work; True if some was done, False if there was none."""
        ...

    def has_work(self, host: RunnerHost) -> bool:
        """Whether a ``step`` now would find work: the same test, with no side effect (nothing is claimed).
        The daemon asks it at the idle exit, after publishing ``stopping``, so a job queued at the last moment
        is not stranded."""
        ...

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        """After the last step: give back the job held, clean up its scratch files, release leases."""
        ...


def return_job(store: Store, job_id: str, *, reason: str) -> JobRecord | None:
    """Give back a job the daemon held, by compare-and-set so a change made meanwhile is never undone.

    A ``running`` job goes back to ``queued`` (its phase cleared; its items, round and progress kept, since
    what it finished is in the cache); a ``cancelling`` job is finished as ``cancelled``. Any other status is
    left alone. Returns the job as changed, or None if nothing changed.
    """
    job = store.get_job(job_id)
    if job is None:
        return None
    if job.status == "running":
        changed = store.update_job(
            job_id, expect_status="running", status="queued", phase=None, message=f"queued again: {reason}"
        )
    elif job.status == "cancelling":
        changed = store.update_job(
            job_id, expect_status="cancelling", status="cancelled", phase=None, message=f"cancelled: {reason}"
        )
    else:
        return None
    if changed is not None:
        log.info("job %s is now %s (%s)", job_id, changed.status, reason)
    return changed


__all__ = [
    "DAEMON_HOLDER",
    "GROUP_ROLES",
    "GpuFacts",
    "JobRunner",
    "ResidencyError",
    "RunnerHost",
    "ShutdownReason",
    "StopMode",
    "WorkerPool",
    "return_job",
]
