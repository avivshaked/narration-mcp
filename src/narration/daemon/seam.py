"""The seam between the daemon and the job engine (plan.md WP30 and WP31).

The daemon (``service.Daemon``) owns the process: the singleton, the workers, the commands, the idle timers
and ``run/daemon.json``. The job engine (WP31: the claim loop, the GPU scheduler, the per-round pipeline)
is a ``JobRunner`` that the daemon drives, one ``step`` at a time, on a thread of its own. Everything the
runner needs from the daemon comes through the ``RunnerHost`` it is handed.

**The contract** (what WP31 implements):

- ``step(host)`` does at most one segment's worth of work and returns: True if it did some, False if it
  found nothing to do. It claims a job itself when it holds none (``Store.claim_next_job`` or
  ``claim_job`` with ``host.holder``), and it may keep holding the job across steps. A segment boundary is
  the return from ``step``: the daemon never interrupts a step, except by ``stop_now``.
- While it holds a job it says so: ``host.job_started(job)`` when it takes the job, ``host.job_phase(phase)``
  on every phase change (``waiting_for_gpu``, ``loading_model``, ``rendering``, …), and
  ``host.job_finished()`` when the job is terminal or given back. The daemon is ``busy`` in between, and
  ``release_gpu`` then changes nothing and names the job.
- Every wait inside a step (for free VRAM, for another job's lease) goes through ``host.sleep``, which
  returns False as soon as a stop is asked for; the step should then return.
- ``host.stop_mode`` says how the daemon is stopping: ``"segment"`` (``stop``: the daemon calls ``step``
  no more once the one in flight returns) or ``"now"`` (``stop_now``: the daemon has killed every worker,
  so the request in flight fails with ``WorkerCrashed``; the step must not retry it, and should return).
- ``has_work(host)`` says whether a job is waiting that ``step`` would take (or the runner still holds one).
  It must be cheap and have no side effects: it claims nothing and changes nothing. The daemon calls it on
  the runner's thread, between steps, only when it is about to exit for want of work (see "The idle exit").
- ``shutdown(host, reason)`` runs on the runner's thread after its last step, whatever the reason: it
  gives back the job it holds (``return_job``: ``running`` goes back to ``queued``, and a ``cancelling``
  job is finished as ``cancelled``), removes the scratch files of the segment it abandoned, and releases
  its leases. After it, the daemon re-queues any job still announced by ``job_started`` as a safety net.
- A job-level failure (``BACKEND_NOT_INSTALLED``, ``GPU_UNAVAILABLE``, …) is recorded on the job by the
  runner. An exception that escapes ``step`` is logged as a bug, and the daemon waits before the next step.
- The workers are the daemon's (``host.workers``, a ``WorkerPool``): one per model group, ``qwen`` or
  ``qa``. Which group is resident, and when to swap, is the runner's choice; the pool only refuses a second
  GPU group while one is loaded (``[gpu] one_group_at_a_time``, ``ResidencyError``, which this module
  exports), and the daemon unloads idle models by itself after ``idle_unload_s``.
- **Worker state belongs to the worker instance.** While the runner holds no job (between ``job_finished``
  and the next ``job_started``), the daemon may stop every worker between two steps: ``release_gpu``, and
  the idle unload. A crash does the same at any time. So anything the runner keeps about what is inside a
  worker (a loaded model, a prepared voice) must be keyed to that worker instance: ``workers.client(group)``
  returns the same client object for as long as that worker process lives, and a new object for a new
  process. Compare the object itself (keep a reference to it); a pid may be reused.
- **A worker that cannot start** (``BACKEND_NOT_INSTALLED``: its venv is missing or broken) is never started
  again for the rest of the daemon's life; every call for its group raises the same error, and the runner
  fails the job with it. ``narration-admin install`` (WP37) stops the daemon after it repairs a worker, and
  the next use starts a fresh daemon.
- ``host.platform`` is the ``narration.platform`` ``Platform`` the daemon uses (in tests, usually
  ``narration.platform.testing.StandInPlatform``). The runner checks paths with it, such as a caller's clip
  (``check_readable_path``, section 17.3) and the files it writes (``check_store_path``, section 17.2).
- ``host.set_gpu_facts`` and ``host.set_est_drain`` feed ``run/daemon.json``: the NVML readings and the
  VRAM wait (DC-2's ``admission.gpu``), and the queue's drain estimate (``admission.queue.est_drain_s``).

Leases taken with ``host.holder`` (``DAEMON_HOLDER``) outlive a daemon that dies only until their TTL, and
the next daemon, holding the same name, may claim those keys again at once: the singleton makes the name
unique to the store's one daemon.

**The idle exit** (why ``has_work`` exists). A front-end that queues a job starts a daemon unless one says
it serves (``idle`` or ``busy``). A daemon that exits for want of work would strand a job queued between its
last empty ``step`` and the moment it says ``stopping``. So it says ``stopping`` first (``run/daemon.json``),
then asks ``has_work`` once more:

- work found: it says ``idle`` again and keeps serving; the job is taken by the next ``step``;
- none: it exits. A job queued after that look finds the daemon ``stopping``, so the front-end starts
  another daemon. That one waits (up to ``takeover_wait_s``) while the holder says anything but ``idle``
  or ``busy`` (``stopping``, then ``stopped`` until it releases the singleton), and takes over once it has
  gone. If the holder says ``idle`` or ``busy`` again (it found work after all), the new daemon gives up
  at once, exits 0 and writes nothing, as any second daemon does.

**The front-end's order** (WP36). This holds only if a front-end commits the job to the store *before* it
reads the daemon's status (``start.ensure_daemon``). The front-end commits, then reads the status; the
daemon writes ``stopping``, then asks ``has_work``. So either the daemon's look comes after the commit and
sees the job, or the front-end's read comes after ``stopping`` and starts another daemon. A front-end that
reads the status first, then commits, can see ``idle`` and still lose its job to the exit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol, runtime_checkable

from narration.config import Config
from narration.contracts.interfaces import Platform, Store, WorkerClient
from narration.contracts.models import JobRecord
from narration.contracts.names import GpuHolder, JobPhase, WorkerRole

__all__ = [
    "DAEMON_HOLDER",
    "GROUP_ROLES",
    "GpuFacts",
    "JobRunner",
    "NullRunner",
    "ResidencyError",
    "RunnerHost",
    "ShutdownReason",
    "StopMode",
    "WorkerPool",
    "return_job",
]

log = logging.getLogger(__name__)

StopMode = Literal["segment", "now"]
"""How the daemon stops: ``segment`` for ``stop`` (finish the segment in flight, then stop), ``now`` for
``stop_now`` (the workers are killed at once; the segment in flight is abandoned and re-queued)."""

ShutdownReason = Literal["segment", "now", "idle"]
"""Why the runner is shut down: one of the ``StopMode``s, or ``idle`` (the idle exit, nothing in flight)."""

DAEMON_HOLDER: Final = "narrationd"
"""The daemon's holder name for jobs (``claimed_by``) and leases. One daemon runs per store."""

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
    """The daemon's workers, one per model group (section 4), started on demand (``supervisor``).

    Each group's worker starts from its venv's Python with the offline environment and the thread cap, at
    below-normal priority, inside the daemon's kill-on-close group. A worker that crashed is started again
    on the next call; one that cannot start (its venv is missing or broken) raises
    ``WorkerFailure("BACKEND_NOT_INSTALLED")``, and keeps raising it without another attempt for the rest of
    the daemon's life. Every call may also raise ``WorkerCrashed`` (the process died, or restarts are
    exhausted) and ``WorkerTimeout``.
    """

    @property
    def gpu_holder(self) -> GpuHolder | None:
        """The group whose models are on the GPU, or None."""
        ...

    def loaded(self) -> frozenset[GpuHolder]:
        """The groups with models loaded, on the GPU or on the CPU."""
        ...

    def client(self, group: GpuHolder, *, cublas_workspace_config: str | None = None) -> WorkerClient:
        """The group's worker, started (or started again after a crash) if need be.

        ``cublas_workspace_config`` is the engine profile's pin (section 10.1). It is fixed when a worker
        starts, so a running worker started with another value is stopped and started again.
        """
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
        """Send ``load`` to the group's worker; returns the reply.

        ``gpu`` says whether the load puts models on the GPU. With ``[gpu] one_group_at_a_time``, a GPU load
        while another group is on the GPU raises ``ResidencyError``: unloading first is the caller's choice.
        """
        ...

    def unload(self, group: GpuHolder, *, timeout_s: float) -> None:
        """Send ``unload`` to the group's worker if it runs; its models are no longer loaded."""
        ...

    def stop(self, group: GpuHolder) -> None:
        """Unload and close the group's worker process (which also frees its CUDA context)."""
        ...


@runtime_checkable
class RunnerHost(Protocol):
    """What the daemon gives the ``JobRunner`` (see the module docstring for the contract)."""

    @property
    def store(self) -> Store: ...

    @property
    def config(self) -> Config: ...

    @property
    def workers(self) -> WorkerPool: ...

    @property
    def platform(self) -> Platform:
        """The daemon's ``narration.platform`` ``Platform``: path checks (sections 17.2 and 17.3) and the rest."""
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
        """Whether any stop was asked for (``stop``, ``stop_now``, or the daemon's own exit)."""
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
    """The job engine the daemon drives (WP31; see the module docstring for the contract)."""

    def step(self, host: RunnerHost) -> bool:
        """Do at most one segment's worth of work; True if some was done, False if there was none."""
        ...

    def has_work(self, host: RunnerHost) -> bool:
        """Whether ``step`` would find work now: a job it would take, or the one it holds. Cheap, and with no
        side effects (see "The idle exit" in the module docstring)."""
        ...

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        """After the last step: give back the job held, clean up its scratch files, release leases."""
        ...


class NullRunner:
    """A ``JobRunner`` that never finds work: the daemon's default until the job engine (WP31) replaces it.

    With it, the daemon starts, answers commands, writes its status, and exits after ``idle_exit_s``.
    """

    def step(self, host: RunnerHost) -> bool:
        return False

    def has_work(self, host: RunnerHost) -> bool:
        return False

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        return None


def return_job(store: Store, job_id: str, *, reason: str) -> JobRecord | None:
    """Give back a job the daemon held, by compare-and-set so a change made meanwhile is never undone.

    A ``running`` job goes back to ``queued`` (its phase cleared; its items, round and progress kept, since
    what it finished is in the cache); a ``cancelling`` job is finished as ``cancelled``, completing the
    cancel it was asked for (section 8). Any other status is left alone. Returns the job as changed, or
    None if nothing changed.
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
