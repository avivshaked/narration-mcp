"""The daemon seam the job engine runs behind: ``narration.daemon.seam`` (plan.md WP30 and WP31), re-exported.

The daemon owns the process (the singleton, the workers, the commands, the idle timers,
``run/daemon.json``) and drives a ``JobRunner`` one ``step`` at a time on a thread of its own; everything
the runner needs comes through the ``RunnerHost`` it is handed. ``narration.daemon.seam`` states the
contract in full. The engine's modules import it from here, so the engine names its dependency on the
daemon in one place.

How the engine keeps the contract:

- ``step(host)`` does at most one segment's worth of work (here: one render, one post-processing or one
  scoring, or one bounded wait) and says whether it did any. It claims a job itself when it holds none.
- While it holds a job it says so: ``job_started``, ``job_phase`` on every phase change, ``job_finished``.
- Every wait inside a step goes through ``host.sleep``, which returns False as soon as a stop is asked for.
- ``stop_mode`` ``"now"`` means the daemon killed every worker: the request in flight fails with
  ``WorkerCrashed``, and the step does not retry it.
- ``has_work(host)`` says whether a ``step`` would find work, without claiming anything.
- ``shutdown(host, reason)`` gives back the job held (``return_job``), removes the scratch files of the
  work it abandoned, and stops the thread that renews its leases (each lease is released when its work
  ends, so none is held between steps).
- Between steps in which no job is held, the daemon may stop every worker (``release_gpu``, the idle
  unload). Whatever a worker holds (models, a prepared voice) is keyed to that worker instance, the client
  object ``workers.client(group)`` returns for as long as its process lives (compared by identity: a pid
  may be reused), and established again in a new one.
- A caller's clip is read only through ``host.platform.check_readable_path`` (section 17.3), unless the
  engine was built with another check (``EngineParts.check_path``).
"""

from __future__ import annotations

from narration.daemon.seam import (
    DAEMON_HOLDER,
    GROUP_ROLES,
    GpuFacts,
    JobRunner,
    ResidencyError,
    RunnerHost,
    ShutdownReason,
    StopMode,
    WorkerPool,
    return_job,
)

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
