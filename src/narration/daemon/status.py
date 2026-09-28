"""``run/daemon.json`` (design sections 4.1, 7.6 and 15; DC-2's ``admission``): what the daemon says of itself.

``StatusBoard`` keeps the facts ``get_server_status`` reports (the state, the pid, the start time, the
workers, the current job, the GPU, the queue's drain estimate) and writes them through
``Store.put_daemon_status`` whenever one changes: on start, on every state or job-phase change, when a
worker starts or stops, when models load or unload, and on exit.

The last write says ``stopped``, with no pid, workers or job, but with the daemon's own start time and why it
stopped (``stop_reason``; contracts 1.6.11): a reader tells from them whether a daemon started after a launch, and
whether an operator's stop ended it (design section 4.1).

``gpu.unload_in_s`` is the time left, as of ``updated_at``, until the idle model is unloaded: a reader
subtracts the time since ``updated_at``. It is null while a job runs or when no model is on the GPU.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from narration.contracts.interfaces import Store
from narration.contracts.models import CurrentJob, DaemonStatus, GpuStatus, JobRecord, WorkerInfo
from narration.contracts.names import DaemonState, GpuHolder, JobPhase, StopReason
from narration.store.store import utc_iso

from .seam import GpuFacts

log = logging.getLogger(__name__)


class StatusBoard:
    """The daemon's status, written to ``run/daemon.json`` on every change (see the module docstring).

    Safe to use from several threads; writes are serialised, and a write that fails is logged and never
    stops the daemon. ``pid`` is the daemon's own ``os.getpid()``: under a venv, the pid its starter got back
    is the launcher's (section 4.1).
    """

    def __init__(self, store: Store, *, pid: int, started_at: str, wall: Callable[[], float] = time.time) -> None:
        self._store = store
        self._pid = pid
        self._started_at = started_at
        self._wall = wall
        self._lock = threading.Lock()
        self._state: DaemonState = "idle"
        self._workers: tuple[WorkerInfo, ...] = ()
        self._current: CurrentJob | None = None
        self._in_use = False
        self._holder: GpuHolder | None = None
        self._unload_at: float | None = None
        self._facts = GpuFacts()
        self._est_drain_s: float | None = None
        self._stop_reason: StopReason | None = None
        self._written: tuple[Any, ...] | None = None
        self.writes = 0
        """How many times the file was written (for tests and the log)."""

    # ------------------------------------------------------------------ what changes
    @property
    def state(self) -> DaemonState:
        with self._lock:
            return self._state

    @property
    def current_job(self) -> CurrentJob | None:
        with self._lock:
            return self._current

    def set_state(self, state: DaemonState) -> None:
        with self._lock:
            self._state = state
            self._publish()

    def set_workers(self, workers: tuple[WorkerInfo, ...], holder: GpuHolder | None) -> None:
        """The running workers, and the group whose models are on the GPU (``in_use`` while there is one)."""
        with self._lock:
            self._workers = workers
            self._holder = holder
            self._in_use = holder is not None
            self._publish()

    def job_started(self, job: JobRecord) -> None:
        with self._lock:
            self._current = CurrentJob(
                job_id=job.job_id, kind=job.kind, label=job.label, phase=job.phase, started_at=self._now()
            )
            if self._state == "idle":
                self._state = "busy"
            self._unload_at = None
            self._publish()

    def job_phase(self, phase: JobPhase | None) -> None:
        with self._lock:
            if self._current is None:
                return
            self._current = dataclasses.replace(self._current, phase=phase)
            self._publish()

    def job_finished(self) -> None:
        with self._lock:
            self._current = None
            if self._state == "busy":
                self._state = "idle"
            self._publish()

    def set_unload_at(self, wall_time: float | None) -> None:
        """When (Unix seconds) the idle models will be unloaded, or None."""
        with self._lock:
            self._unload_at = wall_time
            self._publish()

    def set_gpu_facts(self, facts: GpuFacts) -> None:
        with self._lock:
            self._facts = facts
            self._publish()

    def set_est_drain(self, seconds: float | None) -> None:
        with self._lock:
            self._est_drain_s = seconds
            self._publish()

    def stopped(self, reason: StopReason) -> None:
        """The last write: the daemon is ``stopped`` for ``reason``, with no pid, workers or job. The start time is
        kept (contracts 1.6.11)."""
        with self._lock:
            self._state = "stopped"
            self._stop_reason = reason
            self._workers = ()
            self._current = None
            self._in_use = False
            self._holder = None
            self._unload_at = None
            self._publish()

    def flush(self) -> None:
        """Write the record again if the last write failed (the daemon's control loop calls this each
        turn, so a status that could not be written is not left stale until the next change)."""
        with self._lock:
            self._publish()

    # ------------------------------------------------------------------ the record
    def snapshot(self) -> DaemonStatus:
        """The status as it would be written now."""
        with self._lock:
            return self._record()

    def _now(self) -> str:
        return utc_iso(self._wall())

    def _record(self) -> DaemonStatus:
        stopped = self._state == "stopped"
        unload_in_s: float | None = None
        if self._unload_at is not None and self._in_use and self._current is None:
            unload_in_s = round(max(0.0, self._unload_at - self._wall()), 3)
        facts = self._facts
        return DaemonStatus(
            state=self._state,
            pid=None if stopped else self._pid,
            started_at=self._started_at,
            workers=self._workers,
            current_job=self._current,
            gpu=GpuStatus(
                name=facts.name,
                total_mb=facts.total_mb,
                free_mb=facts.free_mb,
                in_use=self._in_use,
                holder=self._holder,
                unload_in_s=unload_in_s,
                need_mb=dict(facts.need_mb),
                waiting_since=facts.waiting_since,
            ),
            est_drain_s=self._est_drain_s,
            updated_at=self._now(),
            stop_reason=self._stop_reason if stopped else None,
        )

    def _publish(self) -> None:
        """Write the record if a fact changed since the last write (the caller holds the lock)."""
        key = (
            self._state,
            self._workers,
            self._current,
            self._in_use,
            self._holder,
            self._unload_at,
            self._facts,
            self._est_drain_s,
            self._stop_reason,
        )
        if key == self._written:
            return
        try:
            self._store.put_daemon_status(self._record())
        except Exception:
            log.exception("could not write run/daemon.json; will try again")
            return
        self._written = key
        self.writes += 1
