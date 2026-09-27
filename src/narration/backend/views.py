"""What ``get_job`` and ``get_server_status`` report (design sections 4.1, 7.4, 7.6; DC-2).

Both are read from the store: the job rows, the queue, and ``run/daemon.json``, which the daemon writes on
every phase change. DC-2's numbers come from ``narration.jobs.admission``, the same pure functions the
daemon's engine uses, so the front-end and the daemon agree:

- ``poll_after_s``: the earliest ``get_job`` poll worth making; for a queued job a quarter of its wait to
  start, for a running one a tenth of its time left, ``GPU_RECHECK_S`` while it waits for the GPU, 0 once
  it is finished;
- ``eta_s``: a running job's time left; a queued job's wait for the jobs ahead plus its own work;
- ``queue_position``: 0 is next to run;
- ``admission``: whether a submission would be accepted now, and when to come back.

The service suggests; the consumer decides.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from narration.contracts.models import DaemonStatus, JobRecord
from narration.contracts.names import TERMINAL_JOB_STATUSES
from narration.contracts.serial import to_json
from narration.jobs import admission


def ahead_of(job_id: str, queued: Sequence[JobRecord]) -> list[JobRecord]:
    """The active jobs that run before ``job_id``: every running job, and the queued ones ahead of it."""
    order = admission.scheduling_order(queued)
    ahead: list[JobRecord] = []
    for job in order:
        if job.job_id == job_id:
            break
        ahead.append(job)
    return ahead


def wait_to_start_s(job_id: str, queued: Sequence[JobRecord]) -> float:
    """The drain estimate of the jobs ahead of ``job_id`` (DC-2), in wall seconds."""
    ahead = ahead_of(job_id, queued)
    return admission.est_drain_s(ahead) if ahead else 0.0


def job_timing(job: JobRecord, queued: Sequence[JobRecord]) -> tuple[float | None, int | None, float]:
    """``eta_s``, ``queue_position`` and ``poll_after_s`` of a job (section 7.4, DC-2)."""
    if job.status in TERMINAL_JOB_STATUSES:
        return None, None, admission.poll_after_s(job.status, job.phase)
    if job.status == "queued":
        wait = wait_to_start_s(job.job_id, queued)
        own = admission.job_wall_s(
            admission.request_audio_s(job.kind, job.request), wall_per_audio_s=admission.WALL_PER_AUDIO_S
        )
        position = admission.queue_position(job.job_id, queued)
        return round(wait + own, 1), position, admission.poll_after_s("queued", wait_to_start_s=wait)
    eta = admission.eta_s(job.progress) if job.progress.total_s > 0 else None
    return eta, None, admission.poll_after_s(job.status, job.phase, eta=eta)


def job_json(job: JobRecord, queued: Sequence[JobRecord], *, include_segments: bool) -> dict[str, Any]:
    """``get_job``'s result (section 7.4). A failed job is a successful call carrying the job's error."""
    eta, position, poll = job_timing(job, queued)
    out: dict[str, Any] = {
        "job_id": job.job_id,
        "kind": job.kind,
        "label": job.label,
        "status": job.status,
        "phase": job.phase,
        "round": job.round,
        "outcome": job.outcome,
        "progress": to_json(job.progress),
        "eta_s": eta,
        "queue_position": position,
        "poll_after_s": poll,
        "message": job.message,
        "updated_at": job.updated_at,
    }
    if include_segments:
        out["segments"] = [
            {
                "segment_id": item.segment_id,
                "state": item.state,
                "takes_ok": item.takes_ok,
                "retakes_used": item.retakes_used,
            }
            for item in job.items
        ]
    if job.error is not None:
        out["error"] = to_json(job.error)
    return out


def daemon_json(status: DaemonStatus | None) -> dict[str, Any]:
    """``get_server_status``'s ``daemon`` (section 4.1): ``stopped`` when none runs."""
    if status is None:
        return {"state": "stopped", "pid": None, "current_job": None, "workers": []}
    return {
        "state": status.state,
        "pid": status.pid,
        "current_job": to_json(status.current_job) if status.current_job is not None else None,
        "workers": [to_json(w) for w in status.workers],
    }


def gpu_json(status: DaemonStatus | None) -> dict[str, Any]:
    """``get_server_status``'s ``gpu``, as the daemon last saw it; unknown (null) while no daemon runs, since
    the front-end never touches the GPU (section 4)."""
    if status is None:
        return {"name": None, "total_mb": None, "free_mb": None, "in_use": False, "holder": None, "unload_in_s": None}
    gpu = status.gpu
    return {
        "name": gpu.name,
        "total_mb": gpu.total_mb,
        "free_mb": gpu.free_mb,
        "in_use": gpu.in_use,
        "holder": gpu.holder,
        "unload_in_s": gpu.unload_in_s,
    }


def queue_json(queued: Sequence[JobRecord]) -> dict[str, Any]:
    """``get_server_status``'s ``queue``: the jobs waiting (``length``) and every active job, in run order."""
    order = admission.scheduling_order(queued)
    jobs = [
        {
            "job_id": job.job_id,
            "kind": job.kind,
            "label": job.label,
            "status": job.status,
            "priority": job.priority,
            "queue_position": admission.queue_position(job.job_id, queued),
        }
        for job in order
    ]
    return {"length": sum(1 for j in queued if j.status == "queued"), "jobs": jobs}


def rate(count_since: Callable[[float], int], *, now: float, max_per_min: int) -> admission.RateWindow:
    """The submit rate cap's window (DC-2's ``admission.rate``)."""
    return admission.rate_window(count_since, now=now, max_per_window=max_per_min)


def drain_s(queued: Sequence[JobRecord], daemon: DaemonStatus | None) -> float | None:
    """The queue's drain estimate: the daemon's own (it knows the job it holds), else one from the requests."""
    if daemon is not None and daemon.est_drain_s is not None:
        return daemon.est_drain_s
    return admission.est_drain_s(queued) if queued else 0.0


__all__ = [
    "ahead_of",
    "daemon_json",
    "drain_s",
    "gpu_json",
    "job_json",
    "job_timing",
    "queue_json",
    "rate",
    "wait_to_start_s",
]
