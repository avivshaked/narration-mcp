"""DC-2's numbers: when a consumer should come back (design sections 7.3, 7.4, 7.6 and 14; plan.md DC-2).

Retries are already safe (content-hash ids, ``idempotency_key``, an identical request is the same job). These
functions tell a consumer *when* to come back, so a shared GPU is not polled into the ground:

- ``poll_after_s``: the earliest ``get_job`` poll worth making (``submit_job`` and ``get_job``);
- ``queue_full_retry_after_s``, ``rate_window`` and ``GPU_UNAVAILABLE_RETRY_S``: ``retry_after_s`` of the
  retryable errors ``QUEUE_FULL``, ``RATE_LIMITED`` and ``GPU_UNAVAILABLE``;
- ``est_drain_s``, ``eta_s`` and ``queue_position``: the queue's drain estimate (``admission.queue`` and
  ``run/daemon.json``) and a job's place in it;
- ``admission``: the facts ``get_server_status`` reports, so a consumer can decide before it submits.

Every function here is pure: the caller passes what it read (the queue, the daemon's status, the clock), so
the front-end (WP36) and the daemon's engine compute the same numbers. **The service suggests; the consumer
decides.** The constants below are starting points marked ASSUME where nothing has measured them yet; the
engine's ``Throughput`` replaces the rate with what it measures on this machine.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from narration.contracts.models import DaemonStatus, JobRecord, Progress
from narration.contracts.names import TERMINAL_JOB_STATUSES, JobPhase, JobStatus

GPU_RECHECK_S: Final = 15.0
"""Section 4: while a job waits for free VRAM, the VRAM is checked again every 15 s."""
POLL_MIN_S: Final = 1.0
POLL_RUNNING_MAX_S: Final = 15.0
POLL_QUEUED_MIN_S: Final = 2.0
POLL_QUEUED_MAX_S: Final = 30.0
POLL_DEFAULT_RUNNING_S: Final = 5.0
POLL_DEFAULT_QUEUED_S: Final = 10.0
RETRY_MIN_S: Final = 1.0
QUEUE_FULL_DEFAULT_S: Final = 60.0
"""``QUEUE_FULL``'s ``retry_after_s`` when the drain estimate is unknown."""
QUEUE_FULL_MAX_S: Final = 900.0
GPU_UNAVAILABLE_RETRY_S: Final = 60.0
"""``GPU_UNAVAILABLE``'s ``retry_after_s``: the job already waited ``wait_timeout_min`` for free VRAM, and a
resubmission waits again, so a minute is enough to let the other work move on."""
RATE_WINDOW_S: Final = 60.0
"""The submit rate cap's window (``[limits] max_submits_per_min``)."""
WALL_PER_AUDIO_S: Final = 3.0
"""ASSUME: wall seconds per second of audio a job produces (render, post-processing and scoring together),
until ``Throughput`` has measured this machine. App. B's example render ran at 2.3 x real time."""
MODEL_LOAD_S: Final = 20.0
"""ASSUME: one model load (App. A's example: 18.2 s)."""
LOADS_PER_JOB: Final = 2
"""A generation job loads Qwen, then the QA group, at least once each (section 4 item 4)."""
CHARS_PER_AUDIO_S: Final = 15.0
"""ASSUME: spoken characters per second of audio when no pace curve is known (about 150 wpm)."""

_KIND_AUDIO_S: Final[dict[str, float]] = {
    "measure": 35.0 * 60.0 / WALL_PER_AUDIO_S,  # section 3.2: 20-50 min of GPU
    "profile": 1.0,  # CPU only, seconds
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ======================================================================== estimates


class Throughput:
    """Wall seconds per second of audio, as measured on this machine: an exponential moving average.

    The engine records each piece of work (its wall time and the audio seconds it moved forward); until it has
    recorded any, ``WALL_PER_AUDIO_S`` stands in.
    """

    def __init__(self, *, initial: float = WALL_PER_AUDIO_S, weight: float = 0.2) -> None:
        if initial <= 0 or not 0 < weight <= 1:
            raise ValueError("initial must be positive and weight in (0, 1]")
        self._rate = initial
        self._weight = weight
        self.samples = 0

    @property
    def wall_per_audio_s(self) -> float:
        """The current estimate: wall seconds per second of audio made."""
        return self._rate

    def record(self, wall_s: float, audio_s: float) -> None:
        """One piece of work: ``wall_s`` spent on ``audio_s`` seconds of audio. Nonsense is ignored."""
        if not (math.isfinite(wall_s) and math.isfinite(audio_s)) or wall_s < 0 or audio_s <= 0:
            return
        rate = wall_s / audio_s
        self._rate = rate if self.samples == 0 else self._rate + self._weight * (rate - self._rate)
        self.samples += 1


def request_audio_s(kind: str, request: Mapping[str, Any]) -> float:
    """The audio seconds a queued job will produce, from its request alone (no cache lookups).

    A generation or scoring job: every segment's spoken characters (its cues or its text) times the takes it
    asks for, at ``CHARS_PER_AUDIO_S``. Cached work drains faster, so this is an upper estimate. Other kinds
    use a nominal figure (ASSUME). Malformed parts count as nothing.
    """
    if kind in _KIND_AUDIO_S:
        return _KIND_AUDIO_S[kind]
    options = request.get("options")
    takes = options.get("takes", 1) if isinstance(options, Mapping) else 1
    takes = takes if isinstance(takes, int) and not isinstance(takes, bool) and takes > 0 else 1
    chars = 0.0
    if kind == "design":
        text = request.get("design_text")
        per = len(text) if isinstance(text, str) else 200
        count = request.get("takes", 1)
        return per * (count if isinstance(count, int) and count > 0 else 1) / CHARS_PER_AUDIO_S
    if kind == "pronunciation":
        variants = request.get("variants")
        return 3.0 * (len(variants) if isinstance(variants, list) else 1)
    segments = request.get("segments")
    for segment in segments if isinstance(segments, list) else ():
        if not isinstance(segment, Mapping):
            continue
        cues = segment.get("cues")
        text = segment.get("text")
        length = (
            sum(len(c.get("text", "")) + 1 for c in cues if isinstance(c, Mapping))
            if isinstance(cues, list)
            else len(text)
            if isinstance(text, str)
            else 0
        )
        attempts = segment.get("attempts")
        count = len(attempts) if isinstance(attempts, list) and attempts else takes
        chars += length * count
    return chars / CHARS_PER_AUDIO_S


def job_wall_s(audio_s: float, *, wall_per_audio_s: float, loads: int = LOADS_PER_JOB) -> float:
    """Wall seconds for ``audio_s`` of work plus ``loads`` model loads."""
    return max(0.0, audio_s) * wall_per_audio_s + max(0, loads) * MODEL_LOAD_S


def eta_s(progress: Progress, *, wall_per_audio_s: float = WALL_PER_AUDIO_S, loads_left: int = 1) -> float:
    """A running job's time to finish: the audio seconds still to do, at the rate, plus the loads left."""
    return round(job_wall_s(progress.total_s - progress.done_s, wall_per_audio_s=wall_per_audio_s, loads=loads_left), 1)


def scheduling_order(queued: Sequence[JobRecord]) -> list[JobRecord]:
    """The active jobs in the order the engine takes them: priority (``interactive`` first), then FIFO.

    ``Store.queued_jobs`` already lists them so; sorting again (stably) keeps that true for any caller.
    """
    rank = {"interactive": 0, "batch": 1}
    return sorted(queued, key=lambda j: rank[j.priority])


def queue_position(job_id: str, queued: Sequence[JobRecord]) -> int | None:
    """A queued job's place among the queued jobs (0 = next to run), or None when it is not queued."""
    waiting = [j for j in scheduling_order(queued) if j.status == "queued"]
    return next((i for i, j in enumerate(waiting) if j.job_id == job_id), None)


def est_drain_s(
    queued: Sequence[JobRecord],
    *,
    wall_per_audio_s: float = WALL_PER_AUDIO_S,
    running_remaining_s: Mapping[str, float] | None = None,
) -> float:
    """How long the queue will take to drain, in wall seconds: every active job's estimated work.

    ``running_remaining_s`` gives the audio seconds left for the jobs the engine holds (it knows them
    exactly); every other job is estimated from its request (``request_audio_s``) or, for a running job the
    caller knows nothing more of, from its progress.
    """
    held = running_remaining_s or {}
    total = 0.0
    for job in queued:
        if job.status in TERMINAL_JOB_STATUSES:
            continue
        if job.job_id in held:
            total += job_wall_s(held[job.job_id], wall_per_audio_s=wall_per_audio_s, loads=1)
        elif job.status in ("running", "cancelling") and job.progress.total_s > 0:
            left = job.progress.total_s - job.progress.done_s
            total += job_wall_s(left, wall_per_audio_s=wall_per_audio_s, loads=1)
        else:
            total += job_wall_s(request_audio_s(job.kind, job.request), wall_per_audio_s=wall_per_audio_s)
    return round(total, 1)


# ======================================================================== poll_after_s


def poll_after_s(
    status: JobStatus,
    phase: JobPhase | None = None,
    *,
    eta: float | None = None,
    wait_to_start_s: float | None = None,
) -> float:
    """The earliest ``get_job`` poll worth making (DC-2), in seconds.

    - A finished job: 0 (its results are ready).
    - ``waiting_for_gpu``: ``GPU_RECHECK_S``, since the VRAM is checked again no sooner.
    - Queued: a quarter of the wait to start, within ``POLL_QUEUED_MIN_S``..``POLL_QUEUED_MAX_S``.
    - Running or cancelling: a tenth of the time left, within ``POLL_MIN_S``..``POLL_RUNNING_MAX_S``.

    A long-poll (``get_job``'s ``wait_s``) is always better than polling; this is for a consumer that polls.
    """
    if status in TERMINAL_JOB_STATUSES:
        return 0.0
    if phase == "waiting_for_gpu":
        return GPU_RECHECK_S
    if status == "queued":
        if wait_to_start_s is None:
            return POLL_DEFAULT_QUEUED_S
        return round(_clamp(wait_to_start_s / 4.0, POLL_QUEUED_MIN_S, POLL_QUEUED_MAX_S), 1)
    if eta is None:
        return POLL_DEFAULT_RUNNING_S
    return round(_clamp(eta / 10.0, POLL_MIN_S, POLL_RUNNING_MAX_S), 1)


# ======================================================================== retry_after_s


def queue_full_retry_after_s(est_drain: float | None, queue_length: int) -> float:
    """``QUEUE_FULL``'s ``retry_after_s``: about when one job will have left the queue (its drain estimate
    shared out over the jobs in it), within ``RETRY_MIN_S``..``QUEUE_FULL_MAX_S``."""
    if est_drain is None or queue_length <= 0 or not math.isfinite(est_drain):
        return QUEUE_FULL_DEFAULT_S
    return round(_clamp(est_drain / queue_length, RETRY_MIN_S, QUEUE_FULL_MAX_S), 1)


@dataclass(frozen=True, slots=True)
class RateWindow:
    """The submit rate cap (``admission.rate``): how many submissions are left in the window, and in how many
    seconds the next one frees (0 when some are left)."""

    remaining: int
    resets_in_s: float


def rate_window(
    count_since: Callable[[float], int], *, now: float, max_per_window: int, window_s: float = RATE_WINDOW_S
) -> RateWindow:
    """The rate cap from a count of the jobs created since a time (Unix seconds).

    ``count_since(t)`` is ``Store.jobs_created_since`` over ``t`` (the caller turns ``t`` into ISO time). When
    the window is full, the moment it frees is found by bisection: the earliest ``t`` in the next
    ``window_s`` at which fewer than ``max_per_window`` jobs remain in the window ``[now + t - window_s, …)``.
    It is exact to 0.1 s and costs a dozen counts, only when a submission is refused.
    """
    if max_per_window <= 0:
        return RateWindow(remaining=0, resets_in_s=window_s)
    used = count_since(now - window_s)
    if used < max_per_window:
        return RateWindow(remaining=max_per_window - used, resets_in_s=0.0)
    low, high = 0.0, window_s
    while high - low > 0.1:
        mid = (low + high) / 2.0
        if count_since(now + mid - window_s) < max_per_window:
            high = mid
        else:
            low = mid
    return RateWindow(remaining=0, resets_in_s=round(max(high, RETRY_MIN_S), 1))


# ======================================================================== admission


def admission(
    *,
    queue_length: int,
    max_queued: int,
    est_drain: float | None,
    rate: RateWindow,
    daemon: DaemonStatus | None,
) -> dict[str, Any]:
    """``get_server_status``'s ``admission`` (DC-2): {accepting, queue {length, max, est_drain_s}, rate
    {remaining, resets_in_s}, gpu {in_use, holder, free_mb, need_mb by group, waiting_since}}.

    ``accepting`` says whether a submission now would be queued rather than refused: the queue has room and
    the rate cap allows one. Whether narration may run beside other GPU work stays the consumer's decision.
    """
    gpu = daemon.gpu if daemon is not None else None
    return {
        "accepting": queue_length < max_queued and rate.remaining > 0,
        "queue": {"length": queue_length, "max": max_queued, "est_drain_s": est_drain},
        "rate": {"remaining": rate.remaining, "resets_in_s": rate.resets_in_s},
        "gpu": {
            "in_use": gpu.in_use if gpu is not None else False,
            "holder": gpu.holder if gpu is not None else None,
            "free_mb": gpu.free_mb if gpu is not None else None,
            "need_mb": dict(gpu.need_mb) if gpu is not None else {},
            "waiting_since": gpu.waiting_since if gpu is not None else None,
        },
    }


__all__ = [
    "CHARS_PER_AUDIO_S",
    "GPU_RECHECK_S",
    "GPU_UNAVAILABLE_RETRY_S",
    "LOADS_PER_JOB",
    "MODEL_LOAD_S",
    "RATE_WINDOW_S",
    "WALL_PER_AUDIO_S",
    "RateWindow",
    "Throughput",
    "admission",
    "est_drain_s",
    "eta_s",
    "job_wall_s",
    "poll_after_s",
    "queue_full_retry_after_s",
    "queue_position",
    "rate_window",
    "request_audio_s",
    "scheduling_order",
]
