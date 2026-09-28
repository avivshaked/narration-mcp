"""What the ``design`` and ``profile`` job kinds share (design sections 3.1, 3.6, 4, 8 and 14; plan.md WP34).

Both kinds run a piece at a time behind the runner (``narration.jobs.handlers.JobHandler``), on the job engine's
core: its model residency (one resident group across every kind, section 4), its throughput estimate and its
canary outcome. Neither renders through the engine's stages (render, post-process, score); each has its own
pieces, and shares these:

- ``StepRun``: a held job's id, scratch folder, phase, message and retry counts.
- ``Pieces``: one piece of work, with a worker's failure turned into a retry or a job-level error. The rules
  are the engine's (section 4 item 5, section 14): out of GPU memory, unload, wait and retry once, then
  ``GPU_UNAVAILABLE`` (retryable); a worker that crashed, timed out or lost its models, once more, then
  ``INTERNAL`` (retryable); a worker that is not installed, ``BACKEND_NOT_INSTALLED``; any other refusal,
  ``INTERNAL`` (retryable, since what was made is kept and the same request resumes). A design is a handful
  of candidates, so a candidate short of its clip, transcript check or profile fails the job rather than
  leave a gap in it; the same request designs the same candidates again (``seeds``).
- ``Endings``: the job record between pieces (``save``) and at the end (``complete``, ``cancel``, ``fail``),
  each by compare-and-set, so a cancel made meanwhile is never undone.
- ``profile_record``: a ``ProfileRecord`` from the QA worker's ``profile`` reply (App. A).
"""

from __future__ import annotations

import errno
import logging
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from narration.contracts import codes, names
from narration.contracts.errors import NarrationError, WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.interfaces import WorkerClient
from narration.contracts.models import (
    JobRecord,
    ProfileMeasurements,
    ProfilePictures,
    ProfileRecord,
    Progress,
)
from narration.contracts.names import GpuHolder, JobOutcome, JobPhase
from narration.contracts.serial import ContractError, from_json
from narration.jobs.admission import GPU_UNAVAILABLE_RETRY_S, STORE_FULL_RETRY_S
from narration.jobs.core import LOST_STATE, EngineCore, worker_code
from narration.jobs.failures import MAX_RETRIES, OOM_WAIT_S
from narration.jobs.host import RunnerHost
from narration.jobs.stages import Stages
from narration.jobs.state import Outcome

log = logging.getLogger(__name__)

RETRY_AFTER_S: Final = 60.0
"""``retry_after_s`` of a design or profile job that stopped on a worker's failure (DC-2)."""
PICTURES: Final = ("spectrogram", "pitch")
"""The profile's pictures by name (``ProfilePictures``), as the QA worker's ``profile`` reply names them."""


@dataclass(slots=True, eq=False, kw_only=True)
class StepRun:
    """A ``design`` or ``profile`` job the handler holds: the job, its scratch folder (``scratch/jobs/<job_id>/``,
    removed when the job ends or is given back), its phase and message, and the retries of the piece in hand."""

    job: JobRecord
    scratch: Path
    phase: JobPhase | None = None
    message: str | None = None
    oom_retries: int = 0
    """Section 4 item 5: retries of the piece in hand after running out of GPU memory."""
    retries: int = 0
    """Retries of the piece in hand after its worker crashed, timed out or lost its models."""
    done_floor: float = 0.0
    """The most ``progress.done_s`` the job has shown: progress never goes back."""
    prepared: WorkerClient | None = None
    """The Qwen worker loaded for this job since its last load (``narration.jobs.core.ActiveRun``): a load or a
    failure of the worker clears it. A profile job never loads Qwen and leaves it None."""

    @property
    def job_id(self) -> str:
        """The job's id."""
        return self.job.job_id


def set_phase(host: RunnerHost, run: StepRun, phase: JobPhase) -> None:
    """Set the job's phase, and tell the daemon when it changes (``EngineCore.phase``, for any ``StepRun``)."""
    if run.phase != phase:
        run.phase = phase
        host.job_phase(phase)


class Pieces:
    """Runs one piece of a job's work and handles what its worker raises (see the module docstring)."""

    def __init__(self, core: EngineCore, stages: Stages) -> None:
        self.core = core
        self.stages = stages

    def run(self, host: RunnerHost, run: StepRun, group: GpuHolder, what: str, piece: Callable[[], Outcome]) -> Outcome:
        """Run ``piece`` (whose worker is ``group``'s; ``what`` names it in errors, e.g. "design candidate 2")."""
        try:
            outcome = piece()
        except WorkerFailure as exc:
            if host.should_stop():
                return "stopped"
            code = worker_code(exc)
            if code == codes.BACKEND_NOT_INSTALLED:
                raise NarrationError(
                    codes.BACKEND_NOT_INSTALLED,
                    exc.message,
                    details=exc.details,
                    hint="Ask the operator to run narration-admin doctor, then narration-admin install.",
                ) from exc
            if code == codes.GPU_OOM:
                return self._oom(host, run, group, what, exc)
            if code in LOST_STATE:
                self.core.residency.forget(group)
                return self._again(run, what, exc)
            raise NarrationError(
                codes.INTERNAL,
                f"the {group} worker could not {what}: {exc.message}",
                details={"worker_code": code},
                retryable=True,
                retry_after_s=RETRY_AFTER_S,
                hint="Send the same request again; if it fails again, the daemon's log has the details.",
            ) from exc
        except (WorkerCrashed, WorkerTimeout) as exc:
            if host.stop_mode == "now" or host.should_stop():
                return "stopped"  # the daemon killed the workers: the job goes back to the queue as it is
            self.core.residency.forget(group)
            if group == "qwen":
                run.prepared = None
            return self._again(run, what, exc)
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise NarrationError(
                    codes.STORE_FULL,
                    f"the store's disk is full: {exc}",
                    retry_after_s=STORE_FULL_RETRY_S,
                    hint="Ask the operator to free space on the store's disk (narration-admin gc), then send the "
                    "same request again.",
                ) from exc
            raise
        if outcome == "worked":
            run.oom_retries = 0
            run.retries = 0
        return outcome

    def _oom(self, host: RunnerHost, run: StepRun, group: GpuHolder, what: str, exc: WorkerFailure) -> Outcome:
        """Section 4 item 5: unload, wait, retry once; then ``GPU_UNAVAILABLE`` for the job."""
        if run.oom_retries >= MAX_RETRIES:
            raise NarrationError(
                codes.GPU_UNAVAILABLE,
                f"out of GPU memory to {what}, after one retry: {exc.message}",
                details={"group": group},
                retry_after_s=GPU_UNAVAILABLE_RETRY_S,
                hint="Send the same request again once the GPU has more free memory (get_server_status).",
            ) from exc
        run.oom_retries += 1
        log.warning("job %s: out of GPU memory to %s; unloading, waiting, once more", run.job_id, what)
        self.stages.drop(host, group)
        if group == "qwen":
            run.prepared = None
        return "waited" if host.sleep(OOM_WAIT_S) else "stopped"

    @staticmethod
    def _again(run: StepRun, what: str, exc: Exception) -> Outcome:
        if run.retries >= MAX_RETRIES:
            raise NarrationError(
                codes.INTERNAL,
                f"a worker failed twice to {what}: {exc}",
                details={"exception": type(exc).__name__},
                retryable=True,
                retry_after_s=RETRY_AFTER_S,
                hint="Send the same request again; if it fails again, the daemon's log has the details.",
            ) from exc
        run.retries += 1
        log.warning("job %s: a worker failed to %s (%s); once more", run.job_id, what, exc)
        return "worked"


class Endings:
    """The job record of a ``StepRun`` between pieces and at the end (see the module docstring)."""

    @staticmethod
    def progress(run: StepRun, done_s: float, total_s: float, done: int, total: int, *, complete: bool) -> Progress:
        """Progress in estimated audio seconds (section 7.4); ``done_s`` never goes back."""
        if complete:
            done_s = total_s
        done_s = max(done_s, run.done_floor)
        run.done_floor = done_s
        total_s = max(total_s, done_s)
        return Progress(
            done_s=round(done_s, 3),
            total_s=round(total_s, 3),
            fraction=round(done_s / total_s, 4) if total_s > 0 else 1.0,
            segments_done=done,
            segments_total=total,
        )

    @staticmethod
    def save(host: RunnerHost, run: StepRun, progress: Progress, result: Mapping[str, Any] | None) -> bool:
        """Write the job's phase, progress, message and (when given) its result so far; False when it is no longer
        ``running`` (cancelled meanwhile): nothing is written then."""
        changes: dict[str, Any] = {"phase": run.phase, "progress": progress, "message": run.message}
        if result is not None:
            changes["result"] = dict(result)
        return host.store.update_job(run.job_id, expect_status="running", **changes) is not None

    def complete(
        self, host: RunnerHost, run: StepRun, *, outcome: JobOutcome, progress: Progress, result: Mapping[str, Any]
    ) -> JobRecord | None:
        """Complete the job with its result. A job cancelled as it finished ends ``cancelled``, with what it made."""
        store = host.store
        updated = store.update_job(
            run.job_id,
            expect_status="running",
            status="completed",
            phase=None,
            outcome=outcome,
            progress=progress,
            result=dict(result),
            message=run.message,
        )
        if updated is None:
            current = store.get_job(run.job_id)
            if current is not None and current.status == "cancelling":
                updated = store.update_job(
                    run.job_id,
                    expect_status="cancelling",
                    status="cancelled",
                    phase=None,
                    progress=progress,
                    result=dict(result),
                    message="cancelled as it finished; what it made is kept",
                )
        log.info("job %s: %s", run.job_id, run.message)
        self.release(run)
        return updated

    def cancel(
        self, host: RunnerHost, run: StepRun, progress: Progress, result: Mapping[str, Any] | None
    ) -> JobRecord | None:
        """Finish a cancelled job: what it published stays (``result``); nothing else is published."""
        run.message = "cancelled; what it finished is kept"
        changes: dict[str, Any] = {"phase": None, "progress": progress, "message": run.message}
        if result is not None:
            changes["result"] = dict(result)
        updated = host.store.update_job(run.job_id, expect_status="cancelling", status="cancelled", **changes)
        self.release(run)
        return updated

    def fail(
        self, host: RunnerHost, job: JobRecord, error: NarrationError, run: StepRun | None, progress: Progress | None
    ) -> JobRecord | None:
        """Record a job-level failure: status ``failed`` with the error. A job being cancelled is cancelled."""
        changes: dict[str, Any] = {"phase": None, "error": error.error, "message": f"failed: {error.message}"}
        if progress is not None:
            changes["progress"] = progress
        updated = host.store.update_job(job.job_id, expect_status="running", status="failed", **changes)
        if updated is None:
            current = host.store.get_job(job.job_id)
            if current is not None and current.status == "cancelling":
                updated = host.store.update_job(
                    job.job_id, expect_status="cancelling", status="cancelled", phase=None, message="cancelled"
                )
        if run is not None:
            self.release(run)
        log.warning("job %s failed: %s", job.job_id, error)
        return updated

    @staticmethod
    def release(run: StepRun) -> None:
        """Remove the job's scratch files: work in flight is never published half made."""
        shutil.rmtree(run.scratch, ignore_errors=True)


class ProfileReplyError(ValueError):
    """The QA worker's ``profile`` reply cannot be read as a ``ProfileRecord`` (a silent file among the causes:
    it has no brightness to measure)."""


def profile_record(audio_sha256: str, reply: Mapping[str, Any]) -> ProfileRecord:
    """A ``ProfileRecord`` of the audio from the QA worker's ``profile`` reply (``protocol.ProfileReply``): its
    measurements as ``ProfileMeasurements`` has them, and the pictures by the worker's paths (``put_profile``
    moves them into the store). Raises ``ProfileReplyError`` when the reply does not fit the record."""
    measurements = reply.get("measurements")
    pictures = reply.get("pictures")
    if not isinstance(measurements, dict) or not isinstance(pictures, dict):
        raise ProfileReplyError("the profile reply has no measurements or pictures")
    try:
        measured = from_json(ProfileMeasurements, measurements, path="$.measurements")
    except ContractError as exc:
        raise ProfileReplyError(str(exc)) from exc
    paths = {name: pictures.get(name) for name in PICTURES}
    if not all(isinstance(p, str) and p for p in paths.values()):
        raise ProfileReplyError(f"the profile reply names the pictures {sorted(pictures)}, not {list(PICTURES)}")
    return ProfileRecord(
        audio_sha256=audio_sha256,
        profile_version=names.PROFILE_VERSION,
        measurements=measured,
        pictures=ProfilePictures(spectrogram=str(paths["spectrogram"]), pitch=str(paths["pitch"])),
    )


__all__ = [
    "PICTURES",
    "RETRY_AFTER_S",
    "Endings",
    "Pieces",
    "ProfileReplyError",
    "StepRun",
    "profile_record",
    "set_phase",
]
