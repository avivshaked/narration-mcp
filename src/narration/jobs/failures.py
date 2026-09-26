"""A worker's failure, turned into a retry, a flag or a job error (design sections 4 and 14).

A job-level problem raises ``NarrationError`` for the runner to record on the job: a worker that cannot be
installed or loaded, or a store whose disk is full. A take-level execution problem is flagged on its
segment (severity ``error``), and the job goes on with the rest:

- out of GPU memory: unload, wait, retry once, then ``GPU_OOM`` (section 4 item 5);
- a worker that crashed or timed out twice (``WORKER_CRASHED``): each time it is started again;
- a worker that lost its models or the prepared voice: loaded and prepared again, once;
- anything else the worker or the audio tools refuse: ``RENDER_FAILED`` or ``QA_UNAVAILABLE``.

An execution problem is never retaken (section 8): a retake is for a take QA failed.
"""

from __future__ import annotations

import errno
import logging
from typing import Any, Final

from narration.contracts import codes
from narration.contracts.errors import NarrationError, QaUnavailable, WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.models import Flag
from narration.contracts.names import GpuHolder
from narration.contracts.worker import WorkerErrorCode

from .core import EngineCore, worker_code
from .host import RunnerHost
from .stages import Stages
from .state import Attempt, JobRun, Outcome, label

log = logging.getLogger(__name__)

OOM_WAIT_S: Final = 15.0
"""Section 4 item 5: on an out-of-memory error, unload, wait this long, and retry once."""
MAX_RETRIES: Final = 1
"""Retries of one piece of work after an out-of-memory error, a crash or a timeout of its worker."""
LOST_STATE: Final[tuple[WorkerErrorCode, ...]] = ("NOT_LOADED", "VOICE_NOT_PREPARED")
"""The worker codes that say it lost its models or the prepared voice (unloaded meanwhile)."""


class Failures:
    """Runs one piece of work and handles what its worker raises (see the module docstring)."""

    def __init__(self, core: EngineCore, stages: Stages) -> None:
        self.core = core
        self.stages = stages

    def guarded(self, host: RunnerHost, run: JobRun, attempt: Attempt, group: GpuHolder | None) -> Outcome:
        """Run the attempt's next stage; turn a worker's failure into a retry, a flag, or a job-level error."""
        stages = self.stages
        work = {"render": stages.render, "post": stages.post, "score": stages.score}[attempt.stage]
        try:
            return work(host, run, attempt)
        except WorkerFailure as exc:
            if host.should_stop():
                return "stopped"
            code = worker_code(exc)
            if code == codes.BACKEND_NOT_INSTALLED:
                raise NarrationError(codes.BACKEND_NOT_INSTALLED, exc.message, details=exc.details) from exc
            if code == codes.GPU_OOM and group is not None:
                return self._oom(host, run, attempt, group, exc)
            if code in LOST_STATE and group is not None:
                return self._lost_state(run, attempt, group, exc)
            flag = codes.QA_UNAVAILABLE if group == "qa" else codes.RENDER_FAILED
            self.fail_attempt(run, attempt, flag, f"the worker could not {_verb(group)} it: {exc}", exc.code)
            return "worked"
        except (WorkerCrashed, WorkerTimeout) as exc:
            if host.stop_mode == "now" or host.should_stop():
                return "stopped"  # the daemon killed the workers: the job goes back to the queue as it is
            if group is not None:
                self.core.residency.forget(group)
            if group == "qwen":
                run.prepared = None
            if attempt.retries < MAX_RETRIES:
                attempt.retries += 1
                log.warning(
                    "job %s: the %s worker failed on %s (%s); once more", run.job_id, group, label(run, attempt), exc
                )
                return "worked"
            self.fail_attempt(run, attempt, codes.WORKER_CRASHED, f"the {group} worker failed twice on it: {exc}", None)
            return "worked"
        except QaUnavailable as exc:
            self.fail_attempt(run, attempt, codes.QA_UNAVAILABLE, f"QA could not score it: {exc}", None)
            return "worked"
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise NarrationError(codes.STORE_FULL, f"the store's disk is full: {exc}", retry_after_s=60.0) from exc
            log.exception("job %s: %s failed", run.job_id, label(run, attempt))
            code = codes.QA_UNAVAILABLE if group == "qa" else codes.RENDER_FAILED
            self.fail_attempt(run, attempt, code, f"a file could not be read or written: {exc}", None)
            return "worked"
        except (ValueError, RuntimeError) as exc:  # unreadable audio (soundfile), audio post-processing refuses
            log.exception("job %s: %s failed", run.job_id, label(run, attempt))
            code = codes.QA_UNAVAILABLE if group == "qa" else codes.RENDER_FAILED
            self.fail_attempt(
                run, attempt, code, f"it could not be {'scored' if group == 'qa' else 'made'}: {exc}", None
            )
            return "worked"

    def _oom(self, host: RunnerHost, run: JobRun, attempt: Attempt, group: GpuHolder, exc: WorkerFailure) -> Outcome:
        """Section 4 item 5: unload, wait, retry once; then the take slot fails with ``GPU_OOM``."""
        if attempt.retries >= MAX_RETRIES:
            self.fail_attempt(
                run, attempt, codes.GPU_OOM, f"out of GPU memory after one retry: {exc.message}", exc.code
            )
            return "worked"
        attempt.retries += 1
        log.warning("job %s: out of GPU memory on %s; unloading, waiting, once more", run.job_id, label(run, attempt))
        try:
            self.core.residency.unload(host, group)
        except (WorkerFailure, WorkerCrashed, WorkerTimeout) as unload_exc:
            log.warning("unloading the %s group after running out of memory failed: %s", group, unload_exc)
            self.core.residency.forget(group)
        if group == "qwen":
            run.prepared = None
        return "waited" if host.sleep(OOM_WAIT_S) else "stopped"

    def _lost_state(self, run: JobRun, attempt: Attempt, group: GpuHolder, exc: WorkerFailure) -> Outcome:
        """The worker lost its models or the prepared voice (unloaded meanwhile): load again, once."""
        self.core.residency.forget(group)
        run.prepared = None
        if attempt.retries >= MAX_RETRIES:
            code = codes.QA_UNAVAILABLE if group == "qa" else codes.RENDER_FAILED
            self.fail_attempt(run, attempt, code, f"the worker lost its models twice: {exc}", exc.code)
            return "worked"
        attempt.retries += 1
        return "worked"

    @staticmethod
    def fail_attempt(run: JobRun, attempt: Attempt, code: str, why: str, worker_code: str | None) -> None:
        """Record a take-level execution problem on the attempt and its segment (severity ``error``)."""
        seg = run.segments[attempt.segment]
        details: dict[str, Any] = {"attempt": attempt.attempt, "slot": attempt.slot, "stage": attempt.stage}
        if worker_code is not None:
            details["worker_code"] = worker_code
        flag = Flag(
            code=code,
            severity="error",
            message=f"{seg.segment_id}, attempt {attempt.attempt}: {why}",
            segment_id=seg.segment_id,
            retake_trigger=False,
            details=details,
        )
        attempt.error = flag
        seg.flags.append(flag)
        log.warning("job %s: %s", run.job_id, flag.message)


def _verb(group: GpuHolder | None) -> str:
    return "render" if group == "qwen" else "score" if group == "qa" else "post-process"


__all__ = ["LOST_STATE", "MAX_RETRIES", "OOM_WAIT_S", "Failures"]
