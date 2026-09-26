"""The claim loop: the job engine behind the daemon's seam (``JobRunner``; plan.md WP30 and WP31).

``EngineRunner.step`` does one thing and returns:

- **Holding no job**, it claims the next one (``Store.claim_next_job``: ``interactive`` before ``batch``,
  then first come first served) and plans it from its request and the cache.
- **Holding a job**, it first reads the job again. A job being cancelled is finished as ``cancelled`` (what
  it finished stays in the cache). A ``batch`` job gives way to a queued ``interactive`` one: it goes back to
  the queue (``return_job``) with its items kept, and is taken again later, when the cache answers for
  everything it had done. Otherwise the engine does one piece of the job's work (``JobEngine.advance``) and
  writes the job's progress, items and phase.

Which model group stays loaded between jobs is the engine's (``gpu.Residency``): a job whose first work
needs the resident group starts without a load. Every generation job uses the current Base engine profile,
so jobs need no grouping by profile until design jobs run here too.

A job-level failure (``NarrationError``) fails the job with its error. An exception the engine does not
expect is a bug: the job fails with ``INTERNAL`` rather than being tried again at every step.

``shutdown`` gives back the job held (``return_job``: ``running`` goes back to ``queued``, ``cancelling``
becomes ``cancelled``) and removes its scratch files. Leases are released at the end of each piece of work
(and lapse by their TTL if the process dies), so none is held between steps.

The drain estimate the daemon publishes (DC-2's ``admission.queue.est_drain_s``) is refreshed after each step.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Final

from narration.config import Config
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import AlignerCore
from narration.contracts.models import JobRecord
from narration.post import DeliveryPipeline
from narration.qa import Scorer
from narration.text import TextPipeline

from .admission import est_drain_s
from .engine import EngineParts, JobEngine, JobRun
from .gpu import NoProbe, NvmlProbe, VramProbe
from .hooks import EngineGuard, NoGuard
from .host import RunnerHost, ShutdownReason, return_job
from .pins import QaPins
from .voice import PathCheck

log = logging.getLogger(__name__)

PREEMPTING: Final = "interactive"
"""The priority that makes a held ``batch`` job give way between two pieces of its work."""


class EngineRunner:
    """The ``JobRunner`` the daemon drives (see the module docstring). One instance serves one daemon."""

    def __init__(self, engine: JobEngine) -> None:
        self.engine = engine
        self._job: JobRecord | None = None
        self._run: JobRun | None = None

    @property
    def job_id(self) -> str | None:
        """The job held, or None."""
        return self._job.job_id if self._job is not None else None

    # ------------------------------------------------------------------ the JobRunner protocol
    def step(self, host: RunnerHost) -> bool:
        """Claim a job, or do one piece of the held job's work. False when there was nothing to do."""
        try:
            if self._job is None:
                return self._claim(host)
            return self._advance(host)
        finally:
            self._publish_drain(host)

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        """Give back the job held, if any, and remove its scratch files."""
        if self._job is None:
            return
        self._let_go(host, f"the daemon stopped ({reason})")

    # ------------------------------------------------------------------ claiming
    def _claim(self, host: RunnerHost) -> bool:
        job = host.store.claim_next_job(host.holder)
        if job is None:
            return False
        self._job = job
        host.job_started(job)
        log.info("took job %s (%s, %s)", job.job_id, job.kind, job.priority)
        try:
            self._run = self.engine.open(host, job)
        except NarrationError as exc:
            self._failed(host, exc)
            return True
        except Exception as exc:
            log.exception("job %s could not be planned", job.job_id)
            self._failed(host, _internal(exc))
            return True
        if not self.engine.save(host, self._run):
            self._changed(host)
        return True

    # ------------------------------------------------------------------ one piece of work
    def _advance(self, host: RunnerHost) -> bool:
        job, run = self._job, self._run
        assert job is not None and run is not None
        current = host.store.get_job(job.job_id)
        if current is None or current.status != "running":
            self._changed(host, current)
            return True
        if job.priority != PREEMPTING and self._interactive_waiting(host):
            log.info("job %s gives way to an interactive job", job.job_id)
            self._let_go(host, "an interactive job came first")
            return True
        try:
            outcome = self.engine.advance(host, run)
        except NarrationError as exc:
            self._failed(host, exc)
            return True
        except Exception as exc:
            log.exception("job %s: the engine failed", job.job_id)
            self._failed(host, _internal(exc))
            return True
        if outcome == "finished":
            self._done(host)
            return True
        if outcome == "stopped":
            return True  # the daemon is stopping; shutdown gives the job back
        if not self.engine.save(host, run):
            self._changed(host)
        return True

    def _interactive_waiting(self, host: RunnerHost) -> bool:
        return any(j.status == "queued" and j.priority == PREEMPTING for j in host.store.queued_jobs())

    # ------------------------------------------------------------------ endings
    def _changed(self, host: RunnerHost, current: JobRecord | None = None) -> None:
        """The job is no longer ``running`` under us: finish a cancel, or let go of it as it is."""
        job = self._job
        assert job is not None
        current = current if current is not None else host.store.get_job(job.job_id)
        if current is not None and current.status == "cancelling":
            if self._run is not None:
                self.engine.cancel(host, self._run)
            else:
                host.store.update_job(job.job_id, expect_status="cancelling", status="cancelled", phase=None)
            log.info("job %s cancelled", job.job_id)
        else:
            log.info("job %s is %s; letting go of it", job.job_id, current.status if current else "gone")
            if self._run is not None:
                self.engine.release(self._run)
        self._done(host)

    def _failed(self, host: RunnerHost, error: NarrationError) -> None:
        assert self._job is not None
        self.engine.fail(host, self._job, error, self._run)
        self._done(host)

    def _let_go(self, host: RunnerHost, reason: str) -> None:
        """Give the job back to the queue as it is (its finished work is in the cache)."""
        assert self._job is not None
        if self._run is not None:
            host.store.update_job(
                self._job.job_id,
                expect_status="running",
                round=self._run.round,
                progress=self.engine.progress(self._run),
                items=self.engine.items(self._run),
            )
            self.engine.release(self._run)
        return_job(host.store, self._job.job_id, reason=reason)
        self._done(host)

    def _done(self, host: RunnerHost) -> None:
        self._job = None
        self._run = None
        self.engine.residency.reset_wait(host)
        host.job_finished()

    # ------------------------------------------------------------------ the drain estimate (DC-2)
    def _publish_drain(self, host: RunnerHost) -> None:
        try:
            queued = host.store.queued_jobs()
        except Exception:  # the estimate is advice: a store hiccup must not fail a step
            log.debug("the queue could not be read for the drain estimate", exc_info=True)
            return
        held = {self._run.job_id: self.engine.remaining_audio_s(self._run)} if self._run is not None else None
        rate = self.engine.throughput.wall_per_audio_s
        host.set_est_drain(est_drain_s(queued, wall_per_audio_s=rate, running_remaining_s=held) if queued else 0.0)


def _internal(exc: BaseException) -> NarrationError:
    return NarrationError(
        codes.INTERNAL,
        f"the job engine failed: {type(exc).__name__}: {exc}",
        retryable=False,
        hint="This is a bug in the service; the daemon's log has the details. Nothing was published half made.",
    )


def build_runner(
    config: Config,
    *,
    aligner: AlignerCore,
    qa_pins: QaPins,
    guard: EngineGuard | None = None,
    probe: VramProbe | None = None,
    check_path: PathCheck | None = None,
    clock: Callable[[], float] | None = None,
) -> EngineRunner:
    """The runner the daemon uses: the service's text pipeline, post-processing and QA, configured by
    ``config``, with the aligner and QA pins the installation provides (WP15, WP22) and WP32's engine guard.
    ``probe`` defaults to NVML on a ``cuda`` device and to no check otherwise."""
    if probe is None:
        probe = NvmlProbe(config.gpu.device) if config.gpu.device.startswith("cuda") else NoProbe()
    parts = EngineParts(
        text=TextPipeline(config.text),
        delivery=DeliveryPipeline(fade_s=config.delivery.fade_s),
        scorer=Scorer(config.measurement),
        aligner=aligner,
        qa_pins=qa_pins,
        guard=guard if guard is not None else NoGuard(),
        probe=probe,
        check_path=check_path,
        clock=clock if clock is not None else time.monotonic,
    )
    return EngineRunner(JobEngine(config, parts))


__all__ = ["PREEMPTING", "EngineRunner", "build_runner"]
