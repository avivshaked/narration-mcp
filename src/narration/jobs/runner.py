"""The claim loop: the job engine behind the daemon's seam (``JobRunner``; plan.md WP30 and WP31).

``EngineRunner.step`` does one thing and returns:

- **Holding no job**, it claims the next one (``Store.claim_next_job``: ``interactive`` before ``batch``,
  then first come first served) and plans it from its request and the cache.
- **Holding a job**, it first reads the job again. A job being cancelled is finished as ``cancelled`` (what
  it finished stays in the cache). A ``batch`` job gives way to a queued ``interactive`` one: it goes back to
  the queue (``return_job``) with its items kept, and is taken again later, when the cache answers for
  everything it had done. Otherwise the engine does one piece of the job's work (``JobEngine.advance``) and
  writes the job's progress, items and phase.

A claimed job goes to the handler of its kind (``handlers.Registry``: the job engine runs ``generate`` and
``analyse``; WP33 to WP35 add theirs). A kind with no handler fails with ``INTERNAL`` and ``details.kind``.

Which model group stays loaded between jobs is the residency's (``gpu.Residency``, shared by every handler):
a job whose first work needs the resident group starts without a load. Every generation job uses the
current Base engine profile, so jobs need no grouping by profile until design jobs run here too.

A job-level failure (``NarrationError``) fails the job with its error. An exception the engine does not
expect is a bug: the job fails with ``INTERNAL`` rather than being tried again at every step.

``has_work`` answers whether a ``step`` would find work (a job held, or one queued) and claims nothing; the
daemon asks it before an idle exit. ``default_runner`` is the zero-argument factory the daemon loads by
name; it builds the engine at the first job, from the daemon's config.

``shutdown`` gives back the job held (``return_job``: ``running`` goes back to ``queued``, ``cancelling``
becomes ``cancelled``) and removes its scratch files. Leases are released at the end of each piece of work
(and lapse by their TTL if the process dies), so none is held between steps.

The drain estimate the daemon publishes (DC-2's ``admission.queue.est_drain_s``) is refreshed after each step.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Final

from narration.config import Config
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import AlignerCore
from narration.contracts.models import JobRecord
from narration.post import DeliveryPipeline
from narration.qa import Scorer
from narration.text import TextPipeline

from .admission import WALL_PER_AUDIO_S, est_drain_s
from .core import EngineParts
from .engine import JobEngine
from .gpu import NoProbe, NvmlProbe, VramProbe
from .handlers import JobHandler, Registry
from .hooks import EngineGuard, NoGuard
from .host import RunnerHost, ShutdownReason, return_job
from .pins import QaPins
from .voice import PathCheck

log = logging.getLogger(__name__)

PREEMPTING: Final = "interactive"
"""The priority that makes a held ``batch`` job give way between two pieces of its work."""


EngineFactory = Callable[[RunnerHost], "JobEngine | Registry"]
"""Builds the engine (or the registry of handlers) from what the daemon hands the runner, at the first job."""


class EngineRunner:
    """The ``JobRunner`` the daemon drives (see the module docstring). One instance serves one daemon.

    It takes the job engine, a ``Registry`` of handlers by kind, or a factory that builds either at the first
    job claimed (the daemon constructs its runner before it has a host). A factory that raises
    ``NarrationError`` fails that job with the error, and is asked again at the next job, so an installation
    completed meanwhile is picked up.
    """

    def __init__(self, source: JobEngine | Registry | EngineFactory) -> None:
        self._registry: Registry | None = None
        self._build: EngineFactory | None = None
        if isinstance(source, JobEngine | Registry):
            self._registry = _registry(source)
        else:
            self._build = source
        self._job: JobRecord | None = None
        self._handler: JobHandler[Any] | None = None
        self._run: Any = None

    @property
    def registry(self) -> Registry:
        """The handlers by kind; a runner built from a factory has them once it has claimed a job."""
        if self._registry is None:
            raise RuntimeError("the job engine is built when the first job is claimed")
        return self._registry

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

    def has_work(self, host: RunnerHost) -> bool:
        """Whether a ``step`` now would find work: a job held, or one queued. Nothing is claimed."""
        if self._job is not None:
            return True
        return any(job.status == "queued" for job in host.store.queued_jobs())

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
            if self._registry is None:
                assert self._build is not None
                self._registry = _registry(self._build(host))
            handler = self._registry.handler(job.kind)
            if handler is None:
                raise NarrationError(
                    codes.INTERNAL,
                    f"this build of the service does not run {job.kind} jobs",
                    retryable=False,
                    hint=f"The {job.kind} job engine is not in this build; nothing was made. Update the service.",
                    details={"kind": job.kind},
                )
            self._handler = handler
            self._run = handler.open(host, job)
        except NarrationError as exc:
            self._failed(host, exc)
            return True
        except Exception as exc:
            log.exception("job %s could not be planned", job.job_id)
            self._failed(host, _internal(exc))
            return True
        if not handler.save(host, self._run):
            self._changed(host)
        return True

    # ------------------------------------------------------------------ one piece of work
    def _advance(self, host: RunnerHost) -> bool:
        job, handler, run = self._job, self._handler, self._run
        assert job is not None and handler is not None and run is not None
        current = host.store.get_job(job.job_id)
        if current is None or current.status != "running":
            self._changed(host, current)
            return True
        if job.priority != PREEMPTING and self._interactive_waiting(host):
            log.info("job %s gives way to an interactive job", job.job_id)
            self._let_go(host, "an interactive job came first")
            return True
        try:
            outcome = handler.advance(host, run)
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
        if not handler.save(host, run):
            self._changed(host)
        return True

    def _interactive_waiting(self, host: RunnerHost) -> bool:
        return any(j.status == "queued" and j.priority == PREEMPTING for j in host.store.queued_jobs())

    # ------------------------------------------------------------------ endings
    def _changed(self, host: RunnerHost, current: JobRecord | None = None) -> None:
        """The job is no longer ``running`` under us: finish a cancel, or let go of it as it is."""
        job, handler = self._job, self._handler
        assert job is not None
        current = current if current is not None else host.store.get_job(job.job_id)
        if current is not None and current.status == "cancelling":
            if handler is not None and self._run is not None:
                handler.cancel(host, self._run)
            else:
                host.store.update_job(job.job_id, expect_status="cancelling", status="cancelled", phase=None)
            log.info("job %s cancelled", job.job_id)
        else:
            log.info("job %s is %s; letting go of it", job.job_id, current.status if current else "gone")
            if handler is not None and self._run is not None:
                handler.release(self._run)
        self._done(host)

    def _failed(self, host: RunnerHost, error: NarrationError) -> None:
        job = self._job
        assert job is not None
        if self._handler is not None:
            self._handler.fail(host, job, error, self._run)
        else:  # no handler for the job (none built, or none for its kind): record the failure without one
            failed = host.store.update_job(
                job.job_id,
                expect_status="running",
                status="failed",
                phase=None,
                error=error.error,
                message=f"failed: {error.message}",
            )
            if failed is None:
                host.store.update_job(job.job_id, expect_status="cancelling", status="cancelled", phase=None)
            log.warning("job %s failed: %s", job.job_id, error)
        self._done(host)

    def _let_go(self, host: RunnerHost, reason: str) -> None:
        """Give the job back to the queue as it is (its finished work is in the cache). A job cancelled
        meanwhile is finished as ``cancelled`` instead, with every segment's final state."""
        job, handler, run = self._job, self._handler, self._run
        assert job is not None
        if handler is not None and run is not None:
            if not handler.save(host, run):
                current = host.store.get_job(job.job_id)
                if current is not None and current.status == "cancelling":
                    handler.cancel(host, run)
                    log.info("job %s cancelled as the daemon let go of it", job.job_id)
                    self._done(host)
                    return
            handler.release(run)
        return_job(host.store, job.job_id, reason=reason)
        self._done(host)

    def _done(self, host: RunnerHost) -> None:
        self._job = None
        self._handler = None
        self._run = None
        if self._registry is not None:
            self._registry.residency.reset_wait(host)
        host.job_finished()

    # ------------------------------------------------------------------ the drain estimate (DC-2)
    def _publish_drain(self, host: RunnerHost) -> None:
        try:
            queued = host.store.queued_jobs()
        except Exception:  # the estimate is advice: a store hiccup must not fail a step
            log.debug("the queue could not be read for the drain estimate", exc_info=True)
            return
        registry, handler, run = self._registry, self._handler, self._run
        held = {run.job_id: handler.remaining_audio_s(run)} if handler is not None and run is not None else None
        rate = registry.throughput.wall_per_audio_s if registry is not None else WALL_PER_AUDIO_S
        host.set_est_drain(est_drain_s(queued, wall_per_audio_s=rate, running_remaining_s=held) if queued else 0.0)


def _registry(source: JobEngine | Registry) -> Registry:
    return Registry.of(source) if isinstance(source, JobEngine) else source


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


def installed_engine(host: RunnerHost) -> JobEngine:
    """The engine as this installation provides it: built from the daemon's config, with the service's text
    pipeline, post-processing and QA, the cue aligner and the QA group's pinned models.

    The cue aligner (WP15) and the QA models' pins (WP22, installed by ``narration-admin``) are not part of
    this build yet, so no take could be scored: this raises ``BACKEND_NOT_INSTALLED``, and each job the
    daemon claims fails with it, rather than rendering takes it cannot check. Once they are, this assembles
    them with ``build_runner``'s parts.
    """
    raise NarrationError(
        codes.BACKEND_NOT_INSTALLED,
        "this installation cannot score takes: the cue aligner and the QA models are not installed",
        hint="Run narration-admin doctor to see what is missing; nothing was rendered.",
        retryable=False,
    )


def default_runner() -> EngineRunner:
    """The runner the daemon loads by name (``narration.jobs.runner:default_runner``). The engine is built at
    the first job, by ``installed_engine``."""
    return EngineRunner(installed_engine)


__all__ = ["PREEMPTING", "EngineFactory", "EngineRunner", "build_runner", "default_runner", "installed_engine"]
