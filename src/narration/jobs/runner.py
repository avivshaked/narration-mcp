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

**Scheduling across jobs** (section 4 item 3). Jobs are claimed by priority (``interactive`` before
``batch``), then first come (the store's insertion order). Affinity means the resident model group is kept
across a job boundary (``gpu.Residency``, shared by every handler), so a job whose first work needs that
group starts without a load. Affinity never reorders the queue. The engine runs one job at a time: work
from several jobs is not grouped within a round (the lead's decision, pending the owner). Every generation
job uses the current Base engine profile, so jobs need no grouping by profile until design jobs run here
too.

A job given back (a stop, or giving way to an interactive job) and taken again is planned from the cache
and resumes the round it was in; its record never shows less progress than before. A job cancelled as the
daemon lets go of it is finished as ``cancelled``.

A job-level failure (``NarrationError``) fails the job with its error. An exception the engine does not
expect is a bug: the job fails with ``INTERNAL`` rather than being tried again at every step. Every
``INTERNAL`` error names the daemon's log (``details.log``, section 14).

``has_work`` answers whether a ``step`` would find work (a job held, or one queued) and claims nothing; the
daemon asks it before an idle exit. ``default_runner`` is the zero-argument factory the daemon loads by
name; it builds the engine at the first job, from the daemon's config.

``shutdown`` gives back the job held (``return_job``: ``running`` goes back to ``queued``, ``cancelling``
becomes ``cancelled``), removes its scratch files, and closes the handlers (the engine stops the thread that
renews its leases). Leases are renewed while their work runs (one thread per engine, ``leases.LeaseKeeper``),
released at the end of each piece of work, and lapse by their TTL if the process dies, so none is held
between steps.

The drain estimate the daemon publishes (DC-2's ``admission.queue.est_drain_s``) is refreshed after each step.

When a job it held ends (``completed``, ``failed`` or ``cancelled``), the runner logs one INFO line: the job id,
its kind, its final status and outcome, the number of segments, the retakes used, and the wall time since it
took the job (the last time, if the job gave way). Nothing of the request is logged: no text, no transcript,
no path.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from narration.config import Config
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import AlignerCore
from narration.contracts.models import JobRecord
from narration.contracts.names import TERMINAL_JOB_STATUSES
from narration.daemon.settings import LOG_NAME
from narration.post import DeliveryPipeline
from narration.qa import Scorer
from narration.store.layout import LOGS
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
    completed meanwhile is picked up. ``clock`` (monotonic seconds) times each job for the line logged when it
    ends.
    """

    def __init__(
        self, source: JobEngine | Registry | EngineFactory, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._clock = clock
        self._taken_at = 0.0
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
        """Give back the job held, if any, and remove its scratch files; then close the handlers
        (``Registry.close``)."""
        try:
            if self._job is not None:
                self._let_go(host, f"the daemon stopped ({reason})")
        finally:
            if self._registry is not None:
                self._registry.close()

    # ------------------------------------------------------------------ claiming
    def _claim(self, host: RunnerHost) -> bool:
        job = host.store.claim_next_job(host.holder)
        if job is None:
            return False
        self._job = job
        self._taken_at = self._clock()
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
        error = _with_log(error, log_path(host.config))
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
        job = self._job
        self._job = None
        self._handler = None
        self._run = None
        if self._registry is not None:
            self._registry.residency.reset_wait(host)
        host.job_finished()
        if job is not None:
            self._log_end(host, job.job_id)

    def _log_end(self, host: RunnerHost, job_id: str) -> None:
        """One INFO line when the job has ended (see the module docstring); none when it went back to the queue.
        Only the record's counts and names are logged, never the request (its text, transcript or paths)."""
        wall_s = self._clock() - self._taken_at
        try:
            job = host.store.get_job(job_id)
        except Exception:  # the line is advice: a store hiccup must not fail a step
            log.debug("job %s: its record could not be read for the end-of-job line", job_id, exc_info=True)
            return
        if job is None or job.status not in TERMINAL_JOB_STATUSES:
            return
        log.info(
            "job %s ended: kind %s, status %s, outcome %s, segments %d, retakes used %d, wall %.1f s",
            job.job_id,
            job.kind,
            job.status,
            job.outcome or "none",
            job.progress.segments_total,
            sum(segment.retakes_used for segment in job.items),
            wall_s,
        )

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
    """The ``INTERNAL`` error for an exception the engine did not expect: a bug. The message names the
    exception's type only, since its text may carry a caller's words or a local path; the text goes to the
    log, whose path ``_with_log`` adds (section 14)."""
    return NarrationError(
        codes.INTERNAL,
        f"the job engine failed ({type(exc).__name__})",
        retryable=False,
        hint="This is a bug in the service; report it with the log in 'details'. Nothing was published half made.",
        details={"exception": type(exc).__name__},
    )


def log_path(config: Config) -> Path:
    """The daemon's log, which an ``INTERNAL`` error names (section 14): ``<store_root>/logs/daemon.log``, the
    file the daemon writes (``narration.daemon.settings.LOG_NAME``)."""
    return config.server.store_root / LOGS / LOG_NAME


def _with_log(error: NarrationError, path: Path) -> NarrationError:
    """An ``INTERNAL`` error with the log path in ``details.log``, as section 14 promises; others as they are."""
    if error.code != codes.INTERNAL or (error.details or {}).get("log"):
        return error
    return NarrationError(
        error.code,
        error.message,
        field=error.field,
        hint=error.hint,
        details={**(error.details or {}), "log": str(path)},
        retryable=error.retryable,
        retry_after_s=error.retry_after_s,
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
    ``probe`` defaults to NVML on a ``cuda`` device and to no check otherwise; ``check_path`` to the daemon's
    platform check (``host.platform.check_readable_path``, section 17.3)."""
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


def installed_engine(host: RunnerHost) -> Registry:
    """The job engine as this installation provides it (WP32, ``narration.engine.installed``): built from the
    daemon's config with the service's text pipeline, post-processing and QA, the cue aligner (WP15) and the
    QA group's pinned models, the engine guard (the fingerprint check and the canary gate), NVML's free-VRAM
    probe and the daemon platform's path check. It returns the registry of every job kind it runs.

    A model snapshot that is not installed raises ``BACKEND_NOT_INSTALLED``: the runner fails that job with it,
    and builds again at the next job, so an installation completed meanwhile is picked up.
    """
    from narration.engine import installed  # here: narration.engine builds on narration.jobs

    return installed.registry(host)


def default_runner() -> EngineRunner:
    """The runner the daemon loads by name (``narration.jobs.runner:default_runner``). The engine is built at
    the first job, by ``installed_engine``."""
    return EngineRunner(installed_engine)


__all__ = [
    "PREEMPTING",
    "EngineFactory",
    "EngineRunner",
    "build_runner",
    "default_runner",
    "installed_engine",
    "log_path",
]
