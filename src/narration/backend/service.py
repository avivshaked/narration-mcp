"""The real ``Backend`` behind the MCP front-end (design sections 4, 7, 12, 14, 15, 17; plan.md WP36).

``NarrationBackend`` implements ``narration.contracts.interfaces.Backend``. It is the front-end's half of the
service: stateless, one per client, it never touches the GPU and never writes audio (section 4). It checks a
request (the text pipeline, the voice clip, the engine, the voice's measurement, the limits), plans it over
the cache, commits a job row to the store, and starts the daemon, which does the work (``narration.jobs``).
It reads jobs, results and status back from the store. There are no sockets: the store is the channel.

- Every store call runs in a worker thread (``anyio.to_thread``); the store is safe to share across threads.
  A tool that writes finishes its work well inside the front-end's write deadline
  (``narration.mcp.server.WRITE_DEADLINE_S``): store rows, a clip copy of at most 20 MB, a detached start.
- ``get_job``'s long-poll reads the job again every ``poll_s`` until its status changes or ``wait_s`` ends,
  sending a progress notification whenever its progress moves. Cancelling the request ends the wait,
  never the job.
- Errors are ``NarrationError`` with the section 14 code, the field and a hint; every retryable one carries
  ``retry_after_s`` (DC-2). The service suggests; the caller decides.

The tools of the DESIGN step (``design_voice``, ``profile_voice``) and ``audition_pronunciation`` check a
request and queue its job only when the daemon runs that kind (``RUNNABLE_KINDS``). Until their handlers
exist (WP34, WP35), they answer ``BACKEND_NOT_INSTALLED``.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import secrets
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, ParamSpec, TypeVar

import anyio
import anyio.to_thread

import narration
from narration import keys
from narration.config import Config
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Platform, ProgressCallback, ResourceContent, Store, TextPlanner
from narration.contracts.models import (
    EngineProfile,
    EngineRef,
    Flag,
    JobRecord,
    MeasurementRecord,
    Progress,
    SegmentText,
)
from narration.contracts.names import TERMINAL_JOB_STATUSES, EngineKind, JobKind, JobStatus, Priority
from narration.contracts.serial import to_json
from narration.jobs import admission
from narration.jobs.plan import VoiceSpec, estimated_audio_s
from narration.lint import NegationLinter
from narration.post import delivery_tools
from narration.qa import Scorer
from narration.store.store import utc_iso
from narration.text import TextPipeline, segment_too_long

from . import views
from .assemble import assemble, consistency_of, measured_error
from .clips import ClipRef, admit_clip, check_synthetic, refield
from .launch import DAEMON_RETRY_S, DaemonLauncher
from .measures import Measurements, StoreMeasurements
from .planning import AnalysisPins, plan_request
from .requests import (
    check_controls,
    check_limits,
    hints_of,
    parse_submit,
    request_sha256,
    segments_of,
    voice_of,
)
from .steps import audition_request, design_identity, design_request

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

POLL_S: Final = 0.5
"""How often ``get_job``'s long-poll reads the job again."""
RELEASE_WAIT_S: Final = 10.0
"""How long ``release_gpu`` waits for the daemon to answer its command (well inside the write deadline)."""
GIB: Final = 1024**3
REPORT_MD: Final = "report.md"
REPORT_JSON: Final = "report.json"
MEASUREMENT_JSON: Final = "measurement.json"
GENERATION_KINDS: Final = ("generate", "analyse")
RUNNABLE_KINDS: Final[frozenset[JobKind]] = frozenset({"generate", "analyse", "measure"})
"""The job kinds this build's daemon runs. The front-end queues no other: a job of a kind with no handler
would fail in the daemon, so its tool answers ``BACKEND_NOT_INSTALLED`` at once instead. ``design`` and
``profile`` join with WP34's handlers, ``pronunciation`` with WP35's."""
KIND_OF_TOOL: Final[dict[str, JobKind]] = {
    "design_voice": "design",
    "profile_voice": "profile",
    "audition_pronunciation": "pronunciation",
}
HANDLER_WP: Final[dict[str, str]] = {"design": "WP34", "profile": "WP34", "pronunciation": "WP35"}


class NarrationBackend:
    """The ``Backend`` of the MCP front-end (see the module docstring).

    ``config``, ``store`` and ``platform`` are the service's; ``launcher`` starts the daemon and reads its
    status (``launch.DetachedLauncher``). ``pins`` gives the QA models' and the aligner's pins when the
    installation has them (``planning.AnalysisPins``): with them, a plan also looks the analysis layer up.
    ``measurements`` answers whether a voice is measured (``measures.StoreMeasurements``: WP33's rules).
    ``kinds`` are the job kinds the daemon runs (``RUNNABLE_KINDS``); a tool whose kind is not among them
    answers ``BACKEND_NOT_INSTALLED``.
    ``clock`` is Unix seconds (the rate cap, job times); ``poll_s`` is the long-poll's interval.
    """

    def __init__(
        self,
        config: Config,
        store: Store,
        platform: Platform,
        *,
        launcher: DaemonLauncher,
        pins: Callable[[], AnalysisPins | None] | None = None,
        measurements: Measurements | None = None,
        kinds: Collection[JobKind] = RUNNABLE_KINDS,
        text: TextPlanner | None = None,
        scorer: Scorer | None = None,
        clock: Callable[[], float] = time.time,
        poll_s: float = POLL_S,
    ) -> None:
        self.config = config
        self.store = store
        self.platform = platform
        self.launcher = launcher
        self._pins = pins if pins is not None else (lambda: None)
        self.measurements: Measurements = measurements if measurements is not None else StoreMeasurements(store, config)
        self.kinds = frozenset(kinds)
        self.linter = NegationLinter()
        self.text: TextPlanner = text if text is not None else TextPipeline(config.text)
        self.scorer = scorer if scorer is not None else Scorer(config.measurement)
        self.clock = clock
        self.poll_s = poll_s
        self.keys = keys.Keys()
        self.tools = delivery_tools()

    # ================================================================ plumbing
    @staticmethod
    async def _thread(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Run a blocking call (the store's) in a worker thread; it is not cut off by a cancellation. A
        retryable error raised below without ``retry_after_s`` is given its code's (DC-2)."""
        try:
            return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))
        except NarrationError as exc:
            with_wait = with_retry_after(exc)
            if with_wait is exc:
                raise
            raise with_wait from exc

    def _created_since(self, unix_s: float) -> int:
        """Jobs created since a time (Unix seconds), across every front-end: the submit rate cap's count."""
        return self.store.jobs_created_since(utc_iso(unix_s))

    def _now_iso(self) -> str:
        return utc_iso(self.clock())

    def _job(self, job_id: str) -> JobRecord:
        job = self.store.get_job(job_id)
        if job is None:
            raise NarrationError(
                codes.NOT_FOUND,
                f"there is no job {job_id}",
                field="job_id",
                details={"job_id": job_id},
            )
        return job

    def _voice_hash(self, voice: VoiceSpec) -> str:
        return self.measurements.voice_hash(voice)

    def _base_profile(self) -> EngineProfile:
        return self._engine_profile("base")

    def _engine_profile(self, kind: EngineKind) -> EngineProfile:
        """The engine profile pinned for ``base`` (cloning) or ``design`` work; ``BACKEND_NOT_INSTALLED``
        before ``narration-admin engine pin``."""
        profile = self.store.current_engine_profile(kind)
        if profile is None:
            what = "cloning voices (Qwen Base)" if kind == "base" else "designing voices (Qwen VoiceDesign)"
            raise NarrationError(
                codes.BACKEND_NOT_INSTALLED,
                f"no engine profile is pinned for {what}",
                hint="Ask the operator to run narration-admin install, then narration-admin engine pin.",
                details={"engine": kind},
            )
        return profile

    def _measurement(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord:
        """The measurement the request is judged against; ``VOICE_NOT_MEASURED`` when there is none."""
        return self.measurements.require(voice_hash, profile)

    def _check_disk(self) -> None:
        """``STORE_FULL`` (retryable) when the store's disk has less free space than ``min_free_disk_gb``."""
        minimum = self.config.limits.min_free_disk_gb * GIB
        free = self.platform.free_disk_bytes(self.store.root)
        if free < minimum:
            raise NarrationError(
                codes.STORE_FULL,
                f"the store's disk has {free // GIB} GB free; the service needs at least "
                f"{self.config.limits.min_free_disk_gb} GB",
                retry_after_s=admission.STORE_FULL_RETRY_S,
                details={"free_bytes": free, "min_free_bytes": minimum},
            )

    # ================================================================ jobs: admission and the queue (DC-2)
    @staticmethod
    def _identical(queued: Sequence[JobRecord], kind: JobKind, sha: str) -> JobRecord | None:
        """The queued or running job of the same request (the store's ``create_job`` rule)."""
        return next(
            (j for j in queued if j.kind == kind and j.request_sha256 == sha and j.status in ("queued", "running")),
            None,
        )

    def _check_admission(self, queued: Sequence[JobRecord], daemon_drain: float | None) -> None:
        """``RATE_LIMITED`` and ``QUEUE_FULL`` (DC-2), each with ``retry_after_s``."""
        limits = self.config.limits
        window = views.rate(self._created_since, now=self.clock(), max_per_min=limits.max_submits_per_min)
        if window.remaining <= 0:
            raise NarrationError(
                codes.RATE_LIMITED,
                f"more than {limits.max_submits_per_min} jobs were submitted in the last minute",
                retry_after_s=window.resets_in_s,
                details={"max_submits_per_min": limits.max_submits_per_min, "resets_in_s": window.resets_in_s},
            )
        waiting = sum(1 for j in queued if j.status == "queued")
        if waiting >= limits.max_queued_jobs:
            drain = daemon_drain if daemon_drain is not None else admission.est_drain_s(queued)
            retry = admission.queue_full_retry_after_s(drain, waiting)
            raise NarrationError(
                codes.QUEUE_FULL,
                f"{waiting} jobs are waiting; the queue takes at most {limits.max_queued_jobs}",
                retry_after_s=retry,
                details={"length": waiting, "max": limits.max_queued_jobs, "est_drain_s": drain},
            )

    def _new_record(
        self,
        kind: JobKind,
        stored: dict[str, Any],
        sha: str,
        *,
        label: str | None,
        priority: Priority,
        idempotency_key: str | None,
        status: JobStatus = "queued",
        result: dict[str, Any] | None = None,
        segments_total: int = 0,
    ) -> JobRecord:
        now = self._now_iso()
        done = status in TERMINAL_JOB_STATUSES
        return JobRecord(
            job_id=self.keys.new_job_id(),
            kind=kind,
            request=stored,
            request_sha256=sha,
            label=label,
            priority=priority,
            status=status,
            phase=None,
            round=0,
            progress=Progress(
                done_s=0.0,
                total_s=0.0,
                fraction=1.0 if done else 0.0,
                segments_done=0,
                segments_total=segments_total,
            ),
            outcome=None,
            error=None,
            idempotency_key=idempotency_key,
            created_at=now,
            updated_at=now,
            result=result,
            message="answered from the cache" if done else "queued",
        )

    def _enqueue(
        self,
        kind: JobKind,
        stored: dict[str, Any],
        *,
        label: str | None,
        priority: Priority,
        idempotency_key: str | None,
        segments_total: int = 0,
        result: dict[str, Any] | None = None,
        identity: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        """Queue a job, or return the queued or running job of the same request (section 7.3), which is not
        held to the rate cap or the queue's length; then make sure a daemon serves the store, after the job is
        committed (``launch``). A job being cancelled is ending, so the same request again is a new job.
        ``identity`` is what the request's identity hashes, when it is not the whole stored request."""
        sha = request_sha256(kind, identity if identity is not None else stored)
        queued = self.store.queued_jobs()
        job = self._identical(queued, kind, sha)
        if job is None:
            daemon = self.launcher.running(self.store)
            self._check_admission(queued, daemon.est_drain_s if daemon is not None else None)
            record = self._new_record(
                kind,
                stored,
                sha,
                label=label,
                priority=priority,
                idempotency_key=idempotency_key,
                segments_total=segments_total,
                result=result,
            )
            try:
                job, _ = self.store.create_job(record)
            except NarrationError as exc:
                if exc.field == "idempotency_key":  # the argument's own path (DC-6)
                    raise refield(exc, "options.idempotency_key") from exc
                raise
        if job.status not in TERMINAL_JOB_STATUSES:
            self._ensure_daemon(job)
        return job

    def _ensure_daemon(self, job: JobRecord) -> None:
        try:
            self.launcher.ensure(self.store)
        except NarrationError as exc:
            if exc.code != codes.DAEMON_UNAVAILABLE:
                raise
            raise NarrationError(
                exc.code,
                f"job {job.job_id} is queued, but no daemon could be started to run it: {exc.message}",
                hint=exc.hint,
                details={**(exc.details or {}), "job_id": job.job_id},
                retryable=exc.retryable,
                retry_after_s=exc.retry_after_s if exc.retryable else None,
            ) from exc

    # ================================================================ text (sections 3.2, 7.2, 9.1)
    def _text_json(self, text: SegmentText, measurement: MeasurementRecord | None) -> dict[str, Any]:
        """One segment's text echo and length check (``check_text``, ``submit_job``'s dry run; R7, R8)."""
        out = to_json(text)
        warnings = list(text.warnings)
        limit: int | None = None
        over: int | None = None
        est: float | None = None
        if measurement is not None:
            limit = measurement.max_segment_chars
            if limit is not None:
                over = max(0, text.spoken_chars - limit)
                too_long = segment_too_long(
                    text, max_segment_chars=limit, max_segment_seconds=measurement.max_segment_seconds
                )
                if too_long is not None:
                    warnings.append(too_long)
            est = round(estimated_audio_s(text, measurement), 2)
        out.update(
            max_segment_chars=limit, over_by_chars=over, est_duration_s=est, warnings=[to_json(w) for w in warnings]
        )
        return out

    @staticmethod
    def _warnings(texts: Sequence[SegmentText], measurement: MeasurementRecord) -> list[Flag]:
        """``submit_job``'s ``warnings``: every text warning, and ``SEGMENT_TOO_LONG`` (never a refusal)."""
        out: list[Flag] = []
        for text in texts:
            for cue in text.cues:
                out.extend(cue.warnings)
            out.extend(text.warnings)
            if measurement.max_segment_chars is not None:
                too_long = segment_too_long(
                    text,
                    max_segment_chars=measurement.max_segment_chars,
                    max_segment_seconds=measurement.max_segment_seconds,
                )
                if too_long is not None:
                    out.append(too_long)
        return out

    @staticmethod
    def _measurement_summary(measurement: MeasurementRecord) -> dict[str, Any]:
        return {
            "max_segment_chars": measurement.max_segment_chars,
            "max_segment_seconds": measurement.max_segment_seconds,
            "measured_at": measurement.measured_at,
        }

    # ================================================================ submit_job (sections 7.3, 10.2, 12)
    def submit_job_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``submit_job``: check, plan over the cache, then queue (or plan only, with ``dry_run``)."""
        config = self.config
        request = parse_submit(args, config.defaults)
        check_controls(request.segments)
        check_limits(request.segments, request.hints, config.limits)
        texts = self.text.plan_request(request.segments, request.hints, strict_text=request.options.strict_text)
        profile = self._base_profile()
        if request.expect_engine_profile is not None and request.expect_engine_profile != profile.hash:
            raise NarrationError(
                codes.ENGINE_CHANGED,
                f"the request expects engine profile {request.expect_engine_profile}; the service's is {profile.hash}",
                field="expect_engine_profile",
                details={"expected": request.expect_engine_profile, "current": profile.hash},
            )
        clip = ClipRef(path=request.voice.path, sha256=request.voice.sha256)
        check_synthetic(self.store, config.voices.allow_sha256, clip, field="voice")
        dry_run = request.options.dry_run
        if not dry_run:
            self._check_disk()
        admit_clip(self.store, self.platform, clip, max_seconds=config.limits.max_clip_seconds, keep=not dry_run)
        voice_hash = self._voice_hash(request.voice)
        measurement = self._measurement(voice_hash, profile)
        plan = plan_request(
            self.store,
            segments=request.segments,
            texts=texts,
            hints=request.hints,
            takes=request.options.takes,
            voice_hash=voice_hash,
            profile=profile,
            measurement=measurement,
            delivery=config.delivery,
            tools=self.tools,
            pins=self._pins(),
        )
        engine = {"id": profile.engine_profile_id, "hash": profile.hash}
        warnings = [to_json(w) for w in self._warnings(texts, measurement)]
        if dry_run:
            queued = self.store.queued_jobs()
            daemon = self.launcher.running(self.store)
            wait = views.drain_s(queued, daemon) or 0.0
            return {
                "job_id": None,
                "status": "planned",
                "voice_hash": voice_hash,
                "engine_profile": engine,
                "plan": plan.as_json(queue_position=None, wait_s=wait),
                "poll_after_s": 0.0,
                "text": [self._text_json(t, measurement) for t in texts],
                "warnings": warnings,
            }
        job = self._enqueue(
            "generate",
            request.stored,
            label=request.label,
            priority=request.options.priority,
            idempotency_key=request.options.idempotency_key,
            segments_total=len(texts),
        )
        queued = self.store.queued_jobs()
        _, position, poll = views.job_timing(job, queued)
        wait = views.wait_to_start_s(job.job_id, queued) if job.status == "queued" else 0.0
        status = job.status if job.status in ("queued", "running", "completed") else "running"
        return {
            "job_id": job.job_id,
            "status": status,
            "voice_hash": voice_hash,
            "engine_profile": engine,
            "plan": plan.as_json(queue_position=position, wait_s=wait),
            "poll_after_s": poll,
            "warnings": warnings,
        }

    async def submit_job(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``submit_job`` (section 7.3)."""
        return await self._thread(self.submit_job_sync, args)

    # ================================================================ get_job (section 7.4)
    async def get_job(self, args: Mapping[str, Any], progress: ProgressCallback | None) -> dict[str, Any]:
        """``get_job``: the job now, or after a long-poll of up to ``wait_s`` that ends when its status
        changes; progress notifications while it waits."""
        job_id = str(args["job_id"])
        wait_s = float(args.get("wait_s", 0) or 0)
        include = bool(args.get("include_segments", False))
        job = await self._thread(self._job, job_id)
        if progress is not None and job.status not in TERMINAL_JOB_STATUSES:
            await _report(progress, job)
        deadline = anyio.current_time() + wait_s
        while job.status not in TERMINAL_JOB_STATUSES:
            remaining = deadline - anyio.current_time()
            if remaining <= 0:
                break
            await anyio.sleep(min(self.poll_s, remaining))
            again = await self._thread(self._job, job_id)
            moved = again.progress != job.progress or again.message != job.message
            status_changed = again.status != job.status
            job = again
            if progress is not None and moved:
                await _report(progress, job)
            if status_changed:
                break
        queued = await self._thread(self.store.queued_jobs)
        return views.job_json(job, queued, include_segments=include)

    # ================================================================ get_results (section 7.5)
    def get_results_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``get_results`` for any job kind this build runs."""
        job = self._job(str(args["job_id"]))
        include_words = bool(args.get("include_words", True))
        include_transcripts = bool(args.get("include_transcripts", False))
        block: dict[str, Any] = {
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "outcome": job.outcome,
            "label": job.label,
            "error": to_json(job.error) if job.error is not None else None,
        }
        if job.kind in GENERATION_KINDS:
            return self._generation_results(job, block, include_words, include_transcripts)
        if job.kind == "measure":
            return self._measure_results(job, block)
        return {"job": block, **self._step_results(job)}

    async def get_results(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``get_results`` (section 7.5)."""
        return await self._thread(self.get_results_sync, args)

    def _generation_results(
        self, job: JobRecord, block: dict[str, Any], include_words: bool, include_transcripts: bool
    ) -> dict[str, Any]:
        full = self._generation_view(job, block, include_words=True, include_transcripts=False)
        if job.status in TERMINAL_JOB_STATUSES:
            report = self._write_report(job.job_id, full)
            if report is not None:
                full["report_md"] = str(report)
        if include_words and not include_transcripts:
            return full
        shown = self._generation_view(job, block, include_words=include_words, include_transcripts=include_transcripts)
        if "report_md" in full:
            shown["report_md"] = full["report_md"]
        return shown

    def _generation_view(
        self, job: JobRecord, block: dict[str, Any], *, include_words: bool, include_transcripts: bool
    ) -> dict[str, Any]:
        assembled = assemble(
            self.store, self.text, job, include_words=include_words, include_transcripts=include_transcripts
        )
        voice = voice_of(job.request)
        voice_hash = self._voice_hash(voice)
        engine = assembled.engine
        if engine is None:
            current = self.store.current_engine_profile("base")
            engine = EngineRef(id=current.engine_profile_id, hash=current.hash) if current is not None else None
        out: dict[str, Any] = {
            "job": block,
            "voice": {"voice_hash": voice_hash, "clip_sha256": voice.sha256},
        }
        if engine is not None:
            out["engine_profile"] = to_json(engine)
            measurement = self.store.get_measurement(voice_hash, engine.id)
            if measurement is not None:
                out["measurement"] = self._measurement_summary(measurement)
        out["segments"] = [to_json(s) for s in assembled.segments]
        out["consistency"] = to_json(consistency_of(job))
        out["listen_first"] = [to_json(i) for i in self.scorer.listen_first(assembled.segments)]
        out["licence"] = to_json(assembled.licence)
        return out

    def _write_report(self, job_id: str, results: Mapping[str, Any]) -> Path | None:
        """``jobs/<job_id>/report.md`` and ``report.json`` (section 15), from the assembled results; the path of
        ``report.md``, or None when it could not be written."""
        folder = self.store.job_dir(job_id)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            _publish_text(folder / REPORT_JSON, json.dumps(self.scorer.report_json(results), ensure_ascii=False))
            path = folder / REPORT_MD
            _publish_text(path, self.scorer.report_md(results))
        except OSError:
            log.exception("the report of job %s could not be written", job_id)
            return (folder / REPORT_MD) if (folder / REPORT_MD).is_file() else None
        return path

    def _measure_results(self, job: JobRecord, block: dict[str, Any]) -> dict[str, Any]:
        """A ``measure`` job's results: the full measurement, and the path of its JSON file to keep."""
        voice = voice_of(job.request)
        result = job.result or {}
        voice_hash = str(result.get("voice_hash") or self._voice_hash(voice))
        engine = result.get("engine_profile")
        profile_id = engine.get("id") if isinstance(engine, dict) else None  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(profile_id, str):
            current = self.store.current_engine_profile("base")
            profile_id = current.engine_profile_id if current is not None else None
        out: dict[str, Any] = {"job": block, "voice": {"voice_hash": voice_hash, "clip_sha256": voice.sha256}}
        if profile_id is None:
            return out
        measurement = self.store.get_measurement(voice_hash, profile_id)
        if measurement is not None:
            out["engine_profile"] = to_json(measurement.engine_profile)
            out["measurement"] = self._measurement_summary(measurement)
            path = self.store.measurement_dir(voice_hash, profile_id) / MEASUREMENT_JSON
            out["measurement_result"] = {"path": str(result.get("path") or path), "measurement": to_json(measurement)}
        return out

    def _step_results(self, job: JobRecord) -> dict[str, Any]:
        """The results of a DESIGN-step job or an audition, from what its handler (WP34, WP35) left:
        ``design``, the candidates the store holds under the job's ``design_id``; ``profile``, the profile
        its result names (``audio_sha256`` and ``profile_version``); ``audition``, its result's
        ``audition``. Nothing while the job has left none."""
        result = job.result or {}
        if job.kind == "design":
            design_id = str(job.request.get("design_id", ""))
            if not design_id:
                return {}
            candidates = self.store.get_design(design_id)
            return {"design": {"design_id": design_id, "candidates": [to_json(c) for c in candidates]}}
        if job.kind == "profile":
            sha, version = result.get("audio_sha256"), result.get("profile_version")
            if isinstance(sha, str) and isinstance(version, str):
                profile = self.store.get_profile(sha, version)
                if profile is not None:
                    return {"profile": to_json(profile)}
            return {}
        audition = result.get("audition")
        return {"audition": audition} if job.kind == "pronunciation" and isinstance(audition, dict) else {}

    # ================================================================ cancel_job (sections 7.6, 8)
    def cancel_job_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``cancel_job``: a queued job is cancelled at once; a running one is ``cancelling`` until the daemon
        stops it between two pieces of work. Finished renders, takes and analyses stay in the cache."""
        job_id = str(args["job_id"])
        reason = args.get("reason")
        message = f"cancelled: {reason}" if reason else "cancelled"
        for _ in range(5):
            job = self._job(job_id)
            if job.status in TERMINAL_JOB_STATUSES:
                raise NarrationError(
                    codes.JOB_NOT_CANCELLABLE,
                    f"job {job_id} is already {job.status}",
                    field="job_id",
                    details={"status": job.status},
                )
            if job.status == "cancelling":
                return {"status": "cancelling", "completed": False}
            if job.status == "queued":
                if self.store.update_job(job_id, expect_status="queued", status="cancelled", message=message):
                    return {"status": "cancelled", "completed": True}
            elif self.store.update_job(job_id, expect_status="running", status="cancelling", message=message):
                return {"status": "cancelling", "completed": False}
        job = self._job(job_id)
        return {"status": job.status, "completed": job.status == "cancelled"}

    async def cancel_job(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``cancel_job`` (section 7.6)."""
        return await self._thread(self.cancel_job_sync, args)

    # ================================================================ check_text (section 7.6; R6-R8)
    def check_text_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``check_text``: the text echo and length check of every segment; it renders nothing and reads no
        file. With a voice, its measurement (by the sha256 and transcript sent) gives the length check."""
        segments = segments_of(args)
        hints = hints_of(args)
        check_limits(segments, hints, self.config.limits)
        texts = self.text.plan_request(segments, hints, strict_text=False)
        voice_hash: str | None = None
        measurement: MeasurementRecord | None = None
        if args.get("voice"):
            voice_hash = self._voice_hash(voice_of(args))
            profile = self.store.current_engine_profile("base")
            if profile is not None:
                measurement = self.store.get_measurement(voice_hash, profile.engine_profile_id)
        return {
            "voice_hash": voice_hash,
            "measurement": self._measurement_summary(measurement) if measurement is not None else None,
            "segments": [self._text_json(t, measurement) for t in texts],
            "text_checks_version": self.text.checks_info.version,
        }

    async def check_text(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``check_text`` (section 7.6)."""
        return await self._thread(self.check_text_sync, args)

    # ================================================================ measure_voice (sections 3.2, 7.6)
    def measure_voice_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``measure_voice``: the cached measurement at once (with the id of a completed job), or a queued
        ``measure`` job. The clip is checked and copied into the store first (section 17.3)."""
        voice = voice_of(args)
        clip = ClipRef(path=voice.path, sha256=voice.sha256)
        profile = self._base_profile()
        check_synthetic(self.store, self.config.voices.allow_sha256, clip, field="voice")
        self._check_disk()
        admit_clip(self.store, self.platform, clip, max_seconds=self.config.limits.max_clip_seconds, keep=True)
        voice_hash = self._voice_hash(voice)
        stored = {"voice": dict(args["voice"])}
        existing = self.measurements.current(voice_hash, profile)
        if existing is not None:
            self.store.touch("measurement", existing.measurement_key)
            handle = measurement_handle(existing, self.store.measurement_dir(voice_hash, profile.engine_profile_id))
            record = self._new_record(
                "measure",
                stored,
                request_sha256("measure", stored),
                label=None,
                priority="batch",
                idempotency_key=None,
                status="completed",
                result=handle,
            )
            job, _ = self.store.create_job(record)
            return {
                "job_id": job.job_id,
                "status": job.status,
                "poll_after_s": 0.0,
                "voice_hash": voice_hash,
                "measurement": to_json(existing),
            }
        job = self._enqueue("measure", stored, label=None, priority="batch", idempotency_key=None)
        queued = self.store.queued_jobs()
        _, _, poll = views.job_timing(job, queued)
        return {"job_id": job.job_id, "status": job.status, "poll_after_s": poll, "voice_hash": voice_hash}

    async def measure_voice(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``measure_voice`` (section 7.6)."""
        return await self._thread(self.measure_voice_sync, args)

    # ================================================================ get_server_status (sections 4.1, 7.6; DC-2)
    def get_server_status_sync(self, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """``get_server_status``: the daemon, the GPU as it last saw it, the queue, the limits, the engine
        profiles, the capabilities, the measured alignment error, and DC-2's ``admission``."""
        config = self.config
        daemon = self.launcher.running(self.store)
        queued = self.store.queued_jobs()
        waiting = sum(1 for j in queued if j.status == "queued")
        window = views.rate(self._created_since, now=self.clock(), max_per_min=config.limits.max_submits_per_min)
        return {
            "version": narration.__version__,
            "spec_revision": names.SPEC_REVISION,
            "store_root": str(self.store.root),
            "daemon": views.daemon_json(daemon),
            "gpu": views.gpu_json(daemon),
            "cpu_threads": config.workers.cpu_threads,
            "queue": views.queue_json(queued),
            "admission": admission.admission(
                queue_length=waiting,
                max_queued=config.limits.max_queued_jobs,
                est_drain=views.drain_s(queued, daemon),
                rate=window,
                daemon=daemon,
            ),
            "engine_profiles": [self._profile_json(p) for p in self.store.list_engine_profiles()],
            "capabilities": {
                "controls": {"pace": False, "context": False, "instruct": False},
                "text_modes": ["spoken"],
            },
            "limits": to_json(config.limits),
            "text_checks_version": self.text.checks_info.version,
            "alignment": self._alignment_json(),
        }

    async def get_server_status(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``get_server_status`` (section 7.6)."""
        return await self._thread(self.get_server_status_sync, args)

    @staticmethod
    def _profile_json(profile: EngineProfile) -> dict[str, Any]:
        """An engine profile as the status lists it. ``installed``: its pinned snapshot is on disk; ``env_ok``:
        so is its worker project (the fingerprint check itself runs in the daemon, ``ENGINE_DRIFT``)."""
        installed = bool(profile.snapshot_dir) and Path(profile.snapshot_dir).is_dir()
        return {
            "id": profile.engine_profile_id,
            "hash": profile.hash,
            "installed": installed,
            "env_ok": installed and bool(profile.worker_project),
            "determinism_tier": profile.tier,
        }

    def _alignment_json(self) -> dict[str, Any]:
        """``get_server_status``'s ``alignment`` (R1, section 11.2): the method in use and its measured error,
        null until the alignment benchmark has been run."""
        pins = self._pins()
        method_id = pins.aligner_method_id if pins is not None else None
        bench = None
        if method_id is not None:
            bench = self.store.get_alignment_benchmark(method_id)
        else:
            bench = self.store.current_alignment_benchmark()
            method_id = bench.method_id if bench is not None else None
        error = measured_error(self.store, method_id) if method_id is not None else None
        return {
            "method_id": method_id,
            "model": bench.model if bench is not None else self.config.alignment.model,
            "revision": bench.revision if bench is not None else None,
            "measured_error": {
                "p50_s": error.p50_s if error is not None else None,
                "p95_s": error.p95_s if error is not None else None,
                "n": error.n if error is not None else None,
                "benchmark": error.benchmark if error is not None else None,
                "by_kind": to_json(error.by_kind) if error is not None and error.by_kind is not None else None,
            },
            "benchmark": to_json(bench.benchmark) if bench is not None else None,
            "measured_at": bench.measured_at if bench is not None else None,
        }

    # ================================================================ release_gpu (sections 4.1, 7.6)
    def release_gpu_sync(self, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """``release_gpu``: ask the daemon to unload an idle model now (a command through the store). While a
        job runs, it changes nothing and names the job. With no daemon, nothing is loaded."""
        daemon = self.launcher.running(self.store)
        if daemon is None:
            return {"released": False, "holder_before": None, "busy_job": None}
        command = self.store.post_command("release_gpu")
        done = self.store.wait_for_command(command.command_id, timeout_s=RELEASE_WAIT_S)
        result = done.result if done is not None and done.result is not None else None
        busy = daemon.current_job.job_id if daemon.current_job is not None else None
        if result is None:
            return {"released": False, "holder_before": daemon.gpu.holder, "busy_job": busy}
        holder = result.get("holder_before")
        busy_job = result.get("busy_job")
        return {
            "released": bool(result.get("released", False)),
            "holder_before": holder if holder in ("qwen", "qa") else None,
            "busy_job": busy_job if isinstance(busy_job, str) else None,
        }

    async def release_gpu(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``release_gpu`` (section 7.6)."""
        return await self._thread(self.release_gpu_sync, args)

    # ================================================================ the DESIGN step and auditions (3.1, 3.6, 7.6)
    def _require_kind(self, tool: str) -> None:
        """``BACKEND_NOT_INSTALLED`` for a tool whose job kind this build's daemon does not run."""
        kind = KIND_OF_TOOL[tool]
        if kind not in self.kinds:
            raise NarrationError(
                codes.BACKEND_NOT_INSTALLED,
                f"{tool} is not in this build of the service yet",
                hint="Update the service; the narration tools (measure_voice, check_text, submit_job) work meanwhile.",
                details={"tool": tool, "work_package": HANDLER_WP[kind]},
                retryable=False,
            )

    def _submitted(self, job: JobRecord) -> dict[str, Any]:
        """A step tool's answer: the job, its status and DC-2's ``poll_after_s``."""
        _, _, poll = views.job_timing(job, self.store.queued_jobs())
        return {"job_id": job.job_id, "status": job.status, "poll_after_s": poll}

    def design_voice_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``design_voice``: lint the description (warn, never refuse), check the design text, then queue a
        ``design`` job under a new ``design_id``; the same design while its job is active keeps its id."""
        self._require_kind("design_voice")
        self._engine_profile("design")
        stored = design_request(args, self.config, self.text)
        lint = self.linter.lint(stored["description"])
        self._check_disk()
        minted = self.keys.new_design_id()
        job = self._enqueue(
            "design",
            {**stored, "design_id": minted},
            identity=design_identity(stored),
            label=str(args["name"]),
            priority="batch",
            idempotency_key=None,
        )
        design_id = str(job.request.get("design_id", minted))
        return {**self._submitted(job), "design_id": design_id, "lint": to_json(lint)}

    async def design_voice(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``design_voice`` (sections 3.1, 3.5)."""
        return await self._thread(self.design_voice_sync, args)

    def profile_voice_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``profile_voice``: check the audio's path, sha256 and format (any WAV the owner can read, section
        17.3), then queue a ``profile`` job."""
        self._require_kind("profile_voice")
        audio = args["audio"]
        clip = ClipRef(path=str(audio["path"]), sha256=str(audio["sha256"]))
        self._check_disk()
        admit_clip(self.store, self.platform, clip, max_seconds=math.inf, keep=False, field="audio")
        job = self._enqueue("profile", {"audio": dict(audio)}, label=None, priority="batch", idempotency_key=None)
        return self._submitted(job)

    async def profile_voice(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``profile_voice`` (section 3.6)."""
        return await self._thread(self.profile_voice_sync, args)

    def audition_pronunciation_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``audition_pronunciation``: check the term, variants and carrier, and the voice as a clone's
        (section 17.4; it need not be measured), copy the clip into the store, then queue a
        ``pronunciation`` job."""
        self._require_kind("audition_pronunciation")
        stored = audition_request(args, self.text)
        self._base_profile()
        voice = voice_of(args)
        clip = ClipRef(path=voice.path, sha256=voice.sha256)
        check_synthetic(self.store, self.config.voices.allow_sha256, clip, field="voice")
        self._check_disk()
        admit_clip(self.store, self.platform, clip, max_seconds=self.config.limits.max_clip_seconds, keep=True)
        job = self._enqueue("pronunciation", stored, label=None, priority="batch", idempotency_key=None)
        return self._submitted(job)

    async def audition_pronunciation(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """``audition_pronunciation`` (section 7.6)."""
        return await self._thread(self.audition_pronunciation_sync, args)

    # ================================================================ resources (section 7.7)
    async def read_resource(self, uri: str) -> ResourceContent:
        """A ``narration://`` resource (section 7.7); ``NOT_FOUND`` for one that does not exist."""
        from .resources import read_resource

        return await self._thread(read_resource, self, uri)


RETRY_AFTER_S: Final[dict[str, float]] = {
    codes.DAEMON_UNAVAILABLE: DAEMON_RETRY_S,
    codes.GPU_UNAVAILABLE: admission.GPU_UNAVAILABLE_RETRY_S,
    codes.STORE_FULL: admission.STORE_FULL_RETRY_S,
    codes.QUEUE_FULL: admission.QUEUE_FULL_DEFAULT_S,
    codes.RATE_LIMITED: admission.RATE_WINDOW_S,
}
"""DC-2's wait for a retryable error that came without one (the backend sets its own; this covers the
errors of the layers below it)."""


def with_retry_after(exc: NarrationError) -> NarrationError:
    """``exc``, or a copy of it with ``retry_after_s`` when it is retryable and has none (DC-2: set on every
    retryable error)."""
    if not exc.retryable or exc.retry_after_s is not None:
        return exc
    return NarrationError(
        exc.code,
        exc.message,
        field=exc.field,
        hint=exc.hint,
        details=exc.details,
        retryable=True,
        retry_after_s=RETRY_AFTER_S.get(exc.code, admission.RETRY_MIN_S),
    )


def measurement_handle(measurement: MeasurementRecord, folder: Path) -> dict[str, Any]:
    """A ``measure`` job's ``JobRecord.result``, as the measure handler (WP33) writes it. For a measurement
    answered at once, ``ladder_stopped_at`` names the first rung that did not pass, without the reasons: the
    measurement does not keep them."""
    stopped = next(
        ({"paragraph_id": rung.paragraph_id, "chars": rung.chars} for rung in measurement.ladder if not rung.passes),
        None,
    )
    return {
        "voice_hash": measurement.voice_hash,
        "engine_profile": {"id": measurement.engine_profile.id, "hash": measurement.engine_profile.hash},
        "measurement_key": measurement.measurement_key,
        "path": str(folder / MEASUREMENT_JSON),
        "max_segment_chars": measurement.max_segment_chars,
        "max_segment_seconds": measurement.max_segment_seconds,
        "ladder_stopped_at": stopped,
    }


async def _report(progress: ProgressCallback, job: JobRecord) -> None:
    """One progress notification: done and total audio seconds (the total may grow with retakes)."""
    total = job.progress.total_s if job.progress.total_s > 0 else None
    await progress(job.progress.done_s, total, job.message)


def _publish_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` under a temporary name, then rename; nothing when it already says that."""
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == text:
                return
        except (OSError, UnicodeDecodeError):
            pass
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


__all__ = ["POLL_S", "RELEASE_WAIT_S", "NarrationBackend"]
