"""The ``measure`` job kind (design section 3.2; plan.md WP33): measuring a voice once, before generation.

``MeasureHandler`` is a ``narration.jobs.handlers.JobHandler``: the runner dispatches a claimed ``measure``
job to it, and it does the job a piece at a time, so the daemon can stop, cancel or preempt it between
pieces. It shares the job engine's parts and state (``JobEngine.core``: the model residency across every
kind, the throughput estimate, the canary outcome of the Qwen load in use) and reuses its stages: every
calibration and ladder take is rendered, post-processed and scored exactly as a generation take is, answered
by the cache first and made at most once under a lease (sections 4 and 10.2).

**The job, in order.**

1. **The transcript check** (QA group): Whisper transcribes the clip, and a transcript that does not match
   it refuses the job with ``REF_TEXT_MISMATCH`` before anything is rendered (``transcript``). The clip's
   speaker embedding is taken in the same load, for the anchor.
2. **The calibration set and the trend band** (Qwen, then the CPU, then QA): the calibration paragraphs (the
   corpus's design text and its three paragraphs) and the ladder rungs up to ``trend_band_max_chars``, each
   with ``seeds`` attempts (0 .. seeds-1), all rendered in one Qwen load. In the QA load that follows, the
   calibration takes are scored first, with no speaker or pace check; then the anchor and the similarity
   baseline are computed (``baseline``), and the band's takes are scored against them.
3. **The trend, ``tol`` and the speaking share** from the band, and each band rung judged (``ladder``). Pace
   is spoken characters per second of speaking time, as QA measures it (``narration.qa.pace``). The band is rendered
   together because the trend that judges its rungs needs all of them. A ladder with no rung in the band is
   refused when the job is planned (``BACKEND_NOT_INSTALLED``: there would be no trend to judge by).
4. **The rungs above the band**, one at a time from the shortest (a Qwen load, then a QA load each), for as
   long as every rung so far passes. The first failing rung ends the ladder: the rungs above it are never
   rendered.
5. **The measurement** (section 6): ``measurement.json`` is published under ``measurements/<voice_hash>/
   <engine_profile_id>/``, and the job completes with its handle in ``JobRecord.result``.

**What each take is scored against** (``MeasureStages``; the contracts' ``QaInputs`` and
``AnalysisKeyInputs``). A calibration take has no anchor to be judged against yet: it is scored with no
speaker or pace check, and its analysis names no measurement key, so no generation request can ever reuse
that analysis in place of one with the speaker check. A ladder take is scored against the calibration
anchor and similarity baseline (no pace: the pace curve is what the ladder measures), under the key of the
measurement being built, which is known before the ladder runs. The corpus's invented names are sent as
term-only hints, so ``wer_adj`` collapses them as section 11.1 intends (the engine text is unchanged).

**Retakes and failures.** A measurement take is data, never retaken (``max_retakes`` 0). A take-level
execution problem the engine could not overcome (its own retries included: out of memory, a worker that
crashed twice) fails the job, since a measurement short of a take would not be the measurement section 3.2
defines: ``GPU_UNAVAILABLE`` for memory, otherwise ``INTERNAL``, both retryable. A trend band with no
measured pace in any rung fails it the same way (``INTERNAL``), rather than publish a trend that fails every
rung. Everything made is kept in the cache, so the identical request resumes where it stopped.

**The clip** is read as a generation's is (``JobEngine.open``, section 17.3): through the engine's own path
check when it was built with one, else the daemon's platform check (``RunnerHost.platform``).

**A current measurement** (its key is the one this job would make) completes the job at once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from narration.contracts import codes, names
from narration.contracts.errors import (
    MaterialError,
    NarrationError,
    WorkerCrashed,
    WorkerFailure,
    WorkerTimeout,
)
from narration.contracts.models import (
    CalibrationTake,
    EngineProfile,
    EngineRef,
    JobRecord,
    MaterialParagraph,
    MaterialSet,
    MeasurementRecord,
    Pace,
    PaceTrend,
    SegmentIn,
    TranscriptCheck,
)
from narration.contracts.names import JobOutcome, SegmentState
from narration.jobs.admission import GPU_UNAVAILABLE_RETRY_S
from narration.jobs.core import LOST_STATE, worker_code
from narration.jobs.engine import SCRATCH_JOBS, JobEngine
from narration.jobs.failures import MAX_RETRIES, OOM_WAIT_S, Failures
from narration.jobs.handlers import Registry
from narration.jobs.host import RunnerHost
from narration.jobs.pins import ProfileError, call_cap, qwen_load_payload
from narration.jobs.plan import GenerateRequest, VoiceSpec, estimated_audio_s, hints_used
from narration.jobs.record import JobView
from narration.jobs.stages import QA_TIMEOUT_S, ScoringFacts, Stages
from narration.jobs.state import VERDICT_RANK, VERDICT_STATE, Attempt, JobRun, Outcome, SegmentWork
from narration.jobs.voice import require_synthetic, stage_clip
from narration.qa.normaliser import NumberReader
from narration.qa.profile import DEFAULT_PROFILE
from narration.store.layout import MEASUREMENT_JSON
from narration.store.store import utc_iso

from . import ladder as lad
from .baseline import Calibration, calibrate, similarities
from .corpus import corpus_version, load_corpus
from .lookup import measurement_key_of, voice_hash_of
from .transcript import check_transcript, mismatch_error

log = logging.getLogger(__name__)

KIND: Final = "measure"
"""The job kind this handler runs (``JobRecord.kind``)."""
RETRY_AFTER_S: Final = 60.0
"""``retry_after_s`` of a measurement that stopped on a take's execution problem (DC-2)."""

Role = Literal["calibration", "ladder"]
CorpusSource = Callable[[str], MaterialSet]
"""Reads the calibration corpus by its set id (``[measurement] corpus``); raises ``MaterialError``."""


@dataclass(frozen=True, slots=True)
class RungPlan:
    """A ladder rung: its segment in the run, its target length, and its paragraph's spoken length."""

    segment: int
    target: int
    chars: int
    paragraph_id: str


@dataclass(slots=True, eq=False)
class MeasureRun(JobRun):
    """A ``measure`` job the handler holds: a ``JobRun`` (so the engine's stages, failure handling and record
    view work on it) with the measurement's own state."""

    measurement_key: str = ""
    corpus_version: str = ""
    roles: list[Role] = field(default_factory=list)
    """Each segment's role, by its index: a calibration paragraph or a ladder rung."""
    rungs: list[RungPlan] = field(default_factory=list)
    """The ladder's rungs, shortest first."""
    active: set[int] = field(default_factory=set)
    """The segments whose takes may be made now: the calibration set, the band, and the rungs reached."""
    current: MeasurementRecord | None = None
    """A current measurement found when the job was planned: the job completes with it at once."""
    transcript: TranscriptCheck | None = None
    clip_embedding: tuple[float, ...] | None = None
    clip_retries: int = 0
    calibration: Calibration | None = None
    trend: PaceTrend | None = None
    tol: float | None = None
    speaking_share: float | None = None
    judgements: list[lad.RungJudgement] = field(default_factory=list)
    """The judged rungs, shortest first; it stops at the first rung that fails."""

    def role(self, attempt: Attempt) -> Role:
        """The role of the attempt's segment."""
        return self.roles[attempt.segment]


def _measure_run(run: JobRun) -> MeasureRun:
    if not isinstance(run, MeasureRun):
        raise TypeError(f"job {run.job_id} is not a measure job")
    return run


class MeasureStages(Stages):
    """The engine's stages, with what a measurement's takes are judged against (the module docstring)."""

    def measurement_key(self, run: JobRun, seg: SegmentWork) -> str | None:
        """Calibration takes name no measurement key; ladder takes name the key of the measurement being built."""
        mrun = _measure_run(run)
        return None if mrun.roles[seg.index] == "calibration" else mrun.measurement_key

    def scoring_facts(self, run: JobRun, seg: SegmentWork) -> ScoringFacts:
        """Calibration takes: no speaker or pace check. Ladder takes: the calibration anchor and baseline."""
        mrun = _measure_run(run)
        if mrun.roles[seg.index] == "calibration":
            return ScoringFacts()
        if mrun.calibration is None:
            raise RuntimeError(f"job {run.job_id}: a ladder take was scored before the calibration set")
        return ScoringFacts(anchor=mrun.calibration.anchor, similarity=mrun.calibration.similarity)


class MeasureHandler:
    """Runs ``measure`` jobs (``JobHandler[MeasureRun]``; the module docstring). Build it with
    ``build_measure_handler``."""

    def __init__(self, engine: JobEngine, corpus: CorpusSource) -> None:
        self.engine = engine
        self.core = engine.core
        self.stages = MeasureStages(self.core)
        self.failures = Failures(self.core, self.stages)
        self.view = JobView(self.core)
        self._corpus_source = corpus
        self._corpora: dict[str, MaterialSet] = {}
        self._reader = NumberReader()

    # ------------------------------------------------------------------ planning
    def corpus(self, set_id: str) -> MaterialSet:
        """The calibration corpus, read once and kept. A corpus that cannot be read is ``BACKEND_NOT_INSTALLED``:
        the installation is incomplete, and no measurement can run."""
        found = self._corpora.get(set_id)
        if found is None:
            try:
                found = self._corpus_source(set_id)
            except MaterialError as exc:
                raise NarrationError(
                    codes.BACKEND_NOT_INSTALLED,
                    f"the calibration corpus {set_id} cannot be read: {exc}",
                    hint="Run narration-admin doctor: the service's material/ folder is missing or changed.",
                    retryable=False,
                ) from exc
            self._corpora[set_id] = found
        return found

    def open(self, host: RunnerHost, job: JobRecord) -> MeasureRun:
        """Plan a claimed ``measure`` job from its request and the cache (the module docstring). Raises
        ``NarrationError`` for a job-level problem: no engine pinned, a clip the service may not clone, a
        corpus or ladder setting that cannot be measured with, a clip that cannot be read."""
        if job.kind != KIND:
            raise NarrationError(
                codes.INTERNAL, f"the measure handler was given a {job.kind} job", details={"kind": job.kind}
            )
        store, config, parts = host.store, self.core.config, self.core.parts
        voice = _voice(job.request)
        profile = _base_profile(host)
        voice_hash = voice_hash_of(clip_sha256=voice.sha256, transcript=voice.transcript, config=config)
        require_synthetic(store, config.voices.allow_sha256, voice.sha256)  # section 17.4: measuring is cloning
        settings = config.measurement
        corpus = self.corpus(settings.corpus)
        key = measurement_key_of(voice_hash=voice_hash, profile=profile, corpus=corpus, config=config)
        paragraphs, rungs = _plan_paragraphs(corpus, settings.length_ladder_spoken_chars, settings.trend_band_max_chars)
        request = GenerateRequest(
            voice=voice,
            hints=corpus.hints,
            segments=tuple(SegmentIn(segment_id=p.segment_id, cues=p.cues) for p in paragraphs),
            takes=settings.seeds,
            max_retakes=0,
            strict_text=False,
            priority=job.priority,
            expect_engine_profile=None,
        )
        found = store.get_measurement(voice_hash, profile.engine_profile_id)
        current = found if found is not None and found.measurement_key == key else None
        run = MeasureRun(
            job=job,
            request=request,
            voice_hash=voice_hash,
            clip=Path(),
            profile=profile,
            measurement=None,
            measured_error=None,
            scratch=store.scratch_path(SCRATCH_JOBS, job.job_id, "work").parent,
            fresh_keys={a.render_key for s in job.items for a in s.attempts if a.fresh},
            done_floor=job.progress.done_s,
            segments_floor=job.progress.segments_done,
            round_floor=job.round,
            measurement_key=key,
            corpus_version=corpus_version(corpus),
            current=current,
        )
        if current is not None:
            store.touch("measurement", current.measurement_key)
            run.message = f"already measured under {profile.engine_profile_id}"
            return run
        # Section 17.3: a caller's file is read only through the daemon's platform check (or the one built in),
        # exactly as JobEngine.open reads a generation's clip.
        check = parts.check_path if parts.check_path is not None else host.platform.check_readable_path
        run.clip = stage_clip(store, voice, check_path=check)
        self.core.residency.need_mb["qwen"] = profile.vram_need_mb
        planned = parts.text.plan_request(request.segments, request.hints, strict_text=False)
        calibration_count = len(paragraphs) - len(rungs)
        band_max = settings.trend_band_max_chars
        for index, text in enumerate(planned):
            role: Role = "calibration" if index < calibration_count else "ladder"
            run.roles.append(role)
            seg = SegmentWork(
                index=index,
                text=text,
                hints=hints_used(text, request.hints),
                est_s=estimated_audio_s(text, None),
                flags=[*text.warnings, *(w for cue in text.cues for w in cue.warnings)],
            )
            run.segments.append(seg)
        run.rungs = [
            RungPlan(
                segment=calibration_count + k,
                target=target,
                chars=planned[calibration_count + k].spoken_chars,
                paragraph_id=paragraph.segment_id,
            )
            for k, (target, paragraph) in enumerate(rungs)
        ]
        run.active = set(range(calibration_count)) | {r.segment for r in run.rungs if r.target <= band_max}
        for seg in run.segments:  # roles first: the stages read them to look analyses up
            for attempt in range(settings.seeds):
                seg.slots.append([self.stages.plan_attempt(host, run, seg, attempt, attempt, 0)])
        run.message = (
            f"planned: {calibration_count} calibration paragraph(s) and a ladder of {len(rungs)} rung(s), "
            f"{settings.seeds} seed(s) each, on {profile.engine_profile_id}"
        )
        log.info("job %s: %s", job.job_id, run.message)
        return run

    # ------------------------------------------------------------------ advancing
    def advance(self, host: RunnerHost, run: MeasureRun) -> Outcome:
        """Do one piece of the measurement, or one bounded wait; ``finished`` once its record is written."""
        if run.current is not None:
            self.finish(host, run)
            return "finished"
        if run.transcript is None or run.clip_embedding is None:
            return self._check_clip(host, run)
        self._raise_on_errors(run)
        now = self.core.parts.clock()
        pending = [a for i in sorted(run.active) for a in run.segments[i].current() if not a.settled]
        ready = [a for a in pending if a.deferred_until <= now]
        todo = next((a for a in ready if a.stage == "render"), None)
        if todo is not None:
            return self.failures.guarded(host, run, todo, "qwen")
        todo = next((a for a in ready if a.stage == "post"), None)
        if todo is not None:
            return self.failures.guarded(host, run, todo, None)
        if not any(a.stage == "render" for a in pending):  # every render first: one Qwen load, then one QA load
            todo = next((a for a in ready if a.stage == "score" and run.role(a) == "calibration"), None)
            if todo is not None:
                return self.failures.guarded(host, run, todo, "qa")
            if not any(run.role(a) == "calibration" for a in pending):
                if run.calibration is None:
                    self._calibrate(run)
                todo = next((a for a in ready if a.stage == "score"), None)
                if todo is not None:
                    return self.failures.guarded(host, run, todo, "qa")
        if pending:
            return self._wait(host, run, pending)
        if run.calibration is None:
            self._calibrate(run)
        if self._judge(host, run):
            return "worked"
        self.finish(host, run)
        return "finished"

    def _wait(self, host: RunnerHost, run: MeasureRun, pending: list[Attempt]) -> Outcome:
        """Everything left is being made by another holder: wait a little before looking again."""
        run.message = "waiting for takes another job is making"
        if not host.sleep(self.core.parts.defer_s):
            return "stopped"
        for attempt in pending:
            attempt.deferred_until = 0.0
        return "waited"

    def _raise_on_errors(self, run: MeasureRun) -> None:
        """A take the engine could not make fails the measurement (the module docstring)."""
        for index in sorted(run.active):
            for attempt in run.segments[index].attempts():
                if attempt.error is None:
                    continue
                seg, flag = run.segments[index], attempt.error
                details = {"segment_id": seg.segment_id, "attempt": attempt.attempt, "flag": flag.code}
                if flag.code == codes.GPU_OOM:
                    raise NarrationError(
                        codes.GPU_UNAVAILABLE,
                        f"the measurement stopped: {flag.message}",
                        details=details,
                        retry_after_s=GPU_UNAVAILABLE_RETRY_S,
                        hint="The GPU ran out of memory twice on one take. Send the same request again once the "
                        "GPU has more free memory; what was made is kept.",
                    )
                raise NarrationError(
                    codes.INTERNAL,
                    f"the measurement stopped: {flag.message}",
                    details=details,
                    retryable=True,
                    retry_after_s=RETRY_AFTER_S,
                    hint="A worker failed on one take of the measurement. Send the same request again; what was "
                    "made is kept. If it fails again, the daemon's log has the details.",
                )

    # ------------------------------------------------------------------ the transcript check
    def _check_clip(self, host: RunnerHost, run: MeasureRun) -> Outcome:
        """Transcribe the clip and check its transcript; embed it for the anchor (the QA group)."""
        core, stages = self.core, self.stages
        try:
            if not stages.ready(host, run, stages.qa_need()):
                return "stopped" if host.should_stop() else "waited"
            core.phase(host, run, "scoring")
            run.message = "checking the transcript against the clip"
            client = host.workers.client("qa")
            heard = client.request(
                "transcribe",
                {"wav": str(run.clip), "language": names.LANGUAGE, "word_timestamps": True, "long_form": True},
                timeout_s=QA_TIMEOUT_S,
            )
            check, errors, words = check_transcript(
                run.request.voice.transcript, str(heard.get("text") or ""), self._reader, DEFAULT_PROFILE
            )
            if not check.ok:
                raise mismatch_error(check, errors, words, DEFAULT_PROFILE)
            device = "cpu" if core.config.gpu.device == "cpu" else "cuda"
            embedded = client.request("embed", {"wav": str(run.clip), "device": device}, timeout_s=QA_TIMEOUT_S)
        except WorkerFailure as exc:
            return self._clip_failed(host, run, exc)
        except (WorkerCrashed, WorkerTimeout) as exc:
            if host.stop_mode == "now" or host.should_stop():
                return "stopped"
            core.residency.forget("qa")
            return self._clip_retry(run, exc)
        run.transcript = check
        run.clip_embedding = tuple(float(v) for v in embedded["embedding"])
        run.message = f"the transcript matches the clip ({errors} word error(s) in {words})"
        return "worked"

    def _clip_failed(self, host: RunnerHost, run: MeasureRun, exc: WorkerFailure) -> Outcome:
        code = worker_code(exc)
        if code == codes.BACKEND_NOT_INSTALLED:
            raise NarrationError(codes.BACKEND_NOT_INSTALLED, exc.message, details=exc.details) from exc
        if code == codes.UNSUPPORTED_AUDIO:
            raise NarrationError(
                codes.UNSUPPORTED_AUDIO,
                f"the QA worker could not read the clip: {exc.message}",
                field="voice",
                details={"worker_code": code},
            ) from exc
        if code == codes.GPU_OOM:
            if run.clip_retries >= MAX_RETRIES:
                raise NarrationError(
                    codes.GPU_UNAVAILABLE,
                    f"out of GPU memory while checking the clip, after one retry: {exc.message}",
                    retry_after_s=GPU_UNAVAILABLE_RETRY_S,
                ) from exc
            run.clip_retries += 1
            self.stages.drop(host, "qa")
            return "waited" if host.sleep(OOM_WAIT_S) else "stopped"
        if code in LOST_STATE:
            self.core.residency.forget("qa")
            return self._clip_retry(run, exc)
        raise NarrationError(
            codes.INTERNAL,
            f"the QA worker could not check the clip: {exc.message}",
            details={"worker_code": code},
            retryable=True,
            retry_after_s=RETRY_AFTER_S,
        ) from exc

    @staticmethod
    def _clip_retry(run: MeasureRun, exc: Exception) -> Outcome:
        if run.clip_retries >= MAX_RETRIES:
            raise NarrationError(
                codes.INTERNAL,
                f"the QA worker failed twice while checking the clip: {exc}",
                details={"exception": type(exc).__name__},
                retryable=True,
                retry_after_s=RETRY_AFTER_S,
            ) from exc
        run.clip_retries += 1
        log.warning("job %s: the QA worker failed on the clip (%s); once more", run.job_id, exc)
        return "worked"

    # ------------------------------------------------------------------ the calibration set
    def _calibrate(self, run: MeasureRun) -> None:
        """The anchor and similarity baseline, from the clip and every calibration take."""
        assert run.clip_embedding is not None
        embeddings: list[tuple[float, ...]] = []
        for index, role in enumerate(run.roles):
            if role != "calibration":
                continue
            for attempt in run.segments[index].attempts():
                embedding = attempt.analysis.embedding if attempt.analysis is not None else None
                if embedding is None:
                    raise NarrationError(
                        codes.INTERNAL,
                        f"calibration take {run.segments[index].segment_id} attempt {attempt.attempt} has no speaker "
                        "embedding",
                        retryable=False,
                    )
                embeddings.append(embedding)
        try:
            run.calibration = calibrate(run.clip_embedding, embeddings, model=self.core.parts.qa_pins.sv.name)
        except ValueError as exc:
            raise NarrationError(
                codes.INTERNAL, f"the calibration takes' embeddings cannot be compared: {exc}", retryable=False
            ) from exc
        run.message = (
            f"calibrated: anchor p5 {run.calibration.similarity.anchor_p5:.4f}, "
            f"consistency p5 {run.calibration.similarity.consistency_p5:.4f}"
        )

    # ------------------------------------------------------------------ the ladder
    def _judge(self, host: RunnerHost, run: MeasureRun) -> bool:
        """Judge every rung reached and not yet judged, shortest first; then start the next rung if every rung so
        far passed. True when a rung was started (there is more to make)."""
        settings = self.core.config.measurement
        calibration = run.calibration
        assert calibration is not None
        if run.trend is None:
            band = [self._rung(run, r) for r in run.rungs if r.target <= settings.trend_band_max_chars]
            try:
                run.trend = lad.fit_trend(band, settings.trend_band_max_chars)
            except ValueError as exc:
                raise NarrationError(
                    codes.INTERNAL,
                    f"the measurement stopped: no rung of the trend band has a measured pace ({exc})",
                    details={"band": [r.paragraph_id for r in band]},
                    retryable=True,
                    retry_after_s=RETRY_AFTER_S,
                    hint="QA found no voiced span in any take of the trend band, so the pace trend cannot be "
                    "fitted and nothing was published. Send the same request again; if it fails again, the "
                    "daemon's log has the details.",
                ) from exc
            run.tol = lad.pace_tol(band, settings.trend_band_max_chars, settings.pace_tol_min)
            run.speaking_share = lad.speaking_share(band, settings.trend_band_max_chars)
        assert run.tol is not None
        sim_warn = round(calibration.similarity.anchor_p5 - settings.sim_warn_margin, 6)
        while len(run.judgements) < len(run.rungs):
            plan = run.rungs[len(run.judgements)]
            if plan.segment not in run.active:
                if not all(j.passes for j in run.judgements):
                    return False
                run.active.add(plan.segment)
                self.core.phase(host, run, "rendering")
                run.message = f"ladder: rung {plan.target} ({plan.paragraph_id}, {plan.chars} spoken characters)"
                log.info("job %s: %s", run.job_id, run.message)
                return True
            judgement = lad.judge(self._rung(run, plan), trend=run.trend, tol=run.tol, sim_warn=sim_warn)
            run.judgements.append(judgement)
            log.info(
                "job %s: rung %d %s%s",
                run.job_id,
                plan.target,
                "passes" if judgement.passes else "fails: ",
                "; ".join(judgement.reasons),
            )
        return False

    def _rung(self, run: MeasureRun, plan: RungPlan) -> lad.Rung:
        seg = run.segments[plan.segment]
        return lad.Rung(
            paragraph_id=plan.paragraph_id,
            target=plan.target,
            chars=plan.chars,
            seeds=tuple(_seed_take(a) for a in seg.attempts()),
        )

    # ------------------------------------------------------------------ the end of the job
    def finish(self, host: RunnerHost, run: MeasureRun) -> JobRecord | None:
        """Publish the measurement and complete the job (``JobRecord.result``: its handle). A job cancelled
        meanwhile ends ``cancelled``."""
        store = host.store
        record = run.current if run.current is not None else store.put_measurement(self._record(run))
        path = store.measurement_dir(record.voice_hash, record.engine_profile.id) / MEASUREMENT_JSON
        outcome = self._states(run, record)
        stopped = next(
            (
                {"paragraph_id": p.paragraph_id, "chars": p.chars, "reasons": list(j.reasons)}
                for p, j in zip(run.rungs, run.judgements, strict=False)
                if not j.passes
            ),
            None,
        )
        result: dict[str, Any] = {
            "voice_hash": record.voice_hash,
            "engine_profile": {"id": record.engine_profile.id, "hash": record.engine_profile.hash},
            "measurement_key": record.measurement_key,
            "path": str(path),
            "max_segment_chars": record.max_segment_chars,
            "max_segment_seconds": record.max_segment_seconds,
            "ladder_stopped_at": stopped,
        }
        if record.max_segment_chars is None:
            run.message = "measured: no ladder rung was read reliably, so this voice has no reliable length"
        else:
            run.message = (
                f"measured: reliable up to {record.max_segment_chars} spoken characters "
                f"({record.max_segment_seconds} s)"
                + (f"; the ladder stopped at {stopped['chars']} characters" if stopped is not None else "")
            )
        updated = store.update_job(
            run.job_id,
            expect_status="running",
            status="completed",
            phase=None,
            round=run.shown_round,
            outcome=outcome,
            progress=self.view.progress(run, complete=True),
            items=self.view.items(run),
            result=result,
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
                    progress=self.view.progress(run, complete=True),
                    items=self.view.items(run),
                    message="cancelled as it finished; the measurement was published",
                )
        log.info("job %s: %s", run.job_id, run.message)
        self.release(run)
        return updated

    def _record(self, run: MeasureRun) -> MeasurementRecord:
        """The measurement (section 6, App. B) from the calibration set and the judged ladder."""
        calibration, transcript, trend, tol = run.calibration, run.transcript, run.trend, run.tol
        assert calibration is not None and transcript is not None and trend is not None and tol is not None
        share = run.speaking_share if run.speaking_share is not None else 1.0
        rungs = [self._rung(run, plan) for plan in run.rungs[: len(run.judgements)]]
        result = lad.outcome(rungs, run.judgements)
        takes: list[CalibrationTake] = []
        for index, role in enumerate(run.roles):
            if role != "calibration":
                continue
            seg = run.segments[index]
            for attempt in seg.attempts():
                analysis, take = attempt.analysis, attempt.take
                assert analysis is not None and analysis.embedding is not None and take is not None
                (sim,) = similarities([analysis.embedding], calibration.anchor.embedding)
                takes.append(
                    CalibrationTake(
                        paragraph_id=seg.segment_id,
                        seed=attempt.seed,
                        attempt=attempt.attempt,
                        take_id=take.take_id,
                        sim_anchor=round(sim, 6),
                    )
                )
        return MeasurementRecord(
            voice_hash=run.voice_hash,
            clip_sha256=run.request.voice.sha256,
            engine_profile=EngineRef(id=run.profile.engine_profile_id, hash=run.profile.hash),
            measurement_key=run.measurement_key,
            transcript_check=transcript,
            corpus=run.corpus_version,
            similarity=calibration.similarity,
            pace=Pace(method=names.PACE_METHOD, trend=trend, tol=tol, curve=result.curve, speaking_share=share),
            max_segment_chars=result.max_segment_chars,
            max_segment_seconds=result.max_segment_seconds,
            ladder=tuple(lad.ladder_rung(r, j) for r, j in zip(rungs, run.judgements, strict=True)),
            anchor=calibration.anchor,
            calibration=tuple(takes),
            measured_at=utc_iso(time.time()),
        )

    def _states(self, run: MeasureRun, record: MeasurementRecord) -> JobOutcome:
        """Each segment's final state, and the job's outcome: ``needs_attention`` when no rung passed or a
        calibration take failed QA; otherwise ``all_passed``."""
        attention = record.max_segment_chars is None
        judged = {plan.segment: j for plan, j in zip(run.rungs, run.judgements, strict=False)}
        for index, seg in enumerate(run.segments):
            verdicts = [a.analysis.qa.verdict for a in seg.attempts() if a.analysis is not None]
            worst = max(verdicts, key=VERDICT_RANK.__getitem__) if verdicts else None
            state: SegmentState
            if run.current is not None:
                state = "cached"
            elif run.roles[index] == "calibration":
                state = VERDICT_STATE[worst] if worst is not None else "error"
                attention = attention or worst == "fail"
            elif index in judged:
                state = ("passed" if worst == "pass" else "warned") if judged[index].passes else "failed_qa"
            else:
                state = "skipped"
            seg.state = state
        return "needs_attention" if attention else "all_passed"

    def cancel(self, host: RunnerHost, run: MeasureRun) -> JobRecord | None:
        """Finish a cancelled measurement: nothing is published; every take made stays in the cache."""
        return self.view.cancel(host, run)

    def fail(self, host: RunnerHost, job: JobRecord, error: NarrationError, run: MeasureRun | None) -> JobRecord | None:
        """Record a job-level failure (``REF_TEXT_MISMATCH`` among them)."""
        return self.view.fail(host, job, error, run)

    def save(self, host: RunnerHost, run: MeasureRun) -> bool:
        """Write the job's progress, items, phase and message; False when it is no longer ``running``."""
        return self.view.save(host, run)

    def release(self, run: MeasureRun) -> None:
        """Remove the job's scratch files."""
        self.view.release(run)

    def remaining_audio_s(self, run: MeasureRun) -> float:
        """Audio seconds of work left, for the queue's drain estimate (every rung counted until the ladder ends)."""
        return self.view.remaining_audio_s(run)

    def close(self) -> None:
        """Stop the thread that renews the leases (``Registry.close`` calls it at shutdown). The handler shares
        the job engine's ``LeaseKeeper``, whose ``close`` may be called more than once; work started later starts
        the thread again."""
        self.core.leases.close()


# ---------------------------------------------------------------------- helpers


def _voice(request: Mapping[str, Any]) -> VoiceSpec:
    """The request's voice (``measure_voice``'s input schema: ``{voice: {path, sha256, transcript}}``)."""
    try:
        voice = request["voice"]
        values = {name: voice[name] for name in ("path", "sha256", "transcript")}
    except (KeyError, TypeError) as exc:
        raise NarrationError(codes.INTERNAL, f"the job's request cannot be read: {exc}", retryable=False) from exc
    if not all(isinstance(v, str) and v for v in values.values()):
        raise NarrationError(codes.INTERNAL, "the job's request has a malformed voice", retryable=False)
    return VoiceSpec(path=values["path"], sha256=values["sha256"], transcript=values["transcript"])


def _base_profile(host: RunnerHost) -> EngineProfile:
    profile = host.store.current_engine_profile("base")
    if profile is None:
        raise NarrationError(
            codes.BACKEND_NOT_INSTALLED,
            "no engine profile is pinned for the Base clone path",
            hint="Ask the operator to run narration-admin install, then narration-admin engine pin.",
        )
    try:
        call_cap(profile, "x")
        qwen_load_payload(profile, host.config.gpu.device)
    except ProfileError as exc:
        raise NarrationError(
            codes.BACKEND_NOT_INSTALLED, str(exc), hint="Ask the operator to pin the engine again."
        ) from exc
    return profile


def _plan_paragraphs(
    corpus: MaterialSet, ladder: tuple[int, ...], band_max_chars: int
) -> tuple[list[MaterialParagraph], list[tuple[int, MaterialParagraph]]]:
    """The measurement's paragraphs in order (the calibration set, then the ladder shortest first), and the
    ladder's (target, paragraph) pairs. Each configured rung needs the corpus paragraph written for it, and at
    least one rung must be in the trend band (at most ``band_max_chars``), or there is no trend to judge by."""
    by_target = {p.target_spoken_chars: p for p in corpus.ladder if p.target_spoken_chars is not None}
    rungs: list[tuple[int, MaterialParagraph]] = []
    for target in sorted(ladder):
        paragraph = by_target.get(target)
        if paragraph is None:
            raise NarrationError(
                codes.BACKEND_NOT_INSTALLED,
                f"the calibration corpus {corpus.set_id} has no ladder paragraph for the rung of {target} spoken "
                "characters ([measurement] length_ladder_spoken_chars)",
                hint="Set length_ladder_spoken_chars to rungs the corpus has, or install a corpus that has them.",
                retryable=False,
            )
        rungs.append((target, paragraph))
    if not any(target <= band_max_chars for target, _ in rungs):
        raise NarrationError(
            codes.BACKEND_NOT_INSTALLED,
            f"no rung of length_ladder_spoken_chars is at or under trend_band_max_chars ({band_max_chars}), so "
            "the pace trend has nothing to be fitted to ([measurement])",
            hint="Add a rung at or under trend_band_max_chars to length_ladder_spoken_chars (the default ladder "
            "starts at 80), or raise trend_band_max_chars.",
            retryable=False,
        )
    if not corpus.paragraphs:
        raise NarrationError(
            codes.BACKEND_NOT_INSTALLED, f"the calibration corpus {corpus.set_id} has no calibration paragraph"
        )
    return [*corpus.paragraphs, *(p for _, p in rungs)], rungs


def _seed_take(attempt: Attempt) -> lad.SeedTake:
    """One seed's take of a rung, from its analysis (``narration.qa``'s metrics and flags)."""
    analysis, render, take = attempt.analysis, attempt.render, attempt.take
    if analysis is None:
        return lad.SeedTake(
            seed=attempt.seed,
            attempt=attempt.attempt,
            take_id=take.take_id if take is not None else None,
            cps=None,
            wer_adj=None,
            word_errors=None,
            sim=None,
            verdict="fail",
            duration_s=None,
            token_cap=bool(render is not None and render.hit_token_cap),
            flags=(attempt.error.code,) if attempt.error is not None else (),
        )
    qa = analysis.qa
    found = {f.code for f in qa.flags}
    return lad.SeedTake(
        seed=attempt.seed,
        attempt=attempt.attempt,
        take_id=take.take_id if take is not None else None,
        cps=qa.metrics.articulation_cps,
        wer_adj=qa.metrics.wer_adj,
        word_errors=qa.metrics.word_errors,
        sim=qa.metrics.spk_sim_anchor,
        verdict=qa.verdict,
        duration_s=take.delivery.duration_s if take is not None else None,
        exact_mismatch=codes.EXACT_SPAN_MISMATCH in found or not qa.metrics.exact_ok,
        head_insertion=codes.HEAD_INSERTION in found,
        token_cap=codes.TOKEN_CAP_HIT in found or bool(render is not None and render.hit_token_cap),
        flags=tuple(dict.fromkeys(f.code for f in qa.flags)),
        spoken_cps=qa.metrics.spoken_cps,
        wpm=qa.metrics.spoken_wpm,
    )


def build_measure_handler(
    engine: JobEngine, *, material_root: Path | None = None, corpus: CorpusSource | None = None
) -> MeasureHandler:
    """The ``measure`` kind's handler, for the runner's ``Registry`` (WP32 assembles it in ``installed_engine``).

    - ``engine``: the daemon's ``JobEngine``. The handler shares its parts and state: the model residency
      (one resident group across every kind), the throughput estimate and the canary outcome of the Qwen load.
    - ``material_root``: the folder that holds the service's ``material/`` sets; default: the checkout's
      (``corpus.default_material_root``).
    - ``corpus``: reads a calibration corpus by set id, in place of ``material_root`` (for tests).

    Use it with ``measure_registry`` to get the registry the runner takes.
    """
    source: CorpusSource = corpus if corpus is not None else (lambda set_id: load_corpus(set_id, material_root))
    return MeasureHandler(engine, source)


def measure_registry(engine: JobEngine, handler: MeasureHandler) -> Registry:
    """The runner's handlers by kind: the job engine's (``generate``, ``analyse``) and ``measure``."""
    base = Registry.of(engine)
    return Registry(handlers={**base.handlers, KIND: handler}, residency=base.residency, throughput=base.throughput)


__all__ = [
    "KIND",
    "RETRY_AFTER_S",
    "CorpusSource",
    "MeasureHandler",
    "MeasureRun",
    "MeasureStages",
    "RungPlan",
    "build_measure_handler",
    "measure_registry",
]
