"""The ``pronunciation`` job kind (design sections 7.6, 8, 9.1, 10 and 11.1; plan.md WP35): ``audition_pronunciation``.

``AuditionHandler`` is a ``narration.jobs.handlers.JobHandler``: the runner dispatches a claimed ``pronunciation``
job to it, and it does the job a piece at a time, so the daemon can stop, cancel or preempt it between pieces. It
is the job engine (``JobEngine``) over the engine's own core (``JobEngine.core``: the model residency across every
kind, the throughput estimate and the canary outcome of the Qwen load in use), with its own plan, stages and
record view. So every variant's take is rendered on Qwen Base, post-processed and scored exactly as a generation
take is: answered by the cache first, made at most once under a lease, and retaken when it is a retake trigger.

**The job, in order.** The request is the one the front-end kept (``narration.backend.steps``): the voice, the
term, the variants and the optional carrier.

1. **Plan** (``variants``): one segment per variant, the carrier (or the term alone) with the variant's hint, so
   its engine text has the respelling in the term's place. Each segment has one take slot, attempt 0. The keys
   and the seed are a generation take's (sections 10.2, 10.3), so a variant is cached like a take, and a later
   ``submit_job`` of the same text with the chosen respelling as its hint finds its render in the cache. The
   voice need not be measured (section 7.6): nothing here reads a measurement.
2. **Render, post-process, score**, round by round (section 8), with retakes of a take that is a retake
   trigger, up to ``[defaults] max_retakes`` per variant, on the next attempt numbers: a take cut at its token
   cap, or one with words inserted before the text, tells the caller nothing about the respelling.
3. **The clip's speaker embedding** (the QA group, once, before the first take is scored), for each take's
   similarity to the clip.
4. **Finish**: each variant's suggestion (section 8's tiers), the outcome, and each take's similarity to the
   clip in ``JobRecord.result`` (``spk_sim_clip``, by take id). ``get_results`` reads the takes from the job's
   items and the cache (``narration.backend.assemble.assemble_audition``).

**What an audition take is judged on** (``AuditionStages``; section 11.1). The scorer's checks on the take and
its text: the signal and the token cap, ``wer_adj`` against the text as sent with the term collapsed (the
variant's respelling is the term's alias, ``variants``), the term, head and end insertions, and the cue
alignment. No speaker or pace check is in its verdict: both judge a take against a measurement, which an audition
does not need. Its analysis names no measurement key, as ``AnalysisKeyInputs`` has it for work with no speaker or
pace check, so no generation request (which names its measurement's key) ever takes it for one of its own.

**Speaker similarity to the clip** is reported, not judged: each take's embedding (in its analysis) against the
clip's, as a cosine, in the job's result. It is a per-job report, like the consistency report: it depends on
the clip as well as the take, and no cached verdict holds it. A take far below the voice's usual similarity
(the ``[measurement] sim_fail_floor`` of 0.90 is a guide) is not the voice, whatever it says.

**Failures** are the engine's (``narration.jobs.failures``): a job-level problem fails the job (no Base engine
pinned, a clip the service may not clone, a clip the worker cannot prepare); a take-level execution problem
flags its variant, and the job goes on with the rest. The clip's embedding fails the job only when the QA worker
cannot read the clip (``UNSUPPORTED_AUDIO``), runs out of memory twice (``GPU_UNAVAILABLE``) or fails twice
(``INTERNAL``, retryable: everything made is kept, and the same request resumes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

from narration.contracts import codes
from narration.contracts.errors import NarrationError, WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.models import EngineProfile, JobRecord, MeasuredError
from narration.jobs.admission import GPU_UNAVAILABLE_RETRY_S
from narration.jobs.core import LOST_STATE, worker_code
from narration.jobs.engine import SCRATCH_JOBS, JobEngine
from narration.jobs.failures import MAX_RETRIES, OOM_WAIT_S, Failures
from narration.jobs.host import RunnerHost
from narration.jobs.pins import ProfileError, call_cap, qwen_load_payload
from narration.jobs.plan import GenerateRequest, VoiceSpec, estimated_audio_s, hints_used
from narration.jobs.record import JobView
from narration.jobs.stages import QA_TIMEOUT_S, ScoringFacts, Stages
from narration.jobs.state import JobRun, Outcome, SegmentWork
from narration.jobs.voice import require_synthetic, stage_clip
from narration.measure.baseline import similarities
from narration.measure.lookup import voice_hash_of

from .variants import AuditionAsk, AuditionRequestError

log = logging.getLogger(__name__)

KIND: Final = "pronunciation"
"""The job kind this handler runs (``JobRecord.kind``; ``audition_pronunciation``'s)."""
RETRY_AFTER_S: Final = 60.0
"""``retry_after_s`` of an audition that stopped on the clip's embedding (DC-2)."""
SIMILARITY: Final = "spk_sim_clip"
"""The key of ``JobRecord.result`` that holds each take's similarity to the clip, by take id."""


@dataclass(slots=True, eq=False)
class AuditionRun(JobRun):
    """A ``pronunciation`` job the handler holds: a ``JobRun`` (so the engine's stages, failure handling and
    record view work on it) with the audition's own state."""

    ask: AuditionAsk | None = None
    clip_embedding: tuple[float, ...] | None = None
    """The clip's speaker embedding, taken once per job (never cached: a report, not a verdict)."""
    clip_retries: int = 0


class AuditionStages(Stages):
    """The engine's stages, with what an audition take is judged against: no measurement (the module docstring)."""

    def measurement_key(self, run: JobRun, seg: SegmentWork) -> str | None:
        """None: an audition take has no speaker or pace check, so its analysis names no measurement."""
        return None

    def scoring_facts(self, run: JobRun, seg: SegmentWork) -> ScoringFacts:
        """Nothing: no anchor, similarity baseline or pace curve, so the scorer checks neither."""
        return ScoringFacts()


class AuditionView(JobView):
    """The engine's record view, with each take's similarity to the clip in the job's result."""

    def result(self, run: JobRun) -> dict[str, Any]:
        """The engine's advice (suggestions, consistency) and ``spk_sim_clip``: each scored take's cosine
        similarity to the clip, by take id, once the clip is embedded."""
        out = super().result(run)
        out[SIMILARITY] = clip_similarity(run)
        return out


def clip_similarity(run: JobRun) -> dict[str, float]:
    """Each scored take's similarity to the clip, by take id; nothing before the clip is embedded."""
    clip = run.clip_embedding if isinstance(run, AuditionRun) else None
    if clip is None:
        return {}
    out: dict[str, float] = {}
    for seg in run.segments:
        for attempt in seg.attempts():
            analysis, take = attempt.analysis, attempt.take
            if analysis is None or take is None or analysis.embedding is None:
                continue
            try:
                (sim,) = similarities([analysis.embedding], clip)
            except ValueError:
                log.warning("job %s: take %s's embedding cannot be compared with the clip's", run.job_id, take.take_id)
                continue
            out[take.take_id] = round(sim, 6)
    return out


class AuditionHandler(JobEngine):
    """Runs ``pronunciation`` jobs (``JobHandler[AuditionRun]``; the module docstring). Build it with
    ``build_audition_handler``: it shares the daemon's job engine's core, and so its residency, throughput and
    canary."""

    def __init__(self, engine: JobEngine) -> None:  # the engine's core, not a new one: no super().__init__
        self.core = engine.core
        self.stages = AuditionStages(self.core)
        self.failures = Failures(self.core, self.stages)
        self.view = AuditionView(self.core)

    # ------------------------------------------------------------------ planning
    def open(self, host: RunnerHost, job: JobRecord) -> AuditionRun:
        """Plan a claimed ``pronunciation`` job from its request and the cache (the module docstring). Raises
        ``NarrationError`` for a job-level problem: no Base engine pinned, a clip the service may not clone, a
        clip that cannot be read."""
        if job.kind != KIND:
            raise NarrationError(
                codes.INTERNAL,
                f"the audition handler was given a {job.kind} job",
                details={"kind": job.kind},
                retryable=False,
                hint="This is a bug in the service; nothing was rendered. Report it.",
            )
        store, config, parts = host.store, self.config, self.parts
        try:
            ask = AuditionAsk.parse(job.request)
        except AuditionRequestError as exc:
            raise NarrationError(codes.INTERNAL, str(exc), retryable=False) from exc
        voice = _voice(job.request)
        profile = base_profile(host)
        voice_hash = voice_hash_of(clip_sha256=voice.sha256, transcript=voice.transcript, config=config)
        require_synthetic(store, config.voices.allow_sha256, voice.sha256)  # section 17.4: an audition clones
        # Section 17.3: a caller's file is read only through the daemon's platform check (or the one built in).
        check = parts.check_path if parts.check_path is not None else host.platform.check_readable_path
        clip = stage_clip(store, voice, check_path=check)
        pairs = ask.segments()
        request = GenerateRequest(
            voice=voice,
            hints=(),
            segments=tuple(segment for segment, _ in pairs),
            takes=1,
            max_retakes=config.defaults.max_retakes,
            strict_text=False,
            priority=job.priority,
            expect_engine_profile=None,
        )
        run = AuditionRun(
            job=job,
            request=request,
            voice_hash=voice_hash,
            clip=clip,
            profile=profile,
            measurement=None,
            measured_error=_measured_error(host),
            scratch=store.scratch_path(SCRATCH_JOBS, job.job_id, "work").parent,
            fresh_keys={a.render_key for s in job.items for a in s.attempts if a.fresh},
            done_floor=job.progress.done_s,
            segments_floor=job.progress.segments_done,
            round_floor=job.round,
            ask=ask,
        )
        self.residency.need_mb["qwen"] = profile.vram_need_mb
        for index, (segment_in, hint) in enumerate(pairs):
            try:
                (text,) = parts.text.plan_request([segment_in], [hint], strict_text=False)
            except NarrationError as exc:  # checked at submit; a change of rules since is refused here
                raise NarrationError(
                    exc.code,
                    exc.message,
                    field="carrier" if ask.carrier else "term",
                    hint=exc.hint,
                    details=exc.details,
                    retryable=exc.retryable,
                ) from exc
            seg = SegmentWork(
                index=index,
                text=text,
                hints=hints_used(text, (hint,)),
                est_s=estimated_audio_s(text, None),
                flags=[*text.warnings, *(w for cue in text.cues for w in cue.warnings)],
            )
            run.segments.append(seg)
            seg.slots.append([self.stages.plan_attempt(host, run, seg, 0, 0, 0)])
        # A job taken again walks the rounds it finished, from the cache, as JobEngine.open does.
        while all(a.settled for seg in run.segments for a in seg.current()) and self._start_retakes(
            host, run, announce=False
        ):
            pass
        run.message = f"planned {len(run.segments)} variant(s) of {ask.term!r} on {profile.engine_profile_id}"
        log.info("job %s: %s", job.job_id, run.message)
        return run

    # ------------------------------------------------------------------ advancing
    def advance(self, host: RunnerHost, run: JobRun) -> Outcome:
        """Do one piece of the audition, or one bounded wait; ``finished`` once its record is written. The clip is
        embedded once every take is rendered, in the QA load that scores them (the module docstring)."""
        audition = _audition_run(run)
        if audition.clip_embedding is None:
            pending = [a for seg in run.segments for a in seg.current() if not a.settled]
            if not any(a.stage == "render" for a in pending):
                return self._embed_clip(host, audition)
        return super().advance(host, run)

    def _embed_clip(self, host: RunnerHost, run: AuditionRun) -> Outcome:
        """The clip's speaker embedding (App. A ``embed``), for each take's similarity to it."""
        core, stages = self.core, self.stages
        try:
            if not stages.ready(host, run, stages.qa_need()):
                return "stopped" if host.should_stop() else "waited"
            core.phase(host, run, "scoring")
            run.message = "embedding the voice's clip, for each take's similarity to it"
            device = "cpu" if core.config.gpu.device == "cpu" else "cuda"
            embedded = host.workers.client("qa").request(
                "embed", {"wav": str(run.clip), "device": device}, timeout_s=QA_TIMEOUT_S
            )
        except WorkerFailure as exc:
            return self._clip_failed(host, run, exc)
        except (WorkerCrashed, WorkerTimeout) as exc:
            if host.stop_mode == "now" or host.should_stop():
                return "stopped"
            core.residency.forget("qa")
            return self._clip_retry(run, exc)
        run.clip_embedding = tuple(float(v) for v in embedded["embedding"])
        run.clip_retries = 0
        return "worked"

    def _clip_failed(self, host: RunnerHost, run: AuditionRun, exc: WorkerFailure) -> Outcome:
        code = worker_code(exc)
        if code == codes.BACKEND_NOT_INSTALLED:
            raise NarrationError(
                codes.BACKEND_NOT_INSTALLED,
                exc.message,
                details=exc.details,
                hint="Ask the operator to run narration-admin doctor, then narration-admin install.",
            ) from exc
        if code == codes.UNSUPPORTED_AUDIO:
            raise NarrationError(
                codes.UNSUPPORTED_AUDIO,
                f"the QA worker could not read the voice's clip: {exc.message}",
                field="voice",
                details={"worker_code": code},
                hint="Send a WAV the service designed (at most 30 s and 20 MB).",
            ) from exc
        if code == codes.GPU_OOM:
            if run.clip_retries >= MAX_RETRIES:
                raise NarrationError(
                    codes.GPU_UNAVAILABLE,
                    f"out of GPU memory while embedding the voice's clip, after one retry: {exc.message}",
                    retry_after_s=GPU_UNAVAILABLE_RETRY_S,
                    hint="Send the same request again once the GPU has more free memory (get_server_status).",
                ) from exc
            run.clip_retries += 1
            self.stages.drop(host, "qa")
            return "waited" if host.sleep(OOM_WAIT_S) else "stopped"
        if code in LOST_STATE:
            self.core.residency.forget("qa")
            return self._clip_retry(run, exc)
        raise NarrationError(
            codes.INTERNAL,
            f"the QA worker could not embed the voice's clip: {exc.message}",
            details={"worker_code": code},
            retryable=True,
            retry_after_s=RETRY_AFTER_S,
            hint="Send the same request again; what was made is kept. If it fails again, the daemon's log has the "
            "details.",
        ) from exc

    @staticmethod
    def _clip_retry(run: AuditionRun, exc: Exception) -> Outcome:
        if run.clip_retries >= MAX_RETRIES:
            raise NarrationError(
                codes.INTERNAL,
                f"the QA worker failed twice while embedding the voice's clip: {exc}",
                details={"exception": type(exc).__name__},
                retryable=True,
                retry_after_s=RETRY_AFTER_S,
                hint="Send the same request again; what was made is kept. If it fails again, the daemon's log has "
                "the details.",
            ) from exc
        run.clip_retries += 1
        log.warning("job %s: the QA worker failed on the clip (%s); once more", run.job_id, exc)
        return "worked"

    def close(self) -> None:
        """Stop the thread that renews the leases (shared with the job engine; ``Registry.close``)."""
        self.core.leases.close()


# ---------------------------------------------------------------------- helpers


def _audition_run(run: JobRun) -> AuditionRun:
    if not isinstance(run, AuditionRun):
        raise TypeError(f"job {run.job_id} is not an audition")
    return run


def _voice(request: Any) -> VoiceSpec:
    """The request's voice (``audition_pronunciation``'s input schema: ``voice: {path, sha256, transcript}``)."""
    try:
        voice = request["voice"]
        values = {name: voice[name] for name in ("path", "sha256", "transcript")}
    except (KeyError, TypeError) as exc:
        raise NarrationError(codes.INTERNAL, f"the job's request cannot be read: {exc}", retryable=False) from exc
    if not all(isinstance(v, str) and v for v in values.values()):
        raise NarrationError(codes.INTERNAL, "the job's request has a malformed voice", retryable=False)
    return VoiceSpec(path=values["path"], sha256=values["sha256"], transcript=values["transcript"])


def base_profile(host: RunnerHost) -> EngineProfile:
    """The pinned Base engine profile, which clones the voice; ``BACKEND_NOT_INSTALLED`` when none is pinned, or
    when it lacks a setting a render needs (every audio-changing setting is passed explicitly, section 10.1)."""
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


def _measured_error(host: RunnerHost) -> MeasuredError | None:
    """The aligner's measured error from the current alignment benchmark, as ``JobEngine.open`` reads it."""
    bench = host.store.current_alignment_benchmark()
    if bench is None:
        return None
    return MeasuredError(
        p50_s=bench.measured_error.p50_s,
        p95_s=bench.measured_error.p95_s,
        n=bench.measured_error.n,
        benchmark=bench.benchmark.id,
        by_kind=dict(bench.by_kind),
    )


def build_audition_handler(engine: JobEngine) -> AuditionHandler:
    """The ``pronunciation`` kind's handler, for the runner's ``Registry`` (``narration.engine.installed``). It
    shares the daemon's job ``engine``'s core: the model residency, the throughput estimate and the canary."""
    return AuditionHandler(engine)


__all__ = [
    "KIND",
    "RETRY_AFTER_S",
    "SIMILARITY",
    "AuditionHandler",
    "AuditionRun",
    "AuditionStages",
    "AuditionView",
    "base_profile",
    "build_audition_handler",
    "clip_similarity",
]
