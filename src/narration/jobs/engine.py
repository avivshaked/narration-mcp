"""The job engine (design sections 4, 8, 10 and 11; plan.md WP31): rounds of render, post-process and score.

``JobEngine`` runs one generation or scoring job (kinds ``generate`` and ``analyse``) a piece at a time, so
the daemon can stop, cancel or preempt it between pieces (``runner.EngineRunner`` drives it through the
daemon's seam). A piece is one render, one post-processing or one scoring, or one bounded wait.

**Rounds** (section 4 item 4, section 8). Round 0 takes each segment's requested attempts, one per take slot.
A round renders all of its attempts on Qwen, post-processes them on the CPU, then scores them on the QA
group, so each round loads each model group at most once. When a round is settled, every slot whose take is
a retake trigger gets one retake, on the next attempt number above every attempt of its segment so far, in
slot order, up to ``max_retakes`` per slot. A take is a retake trigger when one of its QA flags is, by the
contract's rule (``codes.is_retake_trigger``: any fail, ``CUE_UNALIGNED`` except DC-12's
``no_alignable_words``, ``HEAD_INSERTION``). So a job has at most ``1 + max_retakes`` rounds.

**The cache answers first** (sections 4 item 6, 10.2). Every attempt's keys follow from the request and the
service's pins alone: the seed and the ``render_key`` from the voice, the engine text and the attempt number;
the ``delivery_key`` from the raw audio's sha256 and the delivery profile; the ``analysis_key`` from the
delivery's sha256 and the request's inputs for it. Each layer is looked up when an attempt is planned, and
again at claim time. A result that exists is used. One that another holder is producing is waited for (its
lease), and never produced twice. Otherwise this engine takes the lease, does the work and publishes it. A
resubmitted request walks the same attempts, retakes included, and finds them all in the cache, so nothing
is retaken again.

**Verdicts** come from ``QaScorer.score`` alone, on the take and the request's inputs for it. The suggestion
and the per-job consistency report come after every take is scored, and never change a verdict (section
11.1).

**Failures.** A job-level problem raises ``NarrationError`` for the runner to record on the job: no engine
pinned, the engine changed, the voice is not measured or its clip changed, the VRAM wait timed out, a
worker that cannot be installed or loaded. A take-level execution problem is flagged on its segment
(severity ``error``), and the job goes on with the rest: ``GPU_OOM`` after one retry, a worker that crashed
twice (``WORKER_CRASHED``), ``RENDER_FAILED``, ``QA_UNAVAILABLE``.
"""

from __future__ import annotations

import dataclasses
import errno
import logging
import shutil
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

import numpy as np
import soundfile

from narration import keys
from narration.config import Config
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError, QaUnavailable, WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.interfaces import (
    AlignerCore,
    AnalysisKeyInputs,
    DeliveryProcessor,
    QaInputs,
    QaScorer,
    ScoredTake,
    TextPlanner,
)
from narration.contracts.models import (
    AnalysisRecord,
    AnalysisText,
    AnalysisVersions,
    CanaryRecord,
    Consistency,
    CueText,
    DeliveryAudio,
    EngineProfile,
    Flag,
    Hint,
    JobAttempt,
    JobRecord,
    JobSegment,
    Licence,
    MeasuredError,
    MeasurementRecord,
    Progress,
    RawAudio,
    RenderEngine,
    RenderRecord,
    RenderVoice,
    SegmentText,
    Suggestion,
    TakeRecord,
)
from narration.contracts.names import CanaryStatus, GpuHolder, JobPhase, SegmentState
from narration.contracts.serial import to_json
from narration.contracts.worker import AlignReply, AsrWord, HelloReply
from narration.text import segment_too_long

from .admission import Throughput
from .gpu import GroupNeed, NoProbe, Residency, VramProbe
from .hooks import EngineGuard, NoGuard
from .host import RunnerHost
from .pins import ProfileError, QaPins, call_cap, generation, qwen_load_payload
from .plan import GenerateRequest, RequestError, estimated_audio_s, hints_used, next_attempt, requested_attempts
from .voice import PathCheck, stage_clip

log = logging.getLogger(__name__)

KINDS: Final = ("generate", "analyse")
"""The job kinds this engine runs. ``analyse`` is scoring only: its takes are cached, its analyses are not.
Both run the same pipeline, and the cache decides which layers are made."""
RENDER_LEASE_S: Final = 3600.0
POST_LEASE_S: Final = 600.0
SCORE_LEASE_S: Final = 3600.0
"""Leases outlive the work they cover. A daemon that dies leaves them to lapse; its successor, under the same
holder name, may claim them again at once."""
DEFER_S: Final = 2.0
"""How long work that another holder is producing is left before it is looked at again."""
OOM_WAIT_S: Final = 15.0
"""Section 4 item 5: on an out-of-memory error, unload, wait this long, and retry once."""
MAX_RETRIES: Final = 1
"""Retries of one piece of work after an out-of-memory error, a crash or a timeout of its worker."""
PREPARE_TIMEOUT_S: Final = 300.0
QA_TIMEOUT_S: Final = 900.0
FRAMES_PER_SECOND: Final = 12.5
"""Qwen3-TTS-12Hz's codec: a call capped at N tokens makes at most N / 12.5 s of audio."""
SCRATCH_JOBS: Final = "jobs"
"""Each job's working files: ``scratch/jobs/<job_id>/``, removed when the job ends or is given back."""

Stage = Literal["render", "post", "score", "done", "error"]
Outcome = Literal["worked", "waited", "stopped", "finished"]
_STAGE_ORDER: Final[dict[Stage, int]] = {"render": 0, "post": 1, "score": 2, "done": 3, "error": 3}
_STAGE_DONE: Final[dict[Stage, float]] = {"render": 0.0, "post": 1 / 3, "score": 2 / 3, "done": 1.0, "error": 1.0}
_VERDICT_STATE: Final[dict[str, SegmentState]] = {"pass": "passed", "warn": "warned", "fail": "failed_qa"}
_VERDICT_RANK: Final[dict[str, int]] = {"pass": 0, "warn": 1, "fail": 2}


class Scorer(QaScorer, Protocol):
    """The QA scorer, with the version of its number reader, which the analysis key names."""

    @property
    def number_reader(self) -> str: ...


# ======================================================================== the job's state


@dataclass(slots=True, eq=False)
class Attempt:
    """One attempt of one take slot: its keys, and what the cache or this job produced for it, or its error."""

    segment: int
    slot: int
    attempt: int
    round: int
    seed: int
    render_key: str
    render: RenderRecord | None = None
    take: TakeRecord | None = None
    analysis_key: str | None = None
    analysis: AnalysisRecord | None = None
    error: Flag | None = None
    retries: int = 0
    deferred_until: float = 0.0

    @property
    def stage(self) -> Stage:
        """The next layer to make (``render``, ``post``, ``score``), or ``done``, or ``error``."""
        if self.error is not None:
            return "error"
        if self.render is None:
            return "render"
        if self.take is None:
            return "post"
        if self.analysis is None:
            return "score"
        return "done"

    @property
    def settled(self) -> bool:
        return self.stage in ("done", "error")


@dataclass(slots=True, eq=False)
class SegmentWork:
    """One segment of the job: its text, the hints it uses, its take slots (each a list of attempts, the
    current one last), its segment-level flags and, once decided, its final state and suggestion."""

    index: int
    text: SegmentText
    hints: tuple[Hint, ...]
    est_s: float
    slots: list[list[Attempt]] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    state: SegmentState | None = None
    suggested: str | None = None
    suggestion: Suggestion | None = None

    @property
    def segment_id(self) -> str:
        return self.text.segment_id

    def attempts(self) -> Iterator[Attempt]:
        for slot in self.slots:
            yield from slot

    def current(self) -> list[Attempt]:
        return [slot[-1] for slot in self.slots]

    def used(self) -> set[int]:
        return {a.attempt for a in self.attempts()}


@dataclass(slots=True, eq=False)
class JobRun:
    """A job the engine holds, planned from its request and the cache."""

    job: JobRecord
    request: GenerateRequest
    voice_hash: str
    clip: Path
    profile: EngineProfile
    measurement: MeasurementRecord
    measured_error: MeasuredError | None
    scratch: Path
    segments: list[SegmentWork] = field(default_factory=list)
    fresh_keys: set[str] = field(default_factory=set)
    """Render keys this job rendered, including before it was given back and taken again."""
    done_floor: float = 0.0
    round: int = 0
    phase: JobPhase | None = None
    message: str | None = None
    prepared: int | None = None
    """The pid of the Qwen worker the voice was prepared in since its last load, or None."""
    consistency: Consistency | None = None
    outliers: dict[str, Flag] = field(default_factory=dict)
    """``SPK_OUTLIER`` flags by take id, from the consistency report."""

    @property
    def job_id(self) -> str:
        return self.job.job_id


# ======================================================================== the engine


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineParts:
    """What the engine is built from: the service's text pipeline, post-processing, QA scorer and aligner; the
    QA group's pins; the engine guard (WP32's canary gate); the VRAM probe; the path check for a caller's
    clip; a monotonic clock; and how long to leave work another holder is producing (``defer_s``)."""

    text: TextPlanner
    delivery: DeliveryProcessor
    scorer: Scorer
    aligner: AlignerCore
    qa_pins: QaPins
    guard: EngineGuard = field(default_factory=NoGuard)
    probe: VramProbe = field(default_factory=NoProbe)
    check_path: PathCheck | None = None
    clock: Callable[[], float] = time.monotonic
    defer_s: float = DEFER_S


class JobEngine:
    """Plans and advances generation jobs (see the module docstring). One instance serves one daemon: it keeps
    the model residency between jobs, so work that needs the resident group starts without a load."""

    def __init__(self, config: Config, parts: EngineParts) -> None:
        self.config = config
        self.parts = parts
        self.residency = Residency(gpu=config.gpu, probe=parts.probe, clock=parts.clock)
        self.residency.need_mb["qa"] = parts.qa_pins.vram_need_mb
        self.throughput = Throughput()
        self.canary: CanaryStatus = "not_run"
        """The canary outcome of the Qwen load in use, which every render made on it records."""

    # ------------------------------------------------------------------ planning
    def open(self, host: RunnerHost, job: JobRecord) -> JobRun:
        """Plan a claimed job from its request and the cache. Raises ``NarrationError`` for a job-level problem."""
        if job.kind not in KINDS:
            raise NarrationError(
                codes.INTERNAL,
                f"this build of the service does not run {job.kind} jobs",
                retryable=False,
                hint=f"The {job.kind} job engine is not in this build; nothing was made. Update the service.",
            )
        store, config = host.store, self.config
        try:
            request = GenerateRequest.parse(job.request, config.defaults)
        except RequestError as exc:
            raise NarrationError(codes.INTERNAL, str(exc), retryable=False) from exc
        profile = store.current_engine_profile("base")
        if profile is None:
            raise NarrationError(
                codes.BACKEND_NOT_INSTALLED,
                "no engine profile is pinned for the Base clone path",
                hint="Ask the operator to run narration-admin install, then narration-admin engine pin.",
            )
        if request.expect_engine_profile is not None and request.expect_engine_profile != profile.hash:
            raise NarrationError(
                codes.ENGINE_CHANGED,
                f"the job expects engine profile {request.expect_engine_profile}; the service's is {profile.hash}",
                field="expect_engine_profile",
                details={"expected": request.expect_engine_profile, "current": profile.hash},
            )
        try:
            call_cap(profile, "x")
            qwen_load_payload(profile, config.gpu.device)
        except ProfileError as exc:
            raise NarrationError(
                codes.BACKEND_NOT_INSTALLED, str(exc), hint="Ask the operator to pin the engine again."
            ) from exc
        voice_hash = keys.voice_hash(
            model=names.MODEL_QWEN_BASE,
            clip_sha256=request.voice.sha256,
            transcript=request.voice.transcript,
            language=names.LANGUAGE,
            x_vector_only_mode=config.engines.qwen3_base.x_vector_only_mode,
        )
        measurement = store.get_measurement(voice_hash, profile.engine_profile_id)
        if measurement is None:
            raise NarrationError(
                codes.VOICE_NOT_MEASURED,
                f"voice {voice_hash} has no measurement under engine profile {profile.engine_profile_id}",
                field="voice",
            )
        store.touch("measurement", measurement.measurement_key)
        clip = stage_clip(store, request.voice, check_path=self.parts.check_path)
        planned = self.parts.text.plan_request(request.segments, request.hints, strict_text=False)
        bench = store.current_alignment_benchmark()
        measured = (
            MeasuredError(
                p50_s=bench.measured_error.p50_s,
                p95_s=bench.measured_error.p95_s,
                n=bench.measured_error.n,
                benchmark=bench.benchmark.id,
                by_kind=dict(bench.by_kind),
            )
            if bench is not None
            else None
        )
        run = JobRun(
            job=job,
            request=request,
            voice_hash=voice_hash,
            clip=clip,
            profile=profile,
            measurement=measurement,
            measured_error=measured,
            scratch=store.scratch_path(SCRATCH_JOBS, job.job_id, "work").parent,
            fresh_keys={a.render_key for s in job.items for a in s.attempts if a.fresh},
            done_floor=job.progress.done_s,
        )
        self.residency.need_mb["qwen"] = profile.vram_need_mb
        for index, (segment_in, text) in enumerate(zip(request.segments, planned, strict=True)):
            seg = SegmentWork(
                index=index,
                text=text,
                hints=hints_used(text, request.hints),
                est_s=estimated_audio_s(text, measurement),
                flags=[*text.warnings, *(w for cue in text.cues for w in cue.warnings)],
            )
            if measurement.max_segment_chars is not None:
                too_long = segment_too_long(
                    text,
                    max_segment_chars=measurement.max_segment_chars,
                    max_segment_seconds=measurement.max_segment_seconds,
                )
                if too_long is not None:
                    seg.flags.append(too_long)  # warned about, never refused: it is rendered like any other
            run.segments.append(seg)
            for slot, number in enumerate(requested_attempts(segment_in, request.takes)):
                seg.slots.append([self._attempt(host, run, seg, slot, number, 0)])
        run.message = f"planned {len(run.segments)} segment(s) on {profile.engine_profile_id}"
        log.info("job %s: %s", job.job_id, run.message)
        return run

    def _attempt(self, host: RunnerHost, run: JobRun, seg: SegmentWork, slot: int, number: int, round_: int) -> Attempt:
        """A new attempt, its keys from the request alone, looked up in the cache layer by layer."""
        engine_text = seg.text.engine_text
        seed = keys.seed(voice_hash=run.voice_hash, engine_text=engine_text, attempt=number)
        render_key = keys.render_key(
            engine_profile_hash=run.profile.hash, voice_hash=run.voice_hash, engine_text=engine_text, seed=seed
        )
        attempt = Attempt(segment=seg.index, slot=slot, attempt=number, round=round_, seed=seed, render_key=render_key)
        found = host.store.get_render(render_key)
        if found is not None:
            host.store.touch("render", found.render_id)
            attempt.render = found
            self._find_take(host, run, seg, attempt)
        return attempt

    def _delivery_key(self, render: RenderRecord) -> str:
        return keys.delivery_key(
            raw_sha256=render.raw.sha256, profile=self.config.delivery, tools=self.parts.delivery.tools
        )

    def _find_take(self, host: RunnerHost, run: JobRun, seg: SegmentWork, attempt: Attempt) -> None:
        assert attempt.render is not None
        take = host.store.get_take(self._delivery_key(attempt.render))
        if take is None:
            return
        attempt.take = take
        self._find_analysis(host, run, seg, attempt)

    def _find_analysis(self, host: RunnerHost, run: JobRun, seg: SegmentWork, attempt: Attempt) -> None:
        assert attempt.take is not None
        attempt.analysis_key = keys.analysis_key(self._key_inputs(run, seg, attempt.take))
        found = host.store.get_analysis(attempt.analysis_key)
        if found is not None:
            host.store.touch("analysis", found.analysis_id)  # also restarts its take's retention
            attempt.analysis = found
        else:
            host.store.touch("take", attempt.take.take_id)

    def _key_inputs(self, run: JobRun, seg: SegmentWork, take: TakeRecord) -> AnalysisKeyInputs:
        """The analysis key's inputs (section 10.2): the take, the request's inputs for it, the service's pins."""
        text = seg.text
        return AnalysisKeyInputs(
            delivery_sha256=take.delivery.sha256,
            spoken_text=text.spoken_text,
            cue_spans=tuple(c.spoken_span for c in text.cues),
            exact_spans=tuple((c.index, e.words[0], e.words[1]) for c in text.cues for e in c.exact),
            hints_qa=tuple((h.term, h.asr_aliases, h.align_as) for h in seg.hints),
            qa_profile=self.parts.scorer.profile_version,
            text_checks_version=(text.text_checks or self.parts.text.checks_info).version,
            number_reader=self.parts.scorer.number_reader,
            asr_model=self.parts.qa_pins.asr.name,
            sv_model=self.parts.qa_pins.sv.name,
            aligner_method_id=self.parts.aligner.method_id,
            measurement_key=run.measurement.measurement_key,
        )

    # ------------------------------------------------------------------ advancing
    def advance(self, host: RunnerHost, run: JobRun) -> Outcome:
        """Do one piece of the job's work, or one bounded wait; ``finished`` once the job is complete and its
        record written. Raises ``NarrationError`` for a job-level failure."""
        now = self.parts.clock()
        pending = [a for seg in run.segments for a in seg.current() if not a.settled]
        ready = [a for a in pending if a.deferred_until <= now]
        steps: tuple[tuple[Stage, GpuHolder | None], ...] = (("render", "qwen"), ("post", None), ("score", "qa"))
        for stage, group in steps:
            if stage == "score" and any(a.stage == "render" for a in pending):
                break  # a round renders everything before it loads QA, so each group loads once a round
            todo = next((a for a in ready if a.stage == stage), None)
            if todo is not None:
                return self._guarded(host, run, todo, group)
        if pending:
            return self._wait_deferred(host, run, pending)
        if self._start_retakes(host, run):
            return "worked"
        self.finish(host, run)
        return "finished"

    def _wait_deferred(self, host: RunnerHost, run: JobRun, pending: list[Attempt]) -> Outcome:
        """Everything left is being produced by another holder, or waits for what is: wait a little for the
        first piece in flight elsewhere (never for work that merely waits its turn, which nobody is making)."""
        deferred = [a for a in pending if a.deferred_until > 0.0] or pending
        first = min(deferred, key=lambda a: (a.deferred_until, _STAGE_ORDER[a.stage]))
        key = self._key_of(first)
        run.message = f"waiting for {self._label(run, first)}, which another job is making"
        if key is not None:
            host.store.wait_for(key, timeout_s=self.parts.defer_s)
        elif not host.sleep(self.parts.defer_s):
            return "stopped"
        for attempt in pending:
            attempt.deferred_until = 0.0
        return "stopped" if host.should_stop() else "waited"

    def _key_of(self, attempt: Attempt) -> str | None:
        if attempt.stage == "render":
            return attempt.render_key
        if attempt.stage == "post" and attempt.render is not None:
            return self._delivery_key(attempt.render)
        if attempt.stage == "score":
            return attempt.analysis_key
        return None

    def _start_retakes(self, host: RunnerHost, run: JobRun) -> bool:
        """At the end of a settled round: one retake for every slot whose take is a retake trigger, on the next
        attempt numbers in slot order, up to ``max_retakes`` per slot (section 8). False when there is none."""
        created = 0
        for seg in run.segments:
            used = seg.used()
            for slot_index, slot in enumerate(seg.slots):
                current = slot[-1]
                if current.stage != "done" or not is_retake_trigger(current) or len(slot) > run.request.max_retakes:
                    continue
                number = next_attempt(used)
                used.add(number)
                slot.append(self._attempt(host, run, seg, slot_index, number, run.round + 1))
                created += 1
        if not created:
            return False
        run.round += 1
        self._phase(host, run, "retaking")
        run.message = f"round {run.round}: {created} retake(s) of take slots that failed QA"
        log.info("job %s: %s", run.job_id, run.message)
        return True

    # ------------------------------------------------------------------ one piece of work, guarded
    def _guarded(self, host: RunnerHost, run: JobRun, attempt: Attempt, group: GpuHolder | None) -> Outcome:
        """Run one piece of work; turn a worker's failure into a retry, a flag, or a job-level error."""
        work = {"render": self._render, "post": self._post, "score": self._score}[attempt.stage]
        try:
            return work(host, run, attempt)
        except WorkerFailure as exc:
            if host.should_stop():
                return "stopped"
            if exc.code == "BACKEND_NOT_INSTALLED":
                raise NarrationError(codes.BACKEND_NOT_INSTALLED, exc.message, details=exc.details) from exc
            if exc.code == "GPU_OOM" and group is not None:
                return self._oom(host, run, attempt, group, exc)
            if exc.code in ("NOT_LOADED", "VOICE_NOT_PREPARED") and group is not None:
                return self._lost_state(run, attempt, group, exc)
            code = codes.QA_UNAVAILABLE if group == "qa" else codes.RENDER_FAILED
            self._fail_attempt(run, attempt, code, f"the worker could not {_verb(group)} it: {exc}", exc.code)
            return "worked"
        except (WorkerCrashed, WorkerTimeout) as exc:
            if host.stop_mode == "now" or host.should_stop():
                return "stopped"  # the daemon killed the workers: the job goes back to the queue as it is
            if group is not None:
                self.residency.forget(group)
            if group == "qwen":
                run.prepared = None
            if attempt.retries < MAX_RETRIES:
                attempt.retries += 1
                log.warning(
                    "job %s: the %s worker failed on %s (%s); once more",
                    run.job_id,
                    group,
                    self._label(run, attempt),
                    exc,
                )
                return "worked"
            self._fail_attempt(
                run, attempt, codes.WORKER_CRASHED, f"the {group} worker failed twice on it: {exc}", None
            )
            return "worked"
        except QaUnavailable as exc:
            self._fail_attempt(run, attempt, codes.QA_UNAVAILABLE, f"QA could not score it: {exc}", None)
            return "worked"
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise NarrationError(codes.STORE_FULL, f"the store's disk is full: {exc}", retry_after_s=60.0) from exc
            log.exception("job %s: %s failed", run.job_id, self._label(run, attempt))
            code = codes.QA_UNAVAILABLE if group == "qa" else codes.RENDER_FAILED
            self._fail_attempt(run, attempt, code, f"a file could not be read or written: {exc}", None)
            return "worked"
        except (ValueError, RuntimeError) as exc:  # unreadable audio (soundfile), audio post-processing refuses
            log.exception("job %s: %s failed", run.job_id, self._label(run, attempt))
            code = codes.QA_UNAVAILABLE if group == "qa" else codes.RENDER_FAILED
            self._fail_attempt(
                run, attempt, code, f"it could not be {'scored' if group == 'qa' else 'made'}: {exc}", None
            )
            return "worked"

    def _oom(self, host: RunnerHost, run: JobRun, attempt: Attempt, group: GpuHolder, exc: WorkerFailure) -> Outcome:
        """Section 4 item 5: unload, wait, retry once; then the take slot fails with ``GPU_OOM``."""
        if attempt.retries >= MAX_RETRIES:
            self._fail_attempt(
                run, attempt, codes.GPU_OOM, f"out of GPU memory after one retry: {exc.message}", exc.code
            )
            return "worked"
        attempt.retries += 1
        log.warning(
            "job %s: out of GPU memory on %s; unloading, waiting, once more", run.job_id, self._label(run, attempt)
        )
        try:
            self.residency.unload(host, group)
        except (WorkerFailure, WorkerCrashed, WorkerTimeout) as unload_exc:
            log.warning("unloading the %s group after running out of memory failed: %s", group, unload_exc)
            self.residency.forget(group)
        if group == "qwen":
            run.prepared = None
        return "waited" if host.sleep(OOM_WAIT_S) else "stopped"

    def _lost_state(self, run: JobRun, attempt: Attempt, group: GpuHolder, exc: WorkerFailure) -> Outcome:
        """The worker lost its models or the prepared voice (unloaded meanwhile): load again, once."""
        self.residency.forget(group)
        run.prepared = None
        if attempt.retries >= MAX_RETRIES:
            code = codes.QA_UNAVAILABLE if group == "qa" else codes.RENDER_FAILED
            self._fail_attempt(run, attempt, code, f"the worker lost its models twice: {exc}", exc.code)
            return "worked"
        attempt.retries += 1
        return "worked"

    def _fail_attempt(self, run: JobRun, attempt: Attempt, code: str, why: str, worker_code: str | None) -> None:
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

    # ------------------------------------------------------------------ making a model group ready
    def _qwen_need(self, run: JobRun) -> GroupNeed:
        profile = run.profile
        return GroupNeed(
            group="qwen",
            key=profile.hash,
            label=profile.engine_profile_id,
            payload=qwen_load_payload(profile, self.config.gpu.device),
            need_mb=profile.vram_need_mb,
            cublas_workspace_config=profile.determinism.cublas_workspace_config,
        )

    def _qa_need(self) -> GroupNeed:
        pins = self.parts.qa_pins
        return GroupNeed(
            group="qa",
            key=pins.key,
            label="the QA models",
            payload=pins.load_payload(self.config.gpu.device),
            need_mb=pins.vram_need_mb,
        )

    def _ready(self, host: RunnerHost, run: JobRun, need: GroupNeed) -> bool:
        """Make the group ready; False while waiting for free VRAM. After a Qwen load, the engine guard runs.

        A load the worker refuses is a job-level failure (the models as pinned cannot run), except running out
        of memory, which is retried (``_oom``)."""
        try:
            state = self.residency.ensure(host, need, phase=lambda p: self._phase(host, run, p))
        except WorkerFailure as exc:
            if exc.code == "GPU_OOM":
                raise
            code = codes.BACKEND_NOT_INSTALLED if exc.code == "BACKEND_NOT_INSTALLED" else codes.INTERNAL
            raise NarrationError(
                code,
                f"the {need.group} worker could not load {need.label}: {exc.message}",
                details={"worker_code": exc.code, **exc.details},
                retryable=False,
                hint="Run narration-admin doctor; the models as pinned could not be loaded.",
            ) from exc
        if state == "waiting":
            run.message = f"waiting for free GPU memory to load {need.label}"
            return False
        if state == "loaded" and need.group == "qwen":
            run.prepared = None
            client = host.workers.client("qwen", cublas_workspace_config=need.cublas_workspace_config)
            hello = cast(HelloReply | None, getattr(client, "hello", None))
            self._phase(host, run, "canary")
            self.canary = self.parts.guard.after_load(host, run.profile, hello)
        return True

    # ------------------------------------------------------------------ render
    def _render(self, host: RunnerHost, run: JobRun, attempt: Attempt) -> Outcome:
        store, seg = host.store, run.segments[attempt.segment]
        found = store.get_render(attempt.render_key)
        if found is None:
            need = self._qwen_need(run)
            if not self._ready(host, run, need):
                return "waited" if not host.should_stop() else "stopped"
            status, lease = store.claim(attempt.render_key, host.holder, ttl_s=RENDER_LEASE_S)
            if status == "in_flight":
                attempt.deferred_until = self.parts.clock() + self.parts.defer_s
                return "worked"
            if status == "claimed" and lease is not None:
                try:
                    found = self._synthesize(host, run, seg, attempt, need)
                finally:
                    lease.release()
            else:
                found = store.get_render(attempt.render_key)
                if found is None:
                    return "worked"  # published and gone again (collected): looked up afresh next step
        attempt.render = found
        self._find_take(host, run, seg, attempt)  # a render made before may already have its take
        return "worked"

    def _synthesize(
        self, host: RunnerHost, run: JobRun, seg: SegmentWork, attempt: Attempt, need: GroupNeed
    ) -> RenderRecord:
        client = host.workers.client("qwen", cublas_workspace_config=need.cublas_workspace_config)
        if run.prepared is None or run.prepared != client.pid:
            client.request(
                "prepare_voice",
                {
                    "voice_hash": run.voice_hash,
                    "ref_wav": str(run.clip),
                    "ref_text": run.request.voice.transcript,
                    "x_vector_only_mode": self.config.engines.qwen3_base.x_vector_only_mode,
                },
                timeout_s=PREPARE_TIMEOUT_S,
            )
            run.prepared = client.pid
        self._phase(host, run, "rendering" if attempt.round == 0 else "retaking")
        run.message = f"round {attempt.round}: rendering {self._label(run, attempt)}"
        engine_text = seg.text.engine_text
        cap = call_cap(run.profile, engine_text)
        render_id = keys.render_id(attempt.render_key)
        out = run.scratch / f"{render_id}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        started = self.parts.clock()
        reply = client.request(
            "synthesize",
            {
                "voice_hash": run.voice_hash,
                "engine_text": engine_text,
                "language": names.LANGUAGE,
                "seed": attempt.seed,
                "max_new_tokens": cap,
                "out_path": str(out),
            },
            timeout_s=max(300.0, cap / FRAMES_PER_SECOND * 5.0),
        )
        samples, rate, gen_s = int(reply["samples"]), int(reply["sample_rate"]), float(reply["gen_s"])
        audio_s = samples / rate if rate else 0.0
        hello = cast(HelloReply | None, getattr(client, "hello", None))
        record = RenderRecord(
            render_id=render_id,
            render_key=attempt.render_key,
            voice=RenderVoice(
                voice_hash=run.voice_hash,
                clip_sha256=run.request.voice.sha256,
                x_vector_only_mode=self.config.engines.qwen3_base.x_vector_only_mode,
            ),
            engine=RenderEngine(
                engine_profile_id=run.profile.engine_profile_id,
                engine_profile_hash=run.profile.hash,
                model_repo=run.profile.model_repo,
                model_revision=run.profile.model_revision,
                non_streaming_mode=bool(run.profile.settings["non_streaming_mode"]),
                generation={**generation(run.profile), "max_new_tokens": int(reply.get("max_new_tokens", cap))},
                observed=_observed(hello),
            ),
            engine_text=engine_text,
            seed=attempt.seed,
            attempt=attempt.attempt,
            raw=RawAudio(path=str(out), sha256="", sample_rate=rate, samples=samples),
            hit_token_cap=bool(reply["hit_token_cap"]),
            gen_s=gen_s,
            rtf=round(gen_s / audio_s, 4) if audio_s else 0.0,
            canary=CanaryRecord(batch_status=self.canary),
            licence=Licence(generation_model=run.profile.licence, voice_clip="synthetic"),
        )
        published = host.store.put_render(record, out)
        run.fresh_keys.add(attempt.render_key)
        self.throughput.record(self.parts.clock() - started, seg.est_s / 3)
        return published

    # ------------------------------------------------------------------ post-process
    def _post(self, host: RunnerHost, run: JobRun, attempt: Attempt) -> Outcome:
        store, seg, render = host.store, run.segments[attempt.segment], attempt.render
        assert render is not None
        delivery_key = self._delivery_key(render)
        found = store.get_take(delivery_key)
        if found is None:
            status, lease = store.claim(delivery_key, host.holder, ttl_s=POST_LEASE_S)
            if status == "in_flight":
                attempt.deferred_until = self.parts.clock() + self.parts.defer_s
                return "worked"
            if status == "claimed" and lease is not None:
                try:
                    found = self._deliver(host, run, seg, attempt, render, delivery_key)
                finally:
                    lease.release()
            else:
                found = store.get_take(delivery_key)
                if found is None:
                    return "worked"
        attempt.take = found
        self._find_analysis(host, run, seg, attempt)
        return "worked"

    def _deliver(
        self, host: RunnerHost, run: JobRun, seg: SegmentWork, attempt: Attempt, render: RenderRecord, delivery_key: str
    ) -> TakeRecord:
        self._phase(host, run, "postprocessing")
        run.message = f"round {attempt.round}: post-processing {self._label(run, attempt)}"
        take_id = keys.take_id(delivery_key)
        out = run.scratch / f"{take_id}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        started = self.parts.clock()
        made = self.parts.delivery.process(Path(render.raw.path), out, self.config.delivery)
        record = TakeRecord(
            take_id=take_id,
            delivery_key=delivery_key,
            render_id=render.render_id,
            delivery=DeliveryAudio(
                path=str(out), sha256="", sample_rate=made.sample_rate, samples=made.samples, duration_s=made.duration_s
            ),
            trim=made.trim,
            loudness=made.loudness,
            tools=self.parts.delivery.tools,
            flags=made.flags,
        )
        published = host.store.put_take(record, out)
        self.throughput.record(self.parts.clock() - started, seg.est_s / 3)
        return published

    # ------------------------------------------------------------------ score
    def _score(self, host: RunnerHost, run: JobRun, attempt: Attempt) -> Outcome:
        store, seg = host.store, run.segments[attempt.segment]
        take, render, key = attempt.take, attempt.render, attempt.analysis_key
        assert take is not None and render is not None and key is not None
        found = store.get_analysis(key)
        if found is None:
            if not self._ready(host, run, self._qa_need()):
                return "waited" if not host.should_stop() else "stopped"
            status, lease = store.claim(key, host.holder, ttl_s=SCORE_LEASE_S)
            if status == "in_flight":
                attempt.deferred_until = self.parts.clock() + self.parts.defer_s
                return "worked"
            if status == "claimed" and lease is not None:
                try:
                    self._phase(host, run, "scoring")
                    run.message = f"round {attempt.round}: scoring {self._label(run, attempt)}"
                    started = self.parts.clock()
                    found = store.put_analysis(self._analyse(host, run, seg, take, render, key))
                    self.throughput.record(self.parts.clock() - started, seg.est_s / 3)
                finally:
                    lease.release()
            else:
                found = store.get_analysis(key)
                if found is None:
                    return "worked"
        attempt.analysis = found
        return "worked"

    def _analyse(
        self, host: RunnerHost, run: JobRun, seg: SegmentWork, take: TakeRecord, render: RenderRecord, key: str
    ) -> AnalysisRecord:
        """Section 11.1 on the delivery file: ASR, the speaker embedding, the cue alignment, the signal facts,
        then the verdict. The worker returns raw outputs only; every verdict is the scorer's (plan.md P1)."""
        client = host.workers.client("qa")
        wav = take.delivery.path
        heard = client.request(
            "transcribe",
            {"wav": wav, "language": names.LANGUAGE, "word_timestamps": True, "long_form": True},
            timeout_s=QA_TIMEOUT_S,
        )
        device = "cpu" if self.config.gpu.device == "cpu" else "cuda"
        embedded = client.request("embed", {"wav": wav, "device": device}, timeout_s=QA_TIMEOUT_S)
        aligner = self.parts.aligner
        transcript = aligner.build_transcript(seg.text, seg.hints)
        reply: AlignReply | None = None
        if any(ch.isalpha() for token in transcript.tokens for ch in token):  # no letter: nothing to place
            try:
                reply = cast(
                    AlignReply,
                    client.request("align", {"wav": wav, "tokens": list(transcript.tokens)}, timeout_s=QA_TIMEOUT_S),
                )
            except WorkerFailure as exc:
                if exc.code != "ALIGNMENT_ERROR":
                    raise
                log.info("job %s: the aligner could not align %s: %s", run.job_id, take.take_id, exc.message)
        audio, rate = soundfile.read(wav, dtype="float32", always_2d=True)
        samples = np.asarray(audio, dtype=np.float32).mean(axis=1).astype(np.float32)
        asr_words = tuple(cast(list[AsrWord], heard.get("words") or []))
        alignment = aligner.resolve(transcript, reply, samples, int(rate), asr_words, run.measured_error)
        signal = self.parts.delivery.signal_stats(Path(render.raw.path), Path(wav))
        embedding = tuple(float(v) for v in embedded["embedding"])
        qa = self.parts.scorer.score(
            QaInputs(
                segment=seg.text,
                hints=seg.hints,
                voice_transcript=run.request.voice.transcript,
                asr_text=str(heard.get("text") or ""),
                asr_words=asr_words,
                embedding=embedding,
                alignment=alignment,
                signal=signal,
                hit_token_cap=render.hit_token_cap,
                measurement=run.measurement,
            )
        )
        pins = self.parts.qa_pins
        return AnalysisRecord(
            analysis_id=keys.analysis_id(key),
            analysis_key=key,
            take_id=take.take_id,
            text=AnalysisText(
                cues=tuple(_request_free(c) for c in seg.text.cues),
                text_checks=seg.text.text_checks or self.parts.text.checks_info,
                hints_used=seg.hints,
            ),
            versions=AnalysisVersions(
                qa_profile=self.parts.scorer.profile_version,
                asr=pins.asr.name,
                sv=pins.sv.name,
                aligner_method=aligner.method_id,
                number_reader=self.parts.scorer.number_reader,
                measurement=run.measurement.measurement_key,
            ),
            alignment=alignment,
            qa=qa,
            licence=pins.licence,
            embedding=embedding,
        )

    # ------------------------------------------------------------------ the end of a job
    def finish(self, host: RunnerHost, run: JobRun) -> JobRecord | None:
        """Suggest a take per segment, report the request's consistency, decide the outcome, and complete the
        job (section 8). Nothing here changes a verdict. A job cancelled meanwhile ends ``cancelled``, with
        every segment's state. Returns the job as written, or None when its status changed otherwise."""
        self._phase(host, run, "suggesting")
        for seg in run.segments:
            self._decide(seg)
        suggested: list[tuple[str, tuple[float, ...]]] = []
        for seg in run.segments:
            for a in seg.attempts():
                if a.take is not None and a.take.take_id == seg.suggested and a.analysis is not None:
                    if a.analysis.embedding is not None:
                        suggested.append((a.take.take_id, a.analysis.embedding))
                    break
        try:
            consistency, outliers = self.parts.scorer.consistency(suggested, run.measurement)
        except ValueError as exc:
            log.warning("job %s: no consistency report: %s", run.job_id, exc)
            consistency, outliers = Consistency(min=None, median=None), ()
        run.consistency = consistency
        run.outliers = {str((f.details or {}).get("take_id")): f for f in outliers}
        passed = sum(1 for s in run.segments if s.state == "passed")
        outcome = "all_passed" if passed == len(run.segments) else "needs_attention"
        run.message = f"completed: {passed} of {len(run.segments)} segment(s) passed QA"
        updated = host.store.update_job(
            run.job_id,
            expect_status="running",
            status="completed",
            phase=None,
            round=run.round,
            outcome=outcome,
            progress=self.progress(run, complete=True),
            items=self.items(run),
            result=self.result(run),
            message=run.message,
        )
        if updated is None:
            current = host.store.get_job(run.job_id)
            if current is not None and current.status == "cancelling":  # cancelled as it finished: all is made
                run.message = "cancelled as it finished; everything it made is kept in the cache"
                updated = host.store.update_job(
                    run.job_id,
                    expect_status="cancelling",
                    status="cancelled",
                    phase=None,
                    round=run.round,
                    progress=self.progress(run, complete=True),
                    items=self.items(run),
                    message=run.message,
                )
            else:
                log.info("job %s changed while it finished; its new status stands", run.job_id)
        self.release(run)
        return updated

    def result(self, run: JobRun) -> dict[str, Any]:
        """The job's per-job advice, kept with it (``JobRecord.result``): each segment's suggested take and why,
        and the consistency report. Advice only; the caller chooses (section 8)."""
        return {
            "suggestions": [
                {
                    "segment_id": seg.segment_id,
                    "take_id": seg.suggested,
                    "suggestion": to_json(seg.suggestion) if seg.suggestion is not None else None,
                }
                for seg in run.segments
            ],
            "consistency": to_json(run.consistency) if run.consistency is not None else None,
        }

    def _decide(self, seg: SegmentWork) -> None:
        """The segment's suggestion (section 8's tiers, over every take of every slot) and its final state."""
        scored = [
            ScoredTake(
                take_id=a.take.take_id,
                attempt=a.attempt,
                verdict=a.analysis.qa.verdict,
                flags=a.analysis.qa.flags,
                cues=a.analysis.alignment.cues,
            )
            for a in seg.attempts()
            if a.take is not None and a.analysis is not None
        ]
        seg.suggested, seg.suggestion = self.parts.scorer.suggest(scored)
        chosen = next((s for s in scored if s.take_id == seg.suggested), None)
        seg.state = _VERDICT_STATE[chosen.verdict] if chosen is not None else "error"

    def cancel(self, host: RunnerHost, run: JobRun) -> JobRecord | None:
        """Finish a cancelled job: the segments it finished keep their takes and state; the rest are ``skipped``
        with ``CANCELLED``. Everything made stays in the cache (section 8)."""
        for seg in run.segments:
            if seg.slots and all(a.settled for a in seg.current()):
                self._decide(seg)
                continue
            seg.state = "skipped"
            seg.flags.append(
                Flag(
                    code=codes.CANCELLED,
                    severity="error",
                    message=f"{seg.segment_id} was not finished: the job was cancelled",
                    segment_id=seg.segment_id,
                    retake_trigger=False,
                )
            )
        run.message = "cancelled; the work it finished is kept in the cache"
        updated = host.store.update_job(
            run.job_id,
            expect_status="cancelling",
            status="cancelled",
            phase=None,
            round=run.round,
            progress=self.progress(run),
            items=self.items(run),
            message=run.message,
        )
        self.release(run)
        return updated

    def fail(self, host: RunnerHost, job: JobRecord, error: NarrationError, run: JobRun | None) -> JobRecord | None:
        """Record a job-level failure: status ``failed`` with the error. A job being cancelled is cancelled."""
        changes: dict[str, Any] = {"phase": None, "error": error.error, "message": f"failed: {error.message}"}
        if run is not None:
            changes.update(round=run.round, progress=self.progress(run), items=self.items(run))
        updated = host.store.update_job(job.job_id, expect_status="running", status="failed", **changes)
        if updated is None:
            current = host.store.get_job(job.job_id)
            if current is not None and current.status == "cancelling":
                if run is not None:
                    return self.cancel(host, run)
                return host.store.update_job(job.job_id, expect_status="cancelling", status="cancelled", phase=None)
        if run is not None:
            self.release(run)
        log.warning("job %s failed: %s", job.job_id, error)
        return updated

    def save(self, host: RunnerHost, run: JobRun) -> bool:
        """Write the job's progress, items, round, phase and message. False when the job is no longer
        ``running`` (cancelled meanwhile): nothing is written then, so a cancel is never undone."""
        updated = host.store.update_job(
            run.job_id,
            expect_status="running",
            phase=run.phase,
            round=run.round,
            progress=self.progress(run),
            items=self.items(run),
            message=run.message,
        )
        return updated is not None

    def release(self, run: JobRun) -> None:
        """Remove the job's scratch files: work in flight is never published half made."""
        shutil.rmtree(run.scratch, ignore_errors=True)

    # ------------------------------------------------------------------ what the job record shows
    def progress(self, run: JobRun, *, complete: bool = False) -> Progress:
        """Progress in estimated audio seconds (section 7.4). Each attempt counts its segment's estimate, a third
        per layer made; ``total_s`` grows as retakes are added, and ``done_s`` never goes back."""
        total = done = 0.0
        settled = 0
        for seg in run.segments:
            for attempt in seg.attempts():
                total += seg.est_s
                done += seg.est_s * _STAGE_DONE[attempt.stage]
            if seg.state is not None or all(a.settled for a in seg.current()):
                settled += 1
        if complete:
            done = total
        done = max(done, run.done_floor)
        total = max(total, done)
        run.done_floor = done
        return Progress(
            done_s=round(done, 3),
            total_s=round(total, 3),
            fraction=round(done / total, 4) if total > 0 else 1.0,
            segments_done=settled,
            segments_total=len(run.segments),
        )

    def remaining_audio_s(self, run: JobRun) -> float:
        """Audio seconds of work left, for the queue's drain estimate."""
        progress = self.progress(run)
        return max(0.0, progress.total_s - progress.done_s)

    def items(self, run: JobRun) -> tuple[JobSegment, ...]:
        """The job's per-segment state (section 6 ``items[]``), which ``get_job`` and ``get_results`` read."""
        out: list[JobSegment] = []
        for seg in run.segments:
            attempts = [
                self._job_attempt(run, seg, slot, position, attempt)
                for slot in seg.slots
                for position, attempt in enumerate(slot)
            ]
            out.append(
                JobSegment(
                    segment_id=seg.segment_id,
                    state=seg.state or _interim_state(seg, run),
                    attempts=tuple(attempts),
                    takes_ok=sum(1 for a in seg.current() if a.stage == "done" and not is_retake_trigger(a)),
                    retakes_used=sum(len(slot) - 1 for slot in seg.slots),
                    flags=tuple(_stamped(f, seg.segment_id) for f in seg.flags),
                )
            )
        return tuple(out)

    def _job_attempt(
        self, run: JobRun, seg: SegmentWork, slot: list[Attempt], position: int, attempt: Attempt
    ) -> JobAttempt:
        """One attempt as the job keeps it, with its per-job flags (never part of its verdict): ``RETAKEN``,
        ``CANARY_MISMATCH`` and ``SPK_OUTLIER``."""
        flags: list[Flag] = []
        if position > 0:
            earlier = slot[:position]
            flags.append(
                Flag(
                    code=codes.RETAKEN,
                    severity="info",
                    message=(
                        f"retake of take slot {attempt.slot + 1}: attempt"
                        f"{'s' if len(earlier) > 1 else ''} {', '.join(str(a.attempt) for a in earlier)} failed QA"
                    ),
                    segment_id=seg.segment_id,
                    retake_trigger=False,
                    details={
                        "slot": attempt.slot,
                        "replaced": [
                            {
                                "attempt": a.attempt,
                                "take_id": a.take.take_id if a.take is not None else None,
                                "verdict": a.analysis.qa.verdict if a.analysis is not None else None,
                            }
                            for a in earlier
                        ],
                    },
                )
            )
        if attempt.render is not None and attempt.render.canary.batch_status == "similarity_pass":
            flags.append(
                Flag(
                    code=codes.CANARY_MISMATCH,
                    severity="info",
                    message="the canary's audio differed before this take's batch, but its similarity passed",
                    segment_id=seg.segment_id,
                    retake_trigger=False,
                )
            )
        if attempt.take is not None and attempt.take.take_id in run.outliers:
            flags.append(_stamped(run.outliers[attempt.take.take_id], seg.segment_id))
        return JobAttempt(
            attempt=attempt.attempt,
            seed=attempt.seed,
            round=attempt.round,
            render_key=attempt.render_key,
            render_id=attempt.render.render_id if attempt.render is not None else None,
            take_id=attempt.take.take_id if attempt.take is not None else None,
            analysis_id=attempt.analysis.analysis_id if attempt.analysis is not None else None,
            fresh=attempt.render is not None and attempt.render_key in run.fresh_keys,
            verdict=attempt.analysis.qa.verdict if attempt.analysis is not None else None,
            flags=tuple(flags),
        )

    # ------------------------------------------------------------------ small helpers
    def _phase(self, host: RunnerHost, run: JobRun, phase: JobPhase) -> None:
        if run.phase != phase:
            run.phase = phase
            host.job_phase(phase)

    @staticmethod
    def _label(run: JobRun, attempt: Attempt) -> str:
        seg = run.segments[attempt.segment]
        return f"{seg.segment_id} take {attempt.slot + 1} of {len(seg.slots)} (attempt {attempt.attempt})"


def is_retake_trigger(attempt: Attempt) -> bool:
    """Whether the attempt's take is a retake trigger: one of its QA flags is, by the contract's rule
    (``codes.is_retake_trigger``, which reads DC-12's ``details.reason``). QA carries the aligner's flags."""
    analysis = attempt.analysis
    if analysis is None:
        return False
    return any(codes.is_retake_trigger(f.code, f.severity, f.details) for f in analysis.qa.flags)


def _verb(group: GpuHolder | None) -> str:
    return "render" if group == "qwen" else "score" if group == "qa" else "post-process"


def _interim_state(seg: SegmentWork, run: JobRun) -> SegmentState:
    """A segment's state while the job runs: the layer its least advanced take slot is at."""
    pending: list[Stage] = [a.stage for a in seg.current() if not a.settled]
    if not pending:
        attempts = list(seg.attempts())
        verdicts = [a.analysis.qa.verdict for a in seg.current() if a.analysis is not None]
        if not verdicts:
            return "error"
        if all(a.render_key not in run.fresh_keys for a in attempts) and all(a.error is None for a in attempts):
            return "cached"
        return _VERDICT_STATE[min(verdicts, key=_VERDICT_RANK.__getitem__)]
    least = min(pending, key=_STAGE_ORDER.__getitem__)
    if least == "render":
        return "rendering" if any(a.render is not None for a in seg.attempts()) else "planned"
    return "rendered" if least == "post" else "postprocessed"


def _stamped(flag: Flag, segment_id: str) -> Flag:
    return flag if flag.segment_id == segment_id else dataclasses.replace(flag, segment_id=segment_id)


def _request_free(cue: CueText) -> CueText:
    """A cue's text echo as a cached analysis keeps it: nothing the analysis key does not cover. ``received``
    becomes the spoken form, and warnings carry no segment id. The job assembler echoes the request's own
    text (``get_results``), never a cached analysis's."""
    return dataclasses.replace(
        cue,
        received=cue.spoken,
        warnings=tuple(dataclasses.replace(w, segment_id=None) for w in cue.warnings),
    )


def _observed(hello: HelloReply | None) -> dict[str, Any]:
    """What the worker observed of its machine, recorded with the render and never hashed."""
    if hello is None:
        return {}
    fingerprint = cast(Mapping[str, Any], hello.get("fingerprint") or {})
    return {k: fingerprint[k] for k in ("gpu", "driver", "cuda", "cudnn") if fingerprint.get(k) is not None}


__all__ = [
    "DEFER_S",
    "KINDS",
    "MAX_RETRIES",
    "OOM_WAIT_S",
    "SCRATCH_JOBS",
    "Attempt",
    "EngineParts",
    "JobEngine",
    "JobRun",
    "Outcome",
    "Scorer",
    "SegmentWork",
    "is_retake_trigger",
]
