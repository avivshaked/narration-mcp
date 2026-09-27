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

**The cache answers first** (sections 4 item 6, 10.2; ``stages``). A resubmitted request walks the same
attempts, retakes included, and finds them all in the cache, so nothing is retaken again.

**Verdicts** come from ``QaScorer.score`` alone, on the take and the request's inputs for it. The suggestion
and the per-job consistency report come after every take is scored, and never change a verdict (section
11.1; ``record``).

**Failures** (``failures``). A job-level problem raises ``NarrationError`` for the runner to record on the
job: no engine pinned, the engine changed, a clip the service did not design and the owner did not allow
(section 17.4, checked here as well as at submit, since this is where a clip is cloned), the voice is not
measured or its clip changed, the VRAM wait timed out, a worker that cannot be installed or loaded. A
take-level execution problem is flagged on its segment, and the job goes on with the rest.

The engine is split by concern: ``state`` (the job's state), ``stages``, ``failures``, ``record`` (the job
record and the endings); this module plans and advances.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from narration import keys
from narration.config import Config
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.models import JobRecord, JobSegment, MeasuredError, Progress
from narration.contracts.names import CanaryStatus, GpuHolder
from narration.text import segment_too_long

from .admission import Throughput
from .core import EngineCore, EngineParts
from .failures import Failures
from .gpu import Residency
from .host import RunnerHost
from .pins import ProfileError, call_cap, qwen_load_payload
from .plan import GenerateRequest, RequestError, estimated_audio_s, hints_used, next_attempt, requested_attempts
from .record import JobView
from .stages import Stages
from .state import STAGE_ORDER, Attempt, JobRun, Outcome, SegmentWork, Stage, label, wants_retake
from .voice import require_synthetic, stage_clip

log = logging.getLogger(__name__)

KINDS: Final = ("generate", "analyse")
"""The job kinds this engine runs. ``analyse`` is scoring only: its takes are cached, its analyses are not.
Both run the same pipeline, and the cache decides which layers are made."""
SCRATCH_JOBS: Final = "jobs"
"""Each job's working files: ``scratch/jobs/<job_id>/``, removed when the job ends or is given back."""


class JobEngine:
    """Plans and advances generation jobs (see the module docstring). One instance serves one daemon: it keeps
    the model residency between jobs, so work that needs the resident group starts without a load."""

    def __init__(self, config: Config, parts: EngineParts) -> None:
        self.core = EngineCore(config, parts)
        self.stages = Stages(self.core)
        self.failures = Failures(self.core, self.stages)
        self.view = JobView(self.core)

    @property
    def config(self) -> Config:
        """The daemon's config."""
        return self.core.config

    @property
    def parts(self) -> EngineParts:
        """What the engine is built from."""
        return self.core.parts

    @property
    def residency(self) -> Residency:
        """Which model group is loaded, kept across jobs."""
        return self.core.residency

    @property
    def throughput(self) -> Throughput:
        """How fast work goes, for the queue's drain estimate."""
        return self.core.throughput

    @property
    def canary(self) -> CanaryStatus:
        """The canary outcome of the Qwen load in use, which every render made on it records."""
        return self.core.canary

    def close(self) -> None:
        """Stop the thread that renews the engine's leases (``leases.LeaseKeeper``). The runner calls it at
        shutdown, after the last step; work started later starts the thread again."""
        self.core.leases.close()

    # ------------------------------------------------------------------ planning
    def open(self, host: RunnerHost, job: JobRecord) -> JobRun:
        """Plan a claimed job from its request and the cache. Raises ``NarrationError`` for a job-level problem."""
        if job.kind not in KINDS:
            raise NarrationError(
                codes.INTERNAL,
                f"this build of the service does not run {job.kind} jobs",
                retryable=False,
                hint=f"The {job.kind} job engine is not in this build; nothing was made. Update the service.",
                details={"kind": job.kind},
            )
        store, config, parts = host.store, self.config, self.parts
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
        # Section 17.4, checked at submit too: the engine is what clones the clip.
        require_synthetic(store, config.voices.allow_sha256, request.voice.sha256)
        measurement = store.get_measurement(voice_hash, profile.engine_profile_id)
        if measurement is None:
            raise NarrationError(
                codes.VOICE_NOT_MEASURED,
                f"voice {voice_hash} has no measurement under engine profile {profile.engine_profile_id}",
                field="voice",
            )
        store.touch("measurement", measurement.measurement_key)
        clip = stage_clip(store, request.voice, check_path=parts.check_path)
        planned = parts.text.plan_request(request.segments, request.hints, strict_text=False)
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
            segments_floor=job.progress.segments_done,
            round_floor=job.round,
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
                seg.slots.append([self.stages.plan_attempt(host, run, seg, slot, number, 0)])
        # A job taken again (after a stop, or giving way) walks the rounds it finished, from the cache, so it
        # resumes at the round it was in rather than at round 0.
        while all(a.settled for seg in run.segments for a in seg.current()) and self._start_retakes(
            host, run, announce=False
        ):
            pass
        run.message = f"planned {len(run.segments)} segment(s) on {profile.engine_profile_id}"
        log.info("job %s: %s", job.job_id, run.message)
        return run

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
                return self.failures.guarded(host, run, todo, group)
        if pending:
            return self._wait_deferred(host, run, pending)
        if self._start_retakes(host, run):
            return "worked"
        self.finish(host, run)
        return "finished"

    def _wait_deferred(self, host: RunnerHost, run: JobRun, pending: list[Attempt]) -> Outcome:
        """Everything left is being produced by another holder, or waits for what is: wait a little
        (``defer_s``) before looking again. The wait goes through ``host.sleep``, so a stop ends it at once."""
        deferred = [a for a in pending if a.deferred_until > 0.0] or pending
        first = min(deferred, key=lambda a: (a.deferred_until, STAGE_ORDER[a.stage]))
        run.message = f"waiting for {label(run, first)}, which another job is making"
        if not host.sleep(self.parts.defer_s):
            return "stopped"
        for attempt in pending:
            attempt.deferred_until = 0.0
        return "waited"

    def _start_retakes(self, host: RunnerHost, run: JobRun, *, announce: bool = True) -> bool:
        """At the end of a settled round: one retake for every slot whose take is a retake trigger, on the next
        attempt numbers in slot order, up to ``max_retakes`` per slot (section 8). False when there is none.
        ``announce`` False (planning a job taken again) sets no phase and writes no message."""
        created = 0
        for seg in run.segments:
            used = seg.used()
            for slot_index, slot in enumerate(seg.slots):
                if not wants_retake(slot, run.request.max_retakes):
                    continue
                number = next_attempt(used)
                used.add(number)
                slot.append(self.stages.plan_attempt(host, run, seg, slot_index, number, run.round + 1))
                created += 1
        if not created:
            return False
        run.round += 1
        if announce:
            self.core.phase(host, run, "retaking")
            run.message = f"round {run.round}: {created} retake(s) of take slots that failed QA"
            log.info("job %s: %s", run.job_id, run.message)
        return True

    # ------------------------------------------------------------------ the record and the endings
    def finish(self, host: RunnerHost, run: JobRun) -> JobRecord | None:
        """Complete the job (``record.JobView.finish``): suggestions, consistency, outcome."""
        return self.view.finish(host, run)

    def cancel(self, host: RunnerHost, run: JobRun) -> JobRecord | None:
        """Finish a cancelled job, keeping what it made (``record.JobView.cancel``)."""
        return self.view.cancel(host, run)

    def fail(self, host: RunnerHost, job: JobRecord, error: NarrationError, run: JobRun | None) -> JobRecord | None:
        """Record a job-level failure (``record.JobView.fail``)."""
        return self.view.fail(host, job, error, run)

    def save(self, host: RunnerHost, run: JobRun) -> bool:
        """Write the job's state between steps; False when it is no longer ``running`` (``record.JobView.save``)."""
        return self.view.save(host, run)

    def release(self, run: JobRun) -> None:
        """Remove the job's scratch files."""
        self.view.release(run)

    def progress(self, run: JobRun, *, complete: bool = False) -> Progress:
        """The job's progress (section 7.4; ``record.JobView.progress``)."""
        return self.view.progress(run, complete=complete)

    def items(self, run: JobRun) -> tuple[JobSegment, ...]:
        """The job's per-segment state (section 6 ``items[]``)."""
        return self.view.items(run)

    def result(self, run: JobRun) -> dict[str, Any]:
        """The job's advice (``JobRecord.result``): suggestions and the consistency report."""
        return self.view.result(run)

    def remaining_audio_s(self, run: JobRun) -> float:
        """Audio seconds of work left, for the queue's drain estimate."""
        return self.view.remaining_audio_s(run)


__all__ = ["KINDS", "SCRATCH_JOBS", "JobEngine"]
