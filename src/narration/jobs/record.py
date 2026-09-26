"""What the job record shows, and how a job ends (design sections 6, 7.4 and 8).

The record view turns a ``JobRun`` into the job record's ``progress``, ``items`` and ``result``. The endings
write the record's last state by compare-and-set, so a change made meanwhile (a cancel) is never undone:
``finish`` (suggestions, the consistency report, the outcome), ``cancel``, ``fail``; and ``save`` between
steps. Nothing here changes a verdict.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from typing import Any

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import ScoredTake
from narration.contracts.models import Consistency, Flag, JobAttempt, JobRecord, JobSegment, Progress
from narration.contracts.serial import to_json

from .core import EngineCore
from .host import RunnerHost
from .state import STAGE_DONE, VERDICT_STATE, Attempt, JobRun, SegmentWork, interim_state, is_retake_trigger

log = logging.getLogger(__name__)


class JobView:
    """The job record of a ``JobRun``, and the job's endings (see the module docstring)."""

    def __init__(self, core: EngineCore) -> None:
        self.core = core

    # ------------------------------------------------------------------ the end of a job
    def finish(self, host: RunnerHost, run: JobRun) -> JobRecord | None:
        """Suggest a take per segment, report the request's consistency, decide the outcome, and complete the
        job (section 8). Nothing here changes a verdict. A job cancelled meanwhile ends ``cancelled``, with
        every segment's state. Returns the job as written, or None when its status changed otherwise."""
        scorer = self.core.parts.scorer
        self.core.phase(host, run, "suggesting")
        for seg in run.segments:
            self.decide(seg)
        suggested: list[tuple[str, tuple[float, ...]]] = []
        for seg in run.segments:
            for a in seg.attempts():
                if a.take is not None and a.take.take_id == seg.suggested and a.analysis is not None:
                    if a.analysis.embedding is not None:
                        suggested.append((a.take.take_id, a.analysis.embedding))
                    break
        try:
            consistency, outliers = scorer.consistency(suggested, run.measurement)
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
            round=run.shown_round,
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
                    round=run.shown_round,
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

    def decide(self, seg: SegmentWork) -> None:
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
        seg.suggested, seg.suggestion = self.core.parts.scorer.suggest(scored)
        chosen = next((s for s in scored if s.take_id == seg.suggested), None)
        seg.state = VERDICT_STATE[chosen.verdict] if chosen is not None else "error"

    def cancel(self, host: RunnerHost, run: JobRun) -> JobRecord | None:
        """Finish a cancelled job: the segments it finished keep their takes and state; the rest are ``skipped``
        with ``CANCELLED``. Everything made stays in the cache (section 8)."""
        for seg in run.segments:
            if seg.slots and all(a.settled for a in seg.current()):
                self.decide(seg)
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
            round=run.shown_round,
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
            changes.update(round=run.shown_round, progress=self.progress(run), items=self.items(run))
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
            round=run.shown_round,
            progress=self.progress(run),
            items=self.items(run),
            message=run.message,
        )
        return updated is not None

    @staticmethod
    def release(run: JobRun) -> None:
        """Remove the job's scratch files: work in flight is never published half made."""
        shutil.rmtree(run.scratch, ignore_errors=True)

    # ------------------------------------------------------------------ what the job record shows
    def progress(self, run: JobRun, *, complete: bool = False) -> Progress:
        """Progress in estimated audio seconds (section 7.4). Each attempt counts its segment's estimate, a third
        per layer made; ``total_s`` grows as retakes are added. ``done_s`` and ``segments_done`` never go
        back: a segment is done when nothing is left to make for it, retakes included, and a job taken again
        shows no less than it showed before."""
        total = done = 0.0
        finished = 0
        max_retakes = run.request.max_retakes
        for seg in run.segments:
            for attempt in seg.attempts():
                total += seg.est_s
                done += seg.est_s * STAGE_DONE[attempt.stage]
            if seg.state is not None or seg.finished(max_retakes):
                finished += 1
        if complete:
            done = total
        done = max(done, run.done_floor)
        total = max(total, done)
        finished = max(finished, run.segments_floor)
        run.done_floor, run.segments_floor = done, finished
        return Progress(
            done_s=round(done, 3),
            total_s=round(total, 3),
            fraction=round(done / total, 4) if total > 0 else 1.0,
            segments_done=finished,
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
                    state=seg.state or interim_state(seg, run),
                    attempts=tuple(attempts),
                    takes_ok=sum(1 for a in seg.current() if a.stage == "done" and not is_retake_trigger(a)),
                    retakes_used=sum(len(slot) - 1 for slot in seg.slots),
                    flags=tuple(_stamped(f, seg.segment_id) for f in seg.flags),
                )
            )
        return tuple(out)

    @staticmethod
    def _job_attempt(run: JobRun, seg: SegmentWork, slot: list[Attempt], position: int, attempt: Attempt) -> JobAttempt:
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


def _stamped(flag: Flag, segment_id: str) -> Flag:
    return flag if flag.segment_id == segment_id else dataclasses.replace(flag, segment_id=segment_id)


__all__ = ["JobView"]
