"""A job's state while the engine holds it (design sections 4 and 8; plan.md WP31).

``JobRun`` is one job the engine holds: its request as read, its engine profile and measurement, and its
segments. ``SegmentWork`` is one segment: its text and hints, and its take slots, each a list of attempts
with the current one last. ``Attempt`` is one attempt of one slot: its keys, and the layers the cache or
this job made for it (render, take, analysis), or the execution problem that ended it.

Nothing here talks to the store or a worker: the stages (``stages``) fill these in, and the record view
(``record``) turns them into the job record.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from narration.contracts import codes
from narration.contracts.interfaces import WorkerClient
from narration.contracts.models import (
    AnalysisRecord,
    Consistency,
    EngineProfile,
    Flag,
    Hint,
    JobRecord,
    MeasuredError,
    MeasurementRecord,
    RenderRecord,
    SegmentText,
    Suggestion,
    TakeRecord,
)
from narration.contracts.names import JobPhase, SegmentState

from .plan import GenerateRequest

Stage = Literal["render", "post", "score", "done", "error"]
"""The next layer an attempt needs (``render``, ``post``, ``score``), or ``done``, or ``error``."""
Outcome = Literal["worked", "waited", "stopped", "finished"]
"""What one call of ``JobEngine.advance`` did: some work, a bounded wait, stopped for the daemon, or the end."""

STAGE_ORDER: Final[dict[Stage, int]] = {"render": 0, "post": 1, "score": 2, "done": 3, "error": 3}
STAGE_DONE: Final[dict[Stage, float]] = {"render": 0.0, "post": 1 / 3, "score": 2 / 3, "done": 1.0, "error": 1.0}
"""How much of an attempt's work is done at each stage: a third per layer (section 7.4's progress)."""
VERDICT_STATE: Final[dict[str, SegmentState]] = {"pass": "passed", "warn": "warned", "fail": "failed_qa"}
VERDICT_RANK: Final[dict[str, int]] = {"pass": 0, "warn": 1, "fail": 2}


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
    """Retries of this attempt after its worker crashed, timed out or lost its models."""
    deferred_until: float = 0.0
    """While another holder makes this attempt's next layer: when to look again (the engine's clock)."""

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
        """Whether nothing is left to make for this attempt: it is scored, or it ended with an error."""
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
    oom_retries: int = 0
    """Section 4 item 5: retries of this segment's GPU work after running out of memory, since its last
    piece of GPU work went through."""

    @property
    def segment_id(self) -> str:
        """The segment's id, as the request gave it."""
        return self.text.segment_id

    def attempts(self) -> Iterator[Attempt]:
        """Every attempt of every slot, slot by slot, oldest first."""
        for slot in self.slots:
            yield from slot

    def current(self) -> list[Attempt]:
        """Each slot's current attempt (its last)."""
        return [slot[-1] for slot in self.slots]

    def used(self) -> set[int]:
        """The attempt numbers the segment has used so far, in every slot."""
        return {a.attempt for a in self.attempts()}

    def finished(self, max_retakes: int) -> bool:
        """Whether the segment has nothing left to make: every slot's current attempt is settled, and none
        will be retaken. A finished segment stays finished, so ``segments_done`` never drops."""
        return all(a.settled for a in self.current()) and not any(wants_retake(s, max_retakes) for s in self.slots)


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
    """The most ``progress.done_s`` the job has shown: progress never goes back."""
    segments_floor: int = 0
    """The most ``progress.segments_done`` the job has shown."""
    round_floor: int = 0
    """The highest round the job has shown, before it was given back and taken again."""
    round: int = 0
    phase: JobPhase | None = None
    message: str | None = None
    prepared: WorkerClient | None = None
    """The Qwen worker the voice was prepared in since its last load, or None. Compared by identity: a new
    worker process is a new client object, even if its pid is reused."""
    consistency: Consistency | None = None
    outliers: dict[str, Flag] = field(default_factory=dict)
    """``SPK_OUTLIER`` flags by take id, from the consistency report."""

    @property
    def job_id(self) -> str:
        """The job's id."""
        return self.job.job_id

    @property
    def shown_round(self) -> int:
        """The round the job record shows: the current one, never below one it showed before."""
        return max(self.round, self.round_floor)


def is_retake_trigger(attempt: Attempt) -> bool:
    """Whether the attempt's take is a retake trigger: one of its QA flags is, by the contract's rule
    (``codes.is_retake_trigger``, which reads DC-12's ``details.reason``). QA carries the aligner's flags."""
    analysis = attempt.analysis
    if analysis is None:
        return False
    return any(codes.is_retake_trigger(f.code, f.severity, f.details) for f in analysis.qa.flags)


def wants_retake(slot: list[Attempt], max_retakes: int) -> bool:
    """Whether the slot gets a retake when its round is settled: its current take is scored and a retake
    trigger, and the slot has retakes left (``max_retakes`` per slot, section 8)."""
    current = slot[-1]
    return current.stage == "done" and is_retake_trigger(current) and len(slot) <= max_retakes


def interim_state(seg: SegmentWork, run: JobRun) -> SegmentState:
    """A segment's state while the job runs: the layer its least advanced take slot is at."""
    pending: list[Stage] = [a.stage for a in seg.current() if not a.settled]
    if not pending:
        attempts = list(seg.attempts())
        verdicts = [a.analysis.qa.verdict for a in seg.current() if a.analysis is not None]
        if not verdicts:
            return "error"
        if all(a.render_key not in run.fresh_keys for a in attempts) and all(a.error is None for a in attempts):
            return "cached"
        return VERDICT_STATE[min(verdicts, key=VERDICT_RANK.__getitem__)]
    least = min(pending, key=STAGE_ORDER.__getitem__)
    if least == "render":
        return "rendering" if any(a.render is not None for a in seg.attempts()) else "planned"
    return "rendered" if least == "post" else "postprocessed"


def label(run: JobRun, attempt: Attempt) -> str:
    """How the log and the job's message name an attempt."""
    seg = run.segments[attempt.segment]
    return f"{seg.segment_id} take {attempt.slot + 1} of {len(seg.slots)} (attempt {attempt.attempt})"


__all__ = [
    "STAGE_DONE",
    "STAGE_ORDER",
    "VERDICT_RANK",
    "VERDICT_STATE",
    "Attempt",
    "JobRun",
    "Outcome",
    "SegmentWork",
    "Stage",
    "interim_state",
    "is_retake_trigger",
    "label",
    "wants_retake",
]
