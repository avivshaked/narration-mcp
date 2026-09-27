"""The ``design`` job kind (design sections 3.1, 3.5, 3.6, 10.1, 10.3, 11.1 and 17.4; plan.md WP34): ``design_voice``.

``DesignHandler`` is a ``narration.jobs.handlers.JobHandler``: the runner dispatches a claimed ``design`` job to
it, and it does the job a piece at a time, so the daemon can stop, cancel or preempt it between pieces. It
shares the job engine's core (``JobEngine.core``): the model residency across every kind, the throughput
estimate and the canary outcome of the Qwen load in use.

**The job, in order.** The request is the one the front-end kept (``narration.backend.steps``): the opaque
``name``, the verbatim ``description``, ``takes`` (1-4), the ``design_text`` (the caller's, or ``[voice_design]
design_text``) and the ``design_id`` minted at submit.

1. **Design** (Qwen VoiceDesign, one load for every candidate): candidate ``i`` speaks the design text's spoken
   form (NFC, whitespace made single spaces; section 9.1) with the seed ``seeds.design_seed`` derives from the
   request alone (section 10.3). Every audio-changing setting comes from the pinned VoiceDesign engine profile
   (``non_streaming_mode`` true, the sampling values), and each call passes its own ``max_new_tokens`` cap
   (DC-4). After the load, the engine guard runs the VoiceDesign canary gate (section 10.1).
2. **The transcript check** (the QA group): Whisper transcribes each clip, and the design text is checked
   against what it heard by ``measure_voice``'s own rule (``narration.measure.transcript``), so a candidate
   that passes here passes ``measure_voice``'s check too (the same clip, model and rule).
3. **The profile** (the QA worker's ``profile`` op, on the CPU): section 3.6's measurements and pictures,
   with the speaking rate from the design text. The profile kept for the clip's bytes
   (``profiles/<ab>/<sha256>/``, what ``profile_voice`` answers for the same file) has no speaking rate, since
   ``profile_voice`` is sent no transcript and its answer must follow from its request alone; the candidate's
   copy adds the rate, which its own transcript gives.
4. **Publish**: the clip's sha256 is appended to the provenance list (section 17.4: append-only, never
   pruned), then the candidate is published under ``designs/<design_id>/<index>/`` (``Store.put_candidate``:
   the clip read-only, ``candidate.json``). From then on the synthetic-voices rule accepts the clip for
   ``measure_voice``, auditions and generation, with no allowlist edit.

**Each candidate** carries its clip (path, sha256), its exact transcript (the design text's spoken form) with the
transcript check, the verbatim description and its sha256, the design text, the seed, the engine profile, the
positive-only lint (section 3.5: it warns, never refuses) and the profile.

**Flags** (``JobRecord.result``, per candidate; ``get_results`` shows them with the candidate):

- ``CANARY_MISMATCH`` (info): the VoiceDesign canary's hash differed before this candidate's design, but its
  similarity passed (``similarity_pass``, ``bit_exact`` tier only, as for a take). The VoiceDesign gate is a
  weak drift alarm by nature: another seed designs another voice, so its threshold is low (the lead's
  decision, plan.md section 9, 2026-09-27).
- ``TOKEN_CAP_HIT`` (fail): the design reached its call's cap, so the clip may be cut short or run on.
- ``WER_HIGH`` (fail): the clip does not say its design text as Whisper heard it; ``measure_voice`` would refuse
  it with ``REF_TEXT_MISMATCH``.

The job's outcome is ``needs_attention`` when a candidate has a fail flag, else ``all_passed``. Every candidate
is published and on the provenance list either way: each is a clip the service designed, and the caller
listens and chooses (section 3.1).

**Failures** (``work.Pieces``): a candidate short of its clip, check or profile fails the job with a retryable
error; the same request designs the same candidates again. A job given back and taken again keeps the
candidates it published (``Store.get_design``) and designs the rest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.models import (
    AudioRef,
    Candidate,
    EngineProfile,
    EngineRef,
    Flag,
    JobRecord,
    LintResult,
    ProfileRecord,
    Progress,
    ProvenanceEntry,
    SegmentIn,
    TranscriptCheck,
)
from narration.contracts.names import CanaryStatus, JobOutcome
from narration.contracts.serial import ContractError, from_json, to_json
from narration.jobs.admission import CHARS_PER_AUDIO_S
from narration.jobs.engine import SCRATCH_JOBS, JobEngine
from narration.jobs.host import RunnerHost
from narration.jobs.pins import ProfileError, call_cap, qwen_load_payload
from narration.jobs.stages import FRAMES_PER_SECOND, QA_TIMEOUT_S, Stages
from narration.jobs.state import Outcome
from narration.lint import NegationLinter
from narration.measure.transcript import check_transcript
from narration.qa.normaliser import NumberReader
from narration.qa.profile import DEFAULT_PROFILE
from narration.store.store import utc_iso

from .seeds import MAX_INDEX, description_sha256, design_seed
from .work import RETRY_AFTER_S, Endings, Pieces, ProfileReplyError, StepRun, profile_record, set_phase

log = logging.getLogger(__name__)

KIND: Final = "design"
"""The job kind this handler runs (``JobRecord.kind``)."""
MAX_TAKES: Final = 4
"""``design_voice``'s ``takes`` is 1-4 (section 7.6)."""
DESIGN_SEGMENT: Final = "design_text"
"""The segment id the design text is planned under (the text pipeline needs one; it is never shown)."""

Stage = Literal["design", "check", "profile", "publish", "done"]
_DONE: Final[dict[Stage, float]] = {"design": 0.0, "check": 1 / 3, "profile": 1 / 2, "publish": 2 / 3, "done": 1.0}
"""How much of a candidate's work is done at each stage, for the job's progress."""


@dataclass(slots=True, eq=False, kw_only=True)
class CandidateWork:
    """One candidate while the job holds it: its seed, then its clip in the job's scratch folder, its transcript
    check and profile, and once published its ``Candidate`` and flags."""

    index: int
    seed: int
    est_s: float
    clip: Path | None = None
    sha256: str | None = None
    hit_token_cap: bool = False
    canary: CanaryStatus = "not_run"
    """The canary outcome of the VoiceDesign load it was designed on (``EngineCore.canary``)."""
    check: TranscriptCheck | None = None
    word_errors: int = 0
    words: int = 0
    profile: ProfileRecord | None = None
    published: Candidate | None = None
    flags: tuple[Flag, ...] = ()

    @property
    def stage(self) -> Stage:
        """What the candidate needs next."""
        if self.published is not None:
            return "done"
        if self.clip is None:
            return "design"
        if self.check is None:
            return "check"
        if self.profile is None:
            return "profile"
        return "publish"


@dataclass(slots=True, eq=False, kw_only=True)
class DesignRun(StepRun):
    """A ``design`` job the handler holds (a ``narration.jobs.core.ActiveRun``, so the engine's stages load
    VoiceDesign for it and run the canary gate). ``profile`` is the VoiceDesign engine profile; each candidate's
    voice profile is its own ``CandidateWork.profile``."""

    design_id: str
    description: str
    description_sha256: str
    design_text: str
    spoken: str
    """The design text's spoken form: what the model is given, and each candidate's exact transcript."""
    profile: EngineProfile
    lint: LintResult
    candidates: list[CandidateWork] = field(default_factory=list)

    def first_at(self, stage: Stage) -> CandidateWork | None:
        """The first candidate at ``stage``."""
        return next((c for c in self.candidates if c.stage == stage), None)


@dataclass(frozen=True, slots=True, kw_only=True)
class DesignAsk:
    """A ``design`` job's request as the handler reads it (``narration.backend.steps.design_request``)."""

    design_id: str
    description: str
    takes: int
    design_text: str

    @classmethod
    def parse(cls, request: Mapping[str, Any]) -> DesignAsk:
        """Read a stored request; ``INTERNAL`` for one without the shape the front-end gives it."""
        design_id = request.get("design_id")
        description = request.get("description")
        takes = request.get("takes")
        text = request.get("design_text")
        if (
            not names.is_id("design_id", design_id)
            or not isinstance(description, str)
            or not description
            or isinstance(takes, bool)
            or not isinstance(takes, int)
            or not 1 <= takes <= min(MAX_TAKES, MAX_INDEX + 1)
            or not isinstance(text, str)
            or not text
        ):
            raise NarrationError(
                codes.INTERNAL,
                "the design job's request cannot be read (design_id, description, takes, design_text)",
                retryable=False,
                hint="This is a bug in the service; nothing was designed. Report it.",
            )
        assert isinstance(design_id, str)
        return cls(design_id=design_id, description=description, takes=takes, design_text=text)


class DesignHandler:
    """Runs ``design`` jobs (``JobHandler[DesignRun]``; the module docstring). Build it with
    ``build_design_handler``."""

    def __init__(self, engine: JobEngine) -> None:
        self.engine = engine
        self.core = engine.core
        self.stages = Stages(self.core)
        self.pieces = Pieces(self.core, self.stages)
        self.endings = Endings()
        self.linter = NegationLinter()
        self._reader = NumberReader()

    # ------------------------------------------------------------------ planning
    def open(self, host: RunnerHost, job: JobRecord) -> DesignRun:
        """Plan a claimed ``design`` job from its request and what it already published. Raises
        ``NarrationError`` for a job-level problem: no VoiceDesign engine pinned, or one that cannot run."""
        if job.kind != KIND:
            raise NarrationError(
                codes.INTERNAL,
                f"the design handler was given a {job.kind} job",
                details={"kind": job.kind},
                hint="This is a bug in the service; nothing was designed. Report it.",
            )
        ask = DesignAsk.parse(job.request)
        store = host.store
        profile = design_profile(host)
        spoken = self._spoken(ask.design_text)
        digest = description_sha256(ask.description)
        run = DesignRun(
            job=job,
            scratch=store.scratch_path(SCRATCH_JOBS, job.job_id, "work").parent,
            done_floor=job.progress.done_s,
            design_id=ask.design_id,
            description=ask.description,
            description_sha256=digest,
            design_text=ask.design_text,
            spoken=spoken,
            profile=profile,
            lint=self.linter.lint(ask.description),
        )
        published = {c.index: c for c in store.get_design(ask.design_id)}
        earlier = _earlier_flags(job.result)
        est_s = max(1.0, len(spoken) / CHARS_PER_AUDIO_S)
        for index in range(ask.takes):
            seed = design_seed(description_sha256=digest, design_text=spoken, index=index)
            work = CandidateWork(index=index, seed=seed, est_s=est_s)
            found = published.get(index)
            if found is not None and found.seed == seed:  # published before the job was given back
                store.add_provenance(  # the list is append-only; an entry already there is not added again
                    ProvenanceEntry(clip_sha256=found.clip.sha256, design_id=ask.design_id, date=utc_iso(time.time()))
                )
                work.published = found
                work.flags = earlier.get(index, ())
            run.candidates.append(work)
        self.core.residency.need_mb["qwen"] = profile.vram_need_mb
        left = sum(1 for c in run.candidates if c.published is None)
        run.message = f"planned: {left} of {ask.takes} candidate(s) to design on {profile.engine_profile_id}"
        log.info("job %s: %s (design %s)", job.job_id, run.message, ask.design_id)
        return run

    def _spoken(self, design_text: str) -> str:
        """The design text's spoken form (section 9.1): the text pipeline's canonical text. Its refusals were
        checked at submit (``TEXT_REFUSED``, field ``design_text``); a change of rules since is refused here."""
        try:
            (planned,) = self.core.parts.text.plan_request(
                [SegmentIn(segment_id=DESIGN_SEGMENT, text=design_text)], [], strict_text=False
            )
        except NarrationError as exc:
            raise NarrationError(
                exc.code,
                exc.message,
                field="design_text",
                hint=exc.hint,
                details=exc.details,
                retryable=exc.retryable,
            ) from exc
        return planned.spoken_text

    # ------------------------------------------------------------------ advancing
    def advance(self, host: RunnerHost, run: DesignRun) -> Outcome:
        """Do one piece of the design, or one bounded wait; ``finished`` once its record is written. Every
        candidate is designed first (one VoiceDesign load); then each in turn is checked, profiled and published
        (one QA load), so a candidate is published as soon as it is ready."""
        work = run.first_at("design")
        if work is not None:
            return self.pieces.run(
                host, run, "qwen", f"design candidate {work.index + 1}", lambda: self._design(host, run, work)
            )
        work = next((c for c in run.candidates if c.stage != "done"), None)
        if work is None:
            self.finish(host, run)
            return "finished"
        if work.stage == "check":
            return self.pieces.run(
                host, run, "qa", f"check candidate {work.index + 1}'s transcript", lambda: self._check(host, run, work)
            )
        if work.stage == "profile":
            return self.pieces.run(
                host, run, "qa", f"profile candidate {work.index + 1}", lambda: self._profile(host, run, work)
            )
        self._publish(host, run, work)
        return "worked"

    def _design(self, host: RunnerHost, run: DesignRun, work: CandidateWork) -> Outcome:
        """Design one candidate on VoiceDesign (App. A ``design``), every audio-changing setting pinned."""
        core, stages = self.core, self.stages
        need = stages.qwen_need(run)
        if not stages.ready(host, run, need):
            return "stopped" if host.should_stop() else "waited"
        set_phase(host, run, "rendering")
        run.message = f"designing candidate {work.index + 1} of {len(run.candidates)}"
        client = host.workers.client("qwen", cublas_workspace_config=need.cublas_workspace_config)
        cap = call_cap(run.profile, run.spoken)
        out = run.scratch / f"candidate-{work.index}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        started = core.parts.clock()
        reply = client.request(
            "design",
            {
                "description": run.description,
                "design_text": run.spoken,
                "language": names.LANGUAGE,
                "seed": work.seed,
                "max_new_tokens": cap,
                "out_path": str(out),
            },
            timeout_s=max(300.0, cap / FRAMES_PER_SECOND * 5.0),
        )
        work.clip = out
        work.sha256 = _sha256(out)
        work.hit_token_cap = bool(reply.get("hit_token_cap", False))
        work.canary = core.canary
        core.throughput.record(core.parts.clock() - started, work.est_s / 3)
        return "worked"

    def _check(self, host: RunnerHost, run: DesignRun, work: CandidateWork) -> Outcome:
        """Transcribe the clip and check its transcript by ``measure_voice``'s rule (section 3.2, 11.1)."""
        stages = self.stages
        if not stages.ready(host, run, stages.qa_need()):
            return "stopped" if host.should_stop() else "waited"
        set_phase(host, run, "scoring")
        run.message = f"checking candidate {work.index + 1}'s transcript"
        assert work.clip is not None
        heard = host.workers.client("qa").request(
            "transcribe",
            {"wav": str(work.clip), "language": names.LANGUAGE, "word_timestamps": True, "long_form": True},
            timeout_s=QA_TIMEOUT_S,
        )
        work.check, work.word_errors, work.words = check_transcript(
            run.spoken, str(heard.get("text") or ""), self._reader, DEFAULT_PROFILE
        )
        return "worked"

    def _profile(self, host: RunnerHost, run: DesignRun, work: CandidateWork) -> Outcome:
        """The candidate's voice profile (section 3.6), on the QA worker's CPU; the profile of its bytes is kept
        without the speaking rate (the module docstring)."""
        stages, store = self.stages, host.store
        if not stages.ready(host, run, stages.qa_need()):
            return "stopped" if host.should_stop() else "waited"
        set_phase(host, run, "scoring")
        run.message = f"profiling candidate {work.index + 1}"
        assert work.clip is not None and work.sha256 is not None
        out_dir = run.scratch / f"profile-{work.index}"
        reply = host.workers.client("qa").request(
            "profile",
            {"wav": str(work.clip), "out_dir": str(out_dir), "transcript": run.spoken},
            timeout_s=QA_TIMEOUT_S,
        )
        try:
            measured = profile_record(work.sha256, reply)
        except ProfileReplyError as exc:
            raise NarrationError(
                codes.INTERNAL,
                f"candidate {work.index + 1}'s profile cannot be read: {exc}",
                retryable=True,
                retry_after_s=RETRY_AFTER_S,
                hint="Send the same request again; if it fails again, the daemon's log has the details.",
            ) from exc
        rate = measured.measurements.speaking_rate_wpm
        of_bytes = dataclasses.replace(
            measured, measurements=dataclasses.replace(measured.measurements, speaking_rate_wpm=None)
        )
        kept = store.put_profile(of_bytes, out_dir)
        work.profile = dataclasses.replace(
            kept, measurements=dataclasses.replace(kept.measurements, speaking_rate_wpm=rate)
        )
        return "worked"

    def _publish(self, host: RunnerHost, run: DesignRun, work: CandidateWork) -> None:
        """Append the clip to the provenance list, then publish the candidate (section 17.4, 15)."""
        store = host.store
        assert work.clip is not None and work.sha256 is not None and work.check is not None
        store.add_provenance(
            ProvenanceEntry(clip_sha256=work.sha256, design_id=run.design_id, date=utc_iso(time.time()))
        )
        candidate = Candidate(
            design_id=run.design_id,
            index=work.index,
            clip=AudioRef(path=str(work.clip), sha256=work.sha256),
            transcript=run.spoken,
            transcript_check=work.check,
            description=run.description,
            description_sha256=run.description_sha256,
            design_text=run.design_text,
            seed=work.seed,
            engine_profile=EngineRef(id=run.profile.engine_profile_id, hash=run.profile.hash),
            lint=run.lint,
            profile=work.profile,
        )
        work.published = store.put_candidate(candidate, work.clip)
        work.flags = candidate_flags(work, run.profile)
        run.message = f"published candidate {work.index + 1} of {len(run.candidates)}"
        log.info("job %s: candidate %d (%s) published", run.job_id, work.index, work.sha256)

    # ------------------------------------------------------------------ the record and the endings
    def result(self, run: DesignRun) -> dict[str, Any]:
        """``JobRecord.result``: the design id, the engine profile, and each published candidate's handle and
        flags (``get_results`` shows the flags with the candidate)."""
        return {
            "design_id": run.design_id,
            "engine_profile": {"id": run.profile.engine_profile_id, "hash": run.profile.hash},
            "candidates": [
                {
                    "index": c.index,
                    "seed": c.seed,
                    "clip_sha256": c.published.clip.sha256,
                    "transcript_ok": c.published.transcript_check is not None and c.published.transcript_check.ok,
                    "flags": [to_json(f) for f in c.flags],
                }
                for c in run.candidates
                if c.published is not None
            ],
        }

    def progress(self, run: DesignRun, *, complete: bool = False) -> Progress:
        """Progress in estimated audio seconds: a candidate's share by the stage it is at."""
        total = sum(c.est_s for c in run.candidates)
        done = sum(c.est_s * _DONE[c.stage] for c in run.candidates)
        published = sum(1 for c in run.candidates if c.published is not None)
        return self.endings.progress(run, done, total, published, len(run.candidates), complete=complete)

    def finish(self, host: RunnerHost, run: DesignRun) -> JobRecord | None:
        """Complete the job with its result (``needs_attention`` when a candidate has a fail flag)."""
        failing = [c.index + 1 for c in run.candidates if any(f.severity == "fail" for f in c.flags)]
        outcome: JobOutcome = "needs_attention" if failing else "all_passed"
        run.message = f"designed {len(run.candidates)} candidate(s) under design {run.design_id}" + (
            f"; candidate(s) {', '.join(map(str, failing))} have a fail flag" if failing else ""
        )
        return self.endings.complete(
            host, run, outcome=outcome, progress=self.progress(run, complete=True), result=self.result(run)
        )

    def cancel(self, host: RunnerHost, run: DesignRun) -> JobRecord | None:
        """Finish a cancelled design: the candidates published so far stay; nothing else is published."""
        return self.endings.cancel(host, run, self.progress(run), self.result(run))

    def fail(self, host: RunnerHost, job: JobRecord, error: NarrationError, run: DesignRun | None) -> JobRecord | None:
        """Record a job-level failure; the candidates published so far stay."""
        return self.endings.fail(host, job, error, run, self.progress(run) if run is not None else None)

    def save(self, host: RunnerHost, run: DesignRun) -> bool:
        """Write the job's phase, progress, message and the candidates published so far; False when it is no
        longer ``running``."""
        return self.endings.save(host, run, self.progress(run), self.result(run))

    def release(self, run: DesignRun) -> None:
        """Remove the job's scratch files."""
        self.endings.release(run)

    def remaining_audio_s(self, run: DesignRun) -> float:
        """Audio seconds of work left, for the queue's drain estimate."""
        progress = self.progress(run)
        return max(0.0, progress.total_s - progress.done_s)

    def close(self) -> None:
        """Stop the thread that renews the engine's leases (shared with the job engine; ``Registry.close``)."""
        self.core.leases.close()


# ---------------------------------------------------------------------- helpers


def design_profile(host: RunnerHost) -> EngineProfile:
    """The pinned VoiceDesign engine profile; ``BACKEND_NOT_INSTALLED`` when none is pinned, or when it lacks a
    setting a design needs (every audio-changing setting is passed explicitly, section 10.1)."""
    profile = host.store.current_engine_profile("design")
    if profile is None:
        raise NarrationError(
            codes.BACKEND_NOT_INSTALLED,
            "no engine profile is pinned for designing voices (Qwen VoiceDesign)",
            hint="Ask the operator to run narration-admin install, then narration-admin engine pin.",
            details={"engine": "design"},
        )
    try:
        call_cap(profile, "x")
        qwen_load_payload(profile, host.config.gpu.device)
    except ProfileError as exc:
        raise NarrationError(
            codes.BACKEND_NOT_INSTALLED, str(exc), hint="Ask the operator to pin the engine again."
        ) from exc
    return profile


def candidate_flags(work: CandidateWork, profile: EngineProfile) -> tuple[Flag, ...]:
    """A candidate's flags (the module docstring): ``CANARY_MISMATCH``, ``TOKEN_CAP_HIT``, ``WER_HIGH``."""
    flags: list[Flag] = []
    # Section 14: CANARY_MISMATCH only in the bit_exact tier, as for a take (narration.jobs.record).
    if work.canary == "similarity_pass" and profile.tier == "bit_exact":
        flags.append(
            Flag(
                code=codes.CANARY_MISMATCH,
                severity="info",
                message=(
                    "the VoiceDesign canary's audio differed before this candidate was designed, but its similarity "
                    "passed; the VoiceDesign gate is a weak drift alarm, not a check of this voice"
                ),
                retake_trigger=False,
                details={"candidate": work.index},
            )
        )
    if work.hit_token_cap:
        flags.append(
            Flag(
                code=codes.TOKEN_CAP_HIT,
                severity="fail",
                message="the design stopped at its max_new_tokens cap, so the clip may be cut short or run on",
                retake_trigger=False,
                details={"candidate": work.index},
            )
        )
    if work.check is not None and not work.check.ok:
        flags.append(
            Flag(
                code=codes.WER_HIGH,
                severity="fail",
                message=(
                    f"the clip does not say its design text: {work.word_errors} word error(s) in {work.words} words "
                    "as the speech recogniser heard it; measure_voice would refuse it (REF_TEXT_MISMATCH)"
                ),
                retake_trigger=False,
                details={
                    "candidate": work.index,
                    "heard": work.check.heard,
                    "wer": work.check.wer,
                    "word_errors": work.word_errors,
                    "words": work.words,
                },
            )
        )
    return tuple(flags)


def _earlier_flags(result: Mapping[str, Any] | None) -> dict[int, tuple[Flag, ...]]:
    """The flags of the candidates a job published before it was given back (its saved ``result``)."""
    out: dict[int, tuple[Flag, ...]] = {}
    entries = (result or {}).get("candidates")
    if not isinstance(entries, list):
        return out
    for entry in cast(list[Any], entries):
        if not isinstance(entry, dict):
            continue
        fields = cast(dict[str, Any], entry)
        index, flags = fields.get("index"), fields.get("flags")
        if isinstance(index, int) and isinstance(flags, list):
            try:
                out[index] = tuple(from_json(Flag, f) for f in cast(list[Any], flags))
            except ContractError:
                continue
    return out


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_design_handler(engine: JobEngine) -> DesignHandler:
    """The ``design`` kind's handler, for the runner's ``Registry`` (``narration.engine.installed``). It shares
    the daemon's job ``engine``'s core: the model residency, the throughput estimate and the canary."""
    return DesignHandler(engine)


__all__ = [
    "KIND",
    "MAX_TAKES",
    "CandidateWork",
    "DesignAsk",
    "DesignHandler",
    "DesignRun",
    "build_design_handler",
    "candidate_flags",
    "design_profile",
]
