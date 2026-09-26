"""Cue alignment, the pure part (design section 11.2; plan.md WP15): ``CtcAligner`` is the ``AlignerCore``.

The worker (``narration_worker_qa.align``) returns one span of frames per token, with its mean posterior.
``resolve`` turns those spans into cue and word times in delivery-file seconds:

1. **Words.** A word's time runs from its first letter's first frame to its last letter's end frame. A
   word outside the alphabet has null times. The wildcard that stands for a run of such words (plan.md
   DC-11) has a span of its own: the frames of the speech it absorbed.
2. **Confidence** (step 5): the mean posterior of a cue's letters and apostrophes (not ``|``, not ``*``).
   A wildcard's score measures only that there was speech (1 − P(blank)), not which words, so it never
   counts.
3. **Unplaceable cues** (step 7): a cue with no letter in the transcript, or with confidence under
   ``unplaced_below``. Its times, and its words' times, are null, and it gets ``CUE_UNALIGNED`` with a
   ``details.reason``: ``no_alignable_words`` (``codes.CUE_NO_ALIGNABLE_WORDS``, not a retake trigger:
   DC-12), ``low_confidence`` or ``alignment_error``. **It is never interpolated**: no time is ever made up
   from its neighbours. Its spans do not bound the snapping of the cues around it either, because they
   cannot be trusted. A cue whose words are all outside the alphabet has only wildcards, so no letter says
   it was spoken: it is ``no_alignable_words`` too (evidence: ``spikes/b-forced-align-cpu/README.md``).
4. **Snapping** (step 4). Each boundary between two placed cues moves to the edges of a pause (``snap``)
   found between the last aligned span before it and the first aligned span after it; a span is an aligned
   word or a wildcard's span. The cue before ends where the pause starts, and the cue after starts where it
   ends. The first onset and the last offset are snapped the same way, against the start and the end of
   the file. With no pause, the CTC times are kept, and a boundary between two adjacent cues gets
   ``CUE_BOUNDARY_NO_PAUSE`` (info). A pause is never allowed to reach past the aligned spans on either
   side. The word at a snapped edge takes the cue's snapped time.

   Which pause, when there are several: when nothing unaligned lies between the two spans, the longest.
   When unaligned speech does lie there (a whole unplaceable cue, or words no span covers, as in a
   transcript without wildcards), the pause next to the side that owns nothing unaligned is taken. If cue
   *k* ends on its last word, its end is the first pause after that word, and the unaligned speech belongs
   to the cues after it. If cue *k+1* starts on its first word, its start is the last pause before that
   word. When both sides own unaligned speech, it is the longest pause again.

   **Nothing takes speech it does not own.** An edge next to unaligned speech it does not own (an
   unplaceable cue's, including the cues before the first placed cue and after the last) snaps only when
   its pause lies within reach of its span: at most ``snap_reach_start_s`` of speech between the pause and
   the cue's first aligned letter, or ``snap_reach_end_s`` between its last and the pause. That much is the
   aligner's own imprecision at a word's edge. With more, the unplaceable cue's speech runs on into the
   cue's without a pause, and the pause belongs to the far side of that speech: the edge keeps the
   aligner's time, and the boundary gets ``CUE_BOUNDARY_NO_PAUSE`` with ``details.edge`` (``end`` or
   ``start``: which placed edge kept its time) and ``details.speech_s`` (the speech between the span and
   the nearest pause; null with no pause at all). The file's own edges are not limited this way: speech
   before cue 0 or after the last cue is not in the text, and head and end insertions (section 11.1) look
   for it.
5. **Cross-check** (step 6, ``crosscheck``): ``CUE_ALIGNMENT_DISAGREE`` above ``disagree_above_s``;
   ``max_disagreement_s`` is reported.
6. **Failure** (step 3). No reply (the worker reported ``ALIGNMENT_ERROR``), an ``ALIGNMENT_ERROR`` reply,
   or a reply that does not fit the transcript, makes every cue ``CUE_UNALIGNED`` and adds
   ``ALIGNMENT_ERROR`` (fail). Any other ``ok: false`` reply (``INTERNAL``, ``GPU_OOM``, …) is no verdict on
   the take: ``resolve`` raises ``WorkerFailure``, and the caller records ``QA_UNAVAILABLE``.

The thresholds are starting values, ASSUME until the alignment benchmark (WP38) sets them. KNOW for what
they separate on the bake-off's takes: ``spikes/b-forced-align-cpu/README.md``.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import rfc8785

from narration.config import AlignmentConfig
from narration.contracts import codes
from narration.contracts.errors import WorkerFailure
from narration.contracts.interfaces import AlignTranscript
from narration.contracts.models import (
    Alignment,
    CrossCheck,
    CueTiming,
    Flag,
    Hint,
    MeasuredError,
    SegmentText,
    WordTiming,
)
from narration.contracts.names import ALIGNMENT_METHOD, MODEL_ALIGNER, MODEL_ASR, Severity
from narration.contracts.worker import AlignReply, AsrWord

from .alphabet import WILDCARD
from .crosscheck import Boundary, cross_check
from .snap import Pause, PauseParams, find_pauses
from .transcript import build_transcript, guard, has_letters, repeats, wildcard_runs

METHOD_PREFIX: Final = "ctc-snap"
"""The method id's family (App. B: ``ctc-snap/<model>@<revision>``)."""
CTC_MODELS: Final = frozenset({MODEL_ALIGNER})
"""The models ``CtcAligner`` runs: wav2vec2 CTC models with the English character vocabulary, whose
thresholds spike (b) measured. Another model (the Qwen forced aligner, say) is another method, not
``ctc-snap``; the worker also refuses a snapshot that is not a wav2vec2 CTC model."""
DEVICE: Final = "cpu"
LOW_CONFIDENCE: Final = "low_confidence"
"""``CUE_UNALIGNED``'s ``details.reason`` for a cue whose confidence is under ``unplaced_below``."""
ALIGNMENT_FAILED: Final = "alignment_error"
"""``CUE_UNALIGNED``'s ``details.reason`` when the whole take could not be aligned (``ALIGNMENT_ERROR``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignerParams:
    """Every setting ``resolve`` decides by. All of them are part of ``method_id``.

    ``unplaced_below`` and ``low_confidence_below`` are ASSUME (``spikes/b-forced-align-cpu``): spoken
    sentences scored min 0.595 and p5 0.858 there; cues not in the audio scored 0.14 to 0.50, and one 0.70.
    ``disagree_above_s`` is the design's ASSUME 0.25 s. All three come from ``[alignment]``.
    """

    pauses: PauseParams = field(default_factory=PauseParams)
    snap_tolerance_s: float = 0.02
    """How far outside the gap between two aligned spans a pause may start or end and still count: one
    energy frame, for the frames' quantisation."""
    snap_reach_start_s: float = 0.12
    """Next to speech a cue does not own (an unplaced cue's), how much speech may lie between the pause and
    the cue's first aligned letter for the cue's start to snap to that pause. ASSUME: on the bake-off's takes
    the pause before a cue ended at most 0.10 s before its first letter (``spikes/b-forced-align-cpu``,
    ``reach.json``), plus one energy frame."""
    snap_reach_end_s: float = 0.28
    """The same after the cue's last aligned letter: at most 0.26 s there, plus one energy frame."""
    unplaced_below: float = 0.50
    low_confidence_below: float = 0.75
    disagree_above_s: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.unplaced_below <= self.low_confidence_below <= 1.0:
            raise ValueError("the thresholds need 0 <= unplaced_below <= low_confidence_below <= 1")
        if min(self.disagree_above_s, self.snap_tolerance_s, self.snap_reach_start_s, self.snap_reach_end_s) < 0:
            raise ValueError("disagree_above_s and the snap tolerance and reaches must not be negative")


@dataclass(slots=True)
class _Word:
    cue: int
    word: int
    text: str
    start_s: float | None = None
    end_s: float | None = None


@dataclass(slots=True)
class _Span:
    """What bounds the snapping: an aligned word, or a wildcard's span over the words ``first``..``last``."""

    start_s: float
    end_s: float
    first: int
    last: int
    word: _Word | None = None


@dataclass(slots=True)
class _Cue:
    index: int
    words: list[_Word]
    confidence: float | None = None
    placed: bool = False
    start_s: float | None = None
    end_s: float | None = None
    reason: str | None = None
    spans: list[_Span] = field(default_factory=list[_Span])

    def aligned(self) -> list[_Word]:
        return [w for w in self.words if w.start_s is not None]

    def owns_head(self) -> bool:
        """Whether words before the first span have no span (only in a transcript without wildcards)."""
        return self.spans[0].first > 0

    def owns_tail(self) -> bool:
        """Whether words after the last span have no span."""
        return self.spans[-1].last < len(self.words) - 1


class CtcAligner:
    """``AlignerCore`` for ``ctc-forced-align+silence-snap`` with the pinned wav2vec2 model (section 11.2)."""

    def __init__(
        self,
        *,
        revision: str,
        model: str = MODEL_ALIGNER,
        device: str = DEVICE,
        cross_check_model: str = MODEL_ASR,
        params: AlignerParams | None = None,
    ) -> None:
        if device != DEVICE:
            raise ValueError(f"the aligner runs on the CPU only, not {device!r} (design section 11.2)")
        if model not in CTC_MODELS:
            raise ValueError(
                f"{model!r} is not a CTC model this aligner runs ({', '.join(sorted(CTC_MODELS))}); "
                "set [alignment] model to it (design section 11.2)"
            )
        if not revision:
            raise ValueError("the aligner's model revision must be pinned")
        self._model = model
        self._revision = revision
        self._device = device
        self._cross_check_model = cross_check_model
        self._params = params if params is not None else AlignerParams()

    @classmethod
    def from_config(cls, config: AlignmentConfig, *, revision: str, cross_check_model: str = MODEL_ASR) -> CtcAligner:
        """The aligner ``[alignment]`` describes (section 16), at the pinned ``revision``: its model, device and
        thresholds (``disagree_threshold_s``, ``low_confidence_below``, ``unplaced_below``). ``ValueError`` for
        a model outside ``CTC_MODELS`` or a device other than the CPU."""
        params = AlignerParams(
            disagree_above_s=config.disagree_threshold_s,
            low_confidence_below=config.low_confidence_below,
            unplaced_below=config.unplaced_below,
        )
        return cls(
            revision=revision,
            model=config.model,
            device=config.device,
            cross_check_model=cross_check_model,
            params=params,
        )

    # ------------------------------------------------------------------ identity
    @property
    def model(self) -> str:
        return self._model

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def device(self) -> str:
        return self._device

    @property
    def params(self) -> AlignerParams:
        return self._params

    def method_params(self) -> dict[str, Any]:
        """The settings behind ``method_id``, as JSON (for the alignment benchmark's ``snap`` record)."""
        return {"method": ALIGNMENT_METHOD, **asdict(self._params)}

    @property
    def method_id(self) -> str:
        """``ctc-snap/<model name>@<revision>+p<12 hex>``: the model, its revision and every setting.

        The settings' hash covers the snap parameters and the confidence and cross-check thresholds. A
        threshold changes the flags and nulls of an analysis, so it must change the analysis key, and the
        method id is the key's only aligner input (``AnalysisKeyInputs.aligner_method_id``; design section
        11.2 step 8). A new id also needs the alignment benchmark run again for it (WP38).
        """
        digest = hashlib.sha256(rfc8785.dumps(self.method_params())).hexdigest()[:12]
        return f"{METHOD_PREFIX}/{self._model.rsplit('/', 1)[-1]}@{self._revision}+p{digest}"

    # ------------------------------------------------------------------ steps 1 and 3
    def build_transcript(self, segment: SegmentText, hints: Sequence[Hint]) -> AlignTranscript:
        """The transcript of section 11.2 step 1 (``transcript.build_transcript``)."""
        return build_transcript(segment, hints)

    def guard(self, transcript: AlignTranscript, num_frames: int) -> bool:
        """``T ≥ L + R``: frames at least tokens plus repeats."""
        return guard(transcript, num_frames)

    def guard_details(self, transcript: AlignTranscript, num_frames: int) -> dict[str, Any]:
        """The facts behind a failed guard, as the worker reports them: {reason, frames, tokens, repeats}."""
        tokens = transcript.tokens
        reason = "no_tokens" if not has_letters(transcript) else "too_short"
        return {"reason": reason, "frames": num_frames, "tokens": len(tokens), "repeats": repeats(tokens)}

    # ------------------------------------------------------------------ steps 4 to 8
    def resolve(
        self,
        transcript: AlignTranscript,
        reply: AlignReply | None,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
        asr_words: Sequence[AsrWord],
        measured_error: MeasuredError | None,
        *,
        error: Mapping[str, Any] | None = None,
    ) -> Alignment:
        """The take's alignment in delivery-file seconds (section 11.2 steps 4 to 8).

        ``reply`` None means the worker reported ``ALIGNMENT_ERROR`` (its ``details`` may be passed as
        ``error``), or the caller's own guard failed (pass ``guard_details`` as ``error``): every cue is
        unplaced, and the take gets ``ALIGNMENT_ERROR``. Pass None for nothing else. An ``ok: false`` reply
        with another code raises ``WorkerFailure`` (its code, message and details): it says nothing about
        the take, so the caller records ``QA_UNAVAILABLE`` (section 14). A transcript with no letter
        places nothing and needs no reply: every cue is unplaced (``CUE_UNALIGNED``, reason
        ``no_alignable_words``), and no ``ALIGNMENT_ERROR`` is raised, since the text, not the take, is at
        fault.
        """
        cues = _cues(transcript)
        if not has_letters(transcript):
            for cue in cues:
                cue.reason = codes.CUE_NO_ALIGNABLE_WORDS
            return self._alignment(cues, measured_error, self._unplaced_flags(cues), None)
        problem = _reply_problem(transcript, reply, self._model, self._revision) if reply is not None else None
        if reply is None or problem is not None:
            details = dict(problem or error or {"reason": "worker"})
            for cue in cues:
                cue.reason = ALIGNMENT_FAILED
            flags = [
                _flag(
                    codes.ALIGNMENT_ERROR,
                    "fail",
                    "The aligner could not align this take "
                    f"({details.get('reason', 'unknown reason')}); every cue is unplaced. Listen to the take: "
                    "audio too short for its text is a broken take, and a retake is due.",
                    details=details,
                ),
                *self._unplaced_flags(cues),
            ]
            return self._alignment(cues, measured_error, flags, None)

        self._place_words(transcript, reply, cues)
        flags: list[Flag] = []
        for cue in cues:
            self._decide_placement(cue)
        flags.extend(self._unplaced_flags(cues))
        placed = [c for c in cues if c.placed]
        for cue in placed:
            if cue.confidence is not None and cue.confidence < self._params.low_confidence_below:
                flags.append(
                    _flag(
                        codes.CUE_LOW_CONFIDENCE,
                        "warn",
                        f"Cue {cue.index} was placed with low confidence ({cue.confidence:.2f}); listen to it.",
                        cue=cue.index,
                        details={"confidence": cue.confidence, "threshold": self._params.low_confidence_below},
                    )
                )
        flags.extend(self._snap(cues, np.asarray(audio), sample_rate))
        checks = cross_check(
            _boundaries(cues), [(w.cue, w.word, w.text) for c in cues for w in c.words], asr_words, _names(transcript)
        )
        for check in checks:
            if check.disagreement_s > self._params.disagree_above_s:
                b = check.boundary
                flags.append(
                    _flag(
                        codes.CUE_ALIGNMENT_DISAGREE,
                        "warn",
                        f"The boundary after cue {b.before_cue} is {check.disagreement_s:.2f} s from Whisper's; "
                        "listen to it.",
                        cue=b.before_cue,
                        details={
                            "next_cue": b.after_cue,
                            "disagreement_s": check.disagreement_s,
                            "threshold_s": self._params.disagree_above_s,
                            "boundary_s": [b.end_s, b.start_s],
                            "asr_boundary_s": [check.asr_end_s, check.asr_start_s],
                        },
                    )
                )
        worst = max((c.disagreement_s for c in checks), default=None)
        return self._alignment(cues, measured_error, _ordered(flags), worst)

    # ------------------------------------------------------------------ helpers
    def _place_words(self, transcript: AlignTranscript, reply: AlignReply, cues: list[_Cue]) -> None:
        frame_s = float(reply["frame_s"])
        runs = wildcard_runs(transcript)
        by_key = {(w.cue, w.word): w for c in cues for w in c.words}
        scores: dict[int, list[float]] = {}
        frames: dict[tuple[int, int], list[int]] = {}
        wild: dict[int, list[_Span]] = {}
        for i, (span, token, owner) in enumerate(
            zip(reply["spans"], transcript.tokens, transcript.token_words, strict=True)
        ):
            if owner is None:
                continue
            if token == WILDCARD:
                run = runs[i]
                start, end = _s(span["start_frame"] * frame_s), _s(span["end_frame"] * frame_s)
                wild.setdefault(owner[0], []).append(_Span(start, end, run[0][1], run[-1][1]))
                continue
            frames.setdefault(owner, []).extend((span["start_frame"], span["end_frame"]))
            scores.setdefault(owner[0], []).append(float(span["score"]))
        for key, fs in frames.items():
            word = by_key.get(key)
            if word is not None:
                word.start_s, word.end_s = _s(min(fs) * frame_s), _s(max(fs) * frame_s)
        for cue in cues:
            values = scores.get(cue.index)
            cue.confidence = round(sum(values) / len(values), 3) if values else None
            spans = [_Span(_f(w.start_s), _f(w.end_s), w.word, w.word, w) for w in cue.aligned()]
            cue.spans = sorted([*spans, *wild.get(cue.index, [])], key=lambda x: (x.start_s, x.first))

    def _decide_placement(self, cue: _Cue) -> None:
        if not cue.aligned() or cue.confidence is None:
            # No letter says the cue was spoken: a wildcard alone shows speech, not which words.
            cue.reason = codes.CUE_NO_ALIGNABLE_WORDS
        elif cue.confidence < self._params.unplaced_below:
            cue.reason = LOW_CONFIDENCE
        else:
            cue.placed = True
            cue.start_s, cue.end_s = cue.spans[0].start_s, cue.spans[-1].end_s
            return
        cue.spans = []
        for word in cue.words:
            word.start_s = word.end_s = None

    def _unplaced_flags(self, cues: Sequence[_Cue]) -> list[Flag]:
        out: list[Flag] = []
        for cue in cues:
            if cue.placed:
                continue
            details: dict[str, Any] = {"reason": cue.reason}
            if cue.reason == LOW_CONFIDENCE:
                details |= {"confidence": cue.confidence, "threshold": self._params.unplaced_below}
            why = {
                codes.CUE_NO_ALIGNABLE_WORDS: "none of its words can be spelled in the aligner's alphabet, so no "
                "retake can place it (numbers and symbols belong in words)",
                LOW_CONFIDENCE: f"its confidence is very low ({cue.confidence})",
                ALIGNMENT_FAILED: "the take could not be aligned",
            }.get(cue.reason or "", "it could not be placed")
            out.append(
                _flag(
                    codes.CUE_UNALIGNED,
                    "warn",
                    f"Cue {cue.index} could not be placed: {why}. Its times are null (never interpolated); "
                    "listen to it.",
                    cue=cue.index,
                    details=details,
                )
            )
        return out

    def _snap(self, cues: list[_Cue], audio: npt.NDArray[Any], sample_rate: int) -> list[Flag]:
        placed = [c for c in cues if c.placed]
        if not placed:
            return []
        pauses = find_pauses(audio, sample_rate, self._params.pauses) if audio.size and sample_rate > 0 else ()
        duration = audio.shape[0] / sample_rate if sample_rate > 0 else math.inf
        tol = self._params.snap_tolerance_s
        flags: list[Flag] = []

        ctc = {cue.index: (cue.start_s, cue.end_s) for cue in placed}
        reach_start, reach_end = self._params.snap_reach_start_s, self._params.snap_reach_end_s

        first = placed[0]
        head = first.spans[0]
        before_owns = first.index > 0
        guarded = before_owns and not first.owns_head()
        before = [p for p in pauses if p.start_s < head.start_s + tol]
        pause = _choose(before, _start_rule(before_owns=before_owns, owns=first.owns_head(), edge=True))
        if guarded:
            pause, speech = _within_reach(pause, head.start_s, reach_start, start=True)
            if pause is None:
                flags.append(_kept_flag(first.index - 1, "start", speech))
        if pause is not None:
            first.start_s = _s(min(pause.end_s, head.end_s))

        for p, q in itertools.pairwise(placed):
            a, b = p.spans[-1], q.spans[0]
            tail_p, head_q, between = p.owns_tail(), q.owns_head(), q.index - p.index > 1
            window = [x for x in pauses if x.start_s < b.start_s + tol and x.end_s > a.end_s - tol]
            end_pause = _choose(window, _end_rule(owns=tail_p, after_owns=head_q or between, edge=False))
            start_pause = _choose(window, _start_rule(before_owns=tail_p or between, owns=head_q, edge=False))
            if not window and q.index == p.index + 1:
                flags.append(
                    _flag(
                        codes.CUE_BOUNDARY_NO_PAUSE,
                        "info",
                        f"No pause between cues {p.index} and {q.index}; the boundary keeps the aligner's times.",
                        cue=p.index,
                        details={"next_cue": q.index},
                    )
                )
            else:
                # Speech p or q does not own lies between them: an edge snaps only to the pause next to it.
                if not tail_p and (head_q or between):
                    end_pause, speech = _within_reach(end_pause, a.end_s, reach_end, start=False)
                    if end_pause is None:
                        flags.append(_kept_flag(p.index, "end", speech))
                if not head_q and (tail_p or between):
                    start_pause, speech = _within_reach(start_pause, b.start_s, reach_start, start=True)
                    if start_pause is None:
                        flags.append(_kept_flag(q.index - 1, "start", speech))
            if end_pause is not None:
                p.end_s = _s(max(end_pause.start_s, a.start_s))
            if start_pause is not None:
                q.start_s = _s(min(start_pause.end_s, b.end_s))

        last = placed[-1]
        tail = last.spans[-1]
        after_owns = last.index < len(cues) - 1
        guarded = after_owns and not last.owns_tail()
        after = [p for p in pauses if p.end_s > tail.end_s - tol]
        pause = _choose(after, _end_rule(owns=last.owns_tail(), after_owns=after_owns, edge=True))
        if guarded:
            pause, speech = _within_reach(pause, tail.end_s, reach_end, start=False)
            if pause is None:
                flags.append(_kept_flag(last.index, "end", speech))
        if pause is not None:
            last.end_s = _s(max(pause.start_s, tail.start_s))
        if math.isfinite(duration) and last.end_s is not None:
            last.end_s = min(last.end_s, _s(duration))

        for cue in placed:
            if cue.start_s is None or cue.end_s is None or cue.start_s > cue.end_s:
                # Both edges fell in one pause that swallows the cue's spans: its speech sits in silence.
                # Keep what the aligner found rather than a snapped time that contradicts it.
                cue.start_s, cue.end_s = ctc[cue.index]
        for p, q in itertools.pairwise(placed):
            # A cue kept at its aligner times may reach up to one frame into its neighbour's snapped edge.
            if p.end_s is not None and q.start_s is not None and p.end_s > q.start_s:
                p.end_s = max(q.start_s, p.start_s if p.start_s is not None else q.start_s)
                q.start_s = max(q.start_s, p.end_s)
        for cue in placed:
            head, tail = cue.spans[0], cue.spans[-1]
            if head.word is not None and head.first == 0:
                head.word.start_s = cue.start_s
            if tail.word is not None and tail.last == len(cue.words) - 1:
                tail.word.end_s = cue.end_s
            for word in cue.aligned():
                if word.start_s is not None and word.end_s is not None and word.end_s < word.start_s:
                    word.end_s = word.start_s
        return flags

    def _alignment(
        self, cues: Sequence[_Cue], measured_error: MeasuredError | None, flags: Sequence[Flag], worst: float | None
    ) -> Alignment:
        timings = tuple(
            CueTiming(
                index=c.index,
                start_s=c.start_s if c.placed else None,
                end_s=c.end_s if c.placed else None,
                confidence=c.confidence,
                words=tuple(
                    WordTiming(
                        text=w.text,
                        start_s=w.start_s if c.placed else None,
                        end_s=w.end_s if c.placed else None,
                    )
                    for w in c.words
                ),
            )
            for c in cues
        )
        return Alignment(
            method=ALIGNMENT_METHOD,
            model=self._model,
            revision=self._revision,
            device=self._device,
            cross_check=CrossCheck(model=self._cross_check_model, max_disagreement_s=worst),
            measured_error=measured_error,
            cues=timings,
            flags=tuple(flags),
        )


# ---------------------------------------------------------------------- module helpers


def _s(value: float) -> float:
    """Seconds rounded to the millisecond (a frame is 20 ms), so float noise never reaches the sidecar."""
    return round(value, 3)


def _f(value: float | None) -> float:
    if value is None:
        raise AssertionError("an aligned word has times")
    return value


def _flag(
    code: str, severity: Severity, message: str, *, cue: int | None = None, details: dict[str, Any] | None = None
) -> Flag:
    return Flag(
        code=code,
        severity=severity,
        message=message,
        cue=cue,
        retake_trigger=codes.is_retake_trigger(code, severity, details),
        details=details,
    )


def _ordered(flags: Sequence[Flag]) -> tuple[Flag, ...]:
    """Flags in a stable order: by cue (the take's flags first), then by code."""
    return tuple(sorted(flags, key=lambda f: (-1 if f.cue is None else f.cue, f.code)))


def _cues(transcript: AlignTranscript) -> list[_Cue]:
    count = transcript.cue_count
    for cue, _, _ in transcript.words:
        count = max(count, cue + 1)
    cues = [_Cue(index=i, words=[]) for i in range(count)]
    for cue, word, text in transcript.words:
        cues[cue].words.append(_Word(cue, word, text))
    return cues


def _names(transcript: AlignTranscript) -> frozenset[tuple[int, int]]:
    return frozenset(transcript.term_words)


def _boundaries(cues: Sequence[_Cue]) -> list[Boundary]:
    out: list[Boundary] = []
    for p, q in itertools.pairwise(cues):
        if p.placed and q.placed and p.words and q.words and p.end_s is not None and q.start_s is not None:
            out.append(
                Boundary(
                    p.index,
                    q.index,
                    p.end_s,
                    q.start_s,
                    (p.index, p.words[-1].word),
                    (q.index, q.words[0].word),
                )
            )
    return out


def _end_rule(*, owns: bool, after_owns: bool, edge: bool) -> str:
    """Which pause ends a cue. ``owns``: the cue has unaligned words after its last aligned span;
    ``after_owns``: unaligned speech of later cues lies before the next aligned span (or the file's end)."""
    if not owns:
        return "earliest" if (edge or after_owns) else "longest"
    return "latest" if not after_owns else "longest"


def _start_rule(*, before_owns: bool, owns: bool, edge: bool) -> str:
    """Which pause starts a cue. ``owns``: the cue has unaligned words before its first aligned span;
    ``before_owns``: unaligned speech of earlier cues lies after the aligned span before (or the file's start)."""
    if not owns:
        return "latest" if (edge or before_owns) else "longest"
    return "earliest" if not before_owns else "longest"


def _within_reach(
    pause: Pause | None, edge_s: float, reach_s: float, *, start: bool
) -> tuple[Pause | None, float | None]:
    """``pause`` if at most ``reach_s`` of speech lies between it and a span's edge, else None; and that
    speech in seconds (None with no pause). ``start``: the span starts at ``edge_s`` and the pause is before
    it; otherwise the span ends there and the pause is after it."""
    if pause is None:
        return None, None
    speech = _s(max(0.0, edge_s - pause.end_s if start else pause.start_s - edge_s))
    return (pause if speech <= reach_s else None), speech


def _kept_flag(cue: int, edge: str, speech_s: float | None) -> Flag:
    """``CUE_BOUNDARY_NO_PAUSE`` at the boundary after ``cue``, whose placed side kept its aligner time at
    ``edge`` (``end``: cue's end; ``start``: the next cue's start) because no pause lies next to it on the
    side of speech it does not own. ``speech_s``: the speech between its span and the nearest pause."""
    owner = cue if edge == "end" else cue + 1
    return _flag(
        codes.CUE_BOUNDARY_NO_PAUSE,
        "info",
        f"No pause separates cue {owner}'s aligned words from the unaligned speech "
        f"{'after' if edge == 'end' else 'before'} them; its {edge} keeps the aligner's time.",
        cue=cue,
        details={"next_cue": cue + 1, "edge": edge, "speech_s": speech_s},
    )


def _choose(pauses: Sequence[Pause], rule: str) -> Pause | None:
    if not pauses:
        return None
    if rule == "earliest":
        return min(pauses, key=lambda p: p.start_s)
    if rule == "latest":
        return max(pauses, key=lambda p: p.end_s)
    return max(pauses, key=lambda p: (p.duration_s, -p.start_s))


def _reply_problem(transcript: AlignTranscript, reply: AlignReply, model: str, revision: str) -> dict[str, Any] | None:
    """Why a reply cannot be used for this transcript, or None. A reply that does not fit is a failure.

    An ``ok: false`` reply is a verdict only when its code is ``ALIGNMENT_ERROR`` (the guard, or
    ``forced_align`` raising). Any other raises ``WorkerFailure``, as ``WorkerClient.request`` does: it says
    nothing about the take, and the caller records ``QA_UNAVAILABLE`` (section 14).
    """
    if not reply.get("ok", True):
        error: Mapping[str, Any] = reply.get("error") or {}
        code = error.get("code")
        details = dict(error.get("details") or {})
        if code != codes.ALIGNMENT_ERROR:  # the worker's code has the flag's name (Appendix A)
            message = str(error.get("message") or "the aligner replied ok: false without an error")
            raise WorkerFailure(code if isinstance(code, str) else "INTERNAL", message, details)
        return {"reason": "worker_error", **details}
    if reply.get("model") != model or reply.get("revision") != revision:
        return {
            "reason": "model_mismatch",
            "expected": f"{model}@{revision}",
            "got": f"{reply.get('model')}@{reply.get('revision')}",
        }
    spans = reply.get("spans", [])
    frames = reply.get("num_frames", 0)
    frame_s = reply.get("frame_s", 0.0)
    tokens = transcript.tokens
    if not (isinstance(frame_s, int | float) and frame_s > 0 and math.isfinite(frame_s)):
        return {"reason": "bad_reply", "field": "frame_s"}
    if frames < len(tokens) + repeats(tokens):
        return {"reason": "too_short", "frames": frames, "tokens": len(tokens), "repeats": repeats(tokens)}
    if len(spans) != len(tokens):
        return {"reason": "span_mismatch", "spans": len(spans), "tokens": len(tokens)}
    previous_end = 0
    for i, span in enumerate(spans):
        start, end, score = span["start_frame"], span["end_frame"], span["score"]
        if span["token_index"] != i or not (previous_end <= start < end <= frames):
            return {"reason": "bad_reply", "field": f"spans[{i}]"}
        if not (isinstance(score, int | float) and math.isfinite(score) and 0.0 <= score <= 1.0):
            return {"reason": "bad_reply", "field": f"spans[{i}].score"}
        previous_end = end
    return None
