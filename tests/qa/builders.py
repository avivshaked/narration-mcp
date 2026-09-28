"""Builders for QA tests: segments, inputs, measurements and results made from the tests' own text.

The text here is written for these tests (invented names included); no caller's script is used (design
section 9.3). The builders fill every record a scorer reads with plain, clean values, so a test changes only
what it is about.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from narration.contracts.interfaces import QaInputs, ScoredTake, SignalStats
from narration.contracts.models import (
    Alignment,
    Anchor,
    CrossCheck,
    CueIn,
    CueText,
    CueTiming,
    DeliveryAudio,
    EngineRef,
    ExactSpan,
    ExactWords,
    Flag,
    Hint,
    Loudness,
    MeasurementRecord,
    Pace,
    PacePoint,
    PaceTrend,
    PaceValue,
    QaResult,
    SegmentIn,
    SegmentResult,
    SegmentResultText,
    SegmentText,
    SimilarityBaseline,
    TakeAlignment,
    TakeQa,
    TakeResult,
    TranscriptCheck,
    Trim,
)
from narration.contracts.names import ALIGNMENT_METHOD, MODEL_ALIGNER, MODEL_ASR, MODEL_SV, PACE_METHOD, Verdict
from narration.text import TextPipeline
from narration.text import words as text_words

VOICE_TRANSCRIPT = "Morning comes slowly to the harbour, and the gulls wait on the old stone wall for the boats."
DIM = 8


@dataclass(frozen=True)
class Cue:
    """A cue's text and its exact spans as word ranges [first, end)."""

    text: str
    exact: tuple[tuple[int, int], ...] = ()


def segment(*cues: str | Cue, segment_id: str = "p01") -> SegmentText:
    """A segment whose cues are already in canonical form, with no hints applied and no text warnings."""
    built: list[CueText] = []
    offset = 0
    for index, cue in enumerate(cues):
        spec = cue if isinstance(cue, Cue) else Cue(cue)
        cue_words = text_words(spec.text)
        exact = tuple(
            ExactWords(start=cue_words[a].core_start, end=cue_words[b - 1].core_end, words=(a, b))
            for a, b in spec.exact
        )
        span = (offset, offset + len(spec.text))
        built.append(
            CueText(
                index=index,
                received=spec.text,
                spoken=spec.text,
                engine=spec.text,
                spoken_span=span,
                engine_span=span,
                exact=exact,
            )
        )
        offset += len(spec.text) + 1
    join = " ".join(c.spoken for c in built)
    return SegmentText(
        segment_id=segment_id, cues=tuple(built), spoken_text=join, engine_text=join, spoken_chars=len(join)
    )


def planned(
    *cues: str | tuple[str, Sequence[tuple[int, int]]], hints: Sequence[Hint] = (), segment_id: str = "p01"
) -> SegmentText:
    """A segment as the text pipeline plans it (WP10), with its hints applied: each cue is its text as sent,
    or (text, exact spans as code-point ranges of the text as sent)."""
    cue_in = []
    for cue in cues:
        text, spans = (cue, ()) if isinstance(cue, str) else cue
        cue_in.append(CueIn(text=text, exact=tuple(ExactSpan(start=a, end=b) for a, b in spans)))
    return TextPipeline().plan_segment(SegmentIn(segment_id=segment_id, cues=tuple(cue_in)), hints)


def unit(*values: float) -> tuple[float, ...]:
    """A unit vector of dimension ``DIM`` from the given leading values (the rest zero)."""
    padded = [*values, *[0.0] * (DIM - len(values))]
    norm = math.sqrt(sum(v * v for v in padded))
    return tuple(v / norm for v in padded)


def with_similarity(sim: float) -> tuple[float, ...]:
    """An embedding whose cosine similarity to ``ANCHOR`` (the first axis) is exactly ``sim``."""
    return (sim, math.sqrt(max(0.0, 1.0 - sim * sim)), *[0.0] * (DIM - 2))


ANCHOR = Anchor(model=MODEL_SV, dim=DIM, embedding=unit(1.0))


def pace(
    curve: Sequence[tuple[int, float]] = ((80, 16.0), (300, 17.1)),
    *,
    tol: float = 0.10,
    level: float = 16.5,
    intercept: float = 15.6,
    per_100: float = 0.5,
    speaking_share: float = 0.9,
) -> Pace:
    """A pace model in spoken characters per second of speaking time (invented numbers). The trend
    (``intercept``, ``per_100``) is information only (DC-20): QA never reads it."""
    return Pace(
        method=PACE_METHOD,
        level_cps=level,
        trend=PaceTrend(intercept_cps=intercept, per_100_chars=per_100, band_max_chars=300),
        tol=tol,
        curve=tuple(PacePoint(chars=c, cps=v) for c, v in curve),
        speaking_share=speaking_share,
    )


def measurement(
    *,
    anchor_p5: float = 0.975,
    consistency_p5: float = 0.98,
    pace_record: Pace | None = None,
    anchor: Anchor = ANCHOR,
) -> MeasurementRecord:
    return MeasurementRecord(
        voice_hash="sha256:" + "a" * 64,
        clip_sha256="b" * 64,
        engine_profile=EngineRef(id="qwen3-base-1.7b.p1", hash="sha256:" + "c" * 64),
        measurement_key="sha256:" + "d" * 64,
        transcript_check=TranscriptCheck(heard=VOICE_TRANSCRIPT, wer=0.0, ok=True),
        corpus="narration-en.v1",
        similarity=SimilarityBaseline(anchor_p5=anchor_p5, anchor_p50=anchor_p5 + 0.01, consistency_p5=consistency_p5),
        pace=pace_record if pace_record is not None else pace(),
        max_segment_chars=450,
        max_segment_seconds=31.5,
        ladder=(),
        anchor=anchor,
        calibration=(),
        measured_at="2026-09-26T00:00:00Z",
    )


def placed_cues(seg: SegmentText, *, seconds_per_cue: float = 3.0) -> tuple[CueTiming, ...]:
    return tuple(
        CueTiming(
            index=c.index,
            start_s=0.08 + i * seconds_per_cue,
            end_s=0.08 + (i + 1) * seconds_per_cue - 0.3,
            confidence=0.95,
        )
        for i, c in enumerate(seg.cues)
    )


def alignment(seg: SegmentText, *, cues: tuple[CueTiming, ...] | None = None, flags: Sequence[Flag] = ()) -> Alignment:
    return Alignment(
        method=ALIGNMENT_METHOD,
        model=MODEL_ALIGNER,
        revision="0" * 40,
        device="cpu",
        cross_check=CrossCheck(model=MODEL_ASR, max_disagreement_s=0.05),
        measured_error=None,
        cues=cues if cues is not None else placed_cues(seg),
        flags=tuple(flags),
    )


def signal(
    *,
    duration_s: float = 10.0,
    voiced: tuple[float | None, float | None] = (0.08, 9.92),
    silence_s: float = 0.5,
    clipping: float = 0.0,
    nonfinite: bool = False,
    dc: float = 0.0,
    silences: tuple[float, ...] | None = None,
) -> SignalStats:
    return SignalStats(
        raw_samples=int(duration_s * 24000),
        raw_sample_rate=24000,
        raw_clipping_fraction=clipping,
        raw_nonfinite=nonfinite,
        raw_dc_offset=dc,
        delivery_duration_s=duration_s,
        voiced_start_s=voiced[0],
        voiced_end_s=voiced[1],
        longest_internal_silence_s=silence_s,
        internal_silences_s=silences,
    )


def inputs(
    seg: SegmentText,
    transcript: str | None = None,
    *,
    hints: Sequence[Hint] = (),
    embedding: tuple[float, ...] | None = None,
    measurement_record: MeasurementRecord | None = None,
    anchor: Anchor | None = None,
    similarity: SimilarityBaseline | None = None,
    pace_record: Pace | None = None,
    signal_stats: SignalStats | None = None,
    align: Alignment | None = None,
    hit_token_cap: bool = False,
    voice_transcript: str = VOICE_TRANSCRIPT,
) -> QaInputs:
    """QaInputs for one take; by default the transcript is the spoken text, heard perfectly."""
    return QaInputs(
        segment=seg,
        hints=tuple(hints),
        voice_transcript=voice_transcript,
        asr_text=seg.spoken_text if transcript is None else transcript,
        asr_words=(),
        embedding=embedding,
        alignment=align if align is not None else alignment(seg),
        signal=signal_stats if signal_stats is not None else signal(),
        hit_token_cap=hit_token_cap,
        measurement=measurement_record,
        anchor=anchor,
        similarity=similarity,
        pace=pace_record,
    )


def words(n: int, *, stem: str = "calm") -> str:
    """``n`` distinct plain words, for texts with an exact word count."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return " ".join(stem + alphabet[i // 26] + alphabet[i % 26] for i in range(n))


# ======================================================================== results


def scored(
    take_id: str,
    attempt: int,
    verdict: Verdict,
    *,
    flags: Sequence[Flag] = (),
    placed: bool = True,
    n_cues: int = 2,
) -> ScoredTake:
    """A scored take; with ``placed=False`` its cue 0 has null times (and should carry CUE_UNALIGNED)."""
    cues = tuple(
        CueTiming(index=i, start_s=None, end_s=None, confidence=None)
        if i == 0 and not placed
        else CueTiming(index=i, start_s=1.0 * i, end_s=1.0 * i + 0.9, confidence=0.9)
        for i in range(n_cues)
    )
    return ScoredTake(take_id=take_id, attempt=attempt, verdict=verdict, flags=tuple(flags), cues=cues)


def take_result(
    take_id: str,
    qa: QaResult | None,
    seg: SegmentText,
    *,
    attempt: int = 0,
    flags: Sequence[Flag] = (),
    duration_s: float = 10.0,
    cues: tuple[CueTiming, ...] | None = None,
) -> TakeResult:
    """A get_results take around a QaResult (the view WP36 assembles)."""
    timing = cues if cues is not None else placed_cues(seg)
    take_qa = None
    if qa is not None:
        take_qa = TakeQa(
            verdict=qa.verdict,
            flags=qa.flags,
            wer_raw=qa.metrics.wer_raw,
            wer_adj=qa.metrics.wer_adj,
            exact_ok=qa.metrics.exact_ok,
            exact=qa.exact,
            terms=qa.terms,
            spk_sim_anchor=qa.metrics.spk_sim_anchor,
            pace=PaceValue(spoken_wpm=qa.metrics.spoken_wpm, articulation_cps=qa.metrics.articulation_cps),
            pace_expected=PaceValue(
                spoken_wpm=qa.metrics.expected_spoken_wpm, articulation_cps=qa.metrics.expected_articulation_cps
            ),
        )
    return TakeResult(
        take_id=take_id,
        render_id="rn_" + take_id[3:],
        attempt=attempt,
        seed=1000 + attempt,
        fresh=True,
        delivery=DeliveryAudio(
            path="<store_root>/takes/x/delivery.wav",
            sha256="e" * 64,
            sample_rate=48000,
            samples=int(duration_s * 48000),
            duration_s=duration_s,
        ),
        trim=Trim(head_s=0.2, tail_s=0.3, pad_s=0.08),
        loudness=Loudness(measured_lufs=-16.0, gain_db=3.0, true_peak_dbtp=-2.0, ceiling_applied=False),
        analysis_id="an_" + take_id[3:],
        cues=timing,
        alignment=TakeAlignment(
            method=ALIGNMENT_METHOD,
            model=MODEL_ALIGNER,
            revision="0" * 40,
            cross_check="whisper-large-v3 word timestamps",
            max_disagreement_s=0.05,
            measured_error=None,
            flags=(),
        ),
        qa=take_qa,
        flags=tuple(flags),
    )


def segment_result(
    seg: SegmentText,
    takes: Sequence[TakeResult],
    *,
    suggested: str | None,
    flags: Sequence[Flag] = (),
    cue_warnings: dict[int, Sequence[Flag]] | None = None,
) -> SegmentResult:
    cues = tuple(
        CueText(
            index=c.index,
            received=c.received,
            spoken=c.spoken,
            engine=c.engine,
            spoken_span=c.spoken_span,
            engine_span=c.engine_span,
            warnings=tuple((cue_warnings or {}).get(c.index, ())),
            exact=c.exact,
        )
        for c in seg.cues
    )
    return SegmentResult(
        segment_id=seg.segment_id,
        status="passed",
        suggested_take_id=suggested,
        suggestion=None,
        text=SegmentResultText(spoken_chars=seg.spoken_chars, cues=cues),
        takes=tuple(takes),
        flags=tuple(flags),
    )
