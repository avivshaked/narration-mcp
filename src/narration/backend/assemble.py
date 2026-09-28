"""``get_results`` for a generation job, assembled exactly as design section 7.5 lays it out.

The job keeps its request by value and, per segment, every attempt it produced or found (``JobRecord.items``:
the render, take and analysis ids, whether this job rendered it, and its per-job flags). The results are
built from those and the cached records, and from the request itself:

- **The text echo** (R7) is the request's own, planned again from the stored request: a cached analysis
  keeps nothing a request sent beyond what its key covers (``received`` is its spoken form), so the echo
  never comes from it.
- **Restamping.** A cached analysis may serve many requests, so its flags carry no segment id and its exact
  results carry their word ranges. The current request's segment id is stamped on every flag, and each
  exact result, and each ``EXACT_SPAN_MISMATCH`` flag's details, get the request's own code-point offsets,
  matched by cue and word range (``QaScorer.score``'s docstring, section 11.3).
- **The measured error** of the aligner is read from the current alignment benchmark of the analysis's
  method (``Store.get_alignment_benchmark``), never from the cached analysis: a new benchmark of the same
  method changes no analysis key (section 11.2, R1).
- **Fit** is computed per request, only for a segment that gave ``scene_seconds`` (section 12).
- **Takes**: every attempt of every take slot, replaced ones included (section 8), each with ``fresh``.
- **Consistency** and the suggestions are the job's own advice (``JobRecord.result``); ``listen_first`` is
  QA's order over the suggested takes (section 11.1).

A job still queued or running returns what it has so far; a failed or cancelled job returns the segments it
finished and the others with their state. Nothing here changes a verdict.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Store, TextPlanner
from narration.contracts.models import (
    AnalysisRecord,
    Consistency,
    CueTiming,
    EngineRef,
    ExactResult,
    ExactWords,
    Flag,
    JobAttempt,
    JobRecord,
    JobSegment,
    Licence,
    MeasuredError,
    PaceValue,
    RenderRecord,
    SegmentIn,
    SegmentResult,
    SegmentResultText,
    SegmentText,
    Suggestion,
    TakeAlignment,
    TakeQa,
    TakeRecord,
    TakeResult,
)
from narration.contracts.serial import ContractError, from_json, to_json
from narration.qa import fit_report

from .requests import hints_of, segments_of

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Assembled:
    """A generation job's results as records, before they are turned into JSON."""

    segments: tuple[SegmentResult, ...]
    engine: EngineRef | None
    licence: Licence


def stamp(flag: Flag, segment_id: str) -> Flag:
    """The flag with the request's segment id."""
    return flag if flag.segment_id == segment_id else dataclasses.replace(flag, segment_id=segment_id)


def _exact_map(text: SegmentText) -> dict[tuple[int, tuple[int, int]], ExactWords]:
    return {(cue.index, e.words): e for cue in text.cues for e in cue.exact}


def restamp_exact(result: ExactResult, spans: Mapping[tuple[int, tuple[int, int]], ExactWords]) -> ExactResult:
    """An exact result with the request's offsets (matched by cue and word range)."""
    span = spans.get((result.cue, result.words))
    if span is None:
        return result
    return dataclasses.replace(result, start=span.start, end=span.end)


def restamp_flag(flag: Flag, segment_id: str, spans: Mapping[tuple[int, tuple[int, int]], ExactWords]) -> Flag:
    """A cached flag for this request: its segment id, and for ``EXACT_SPAN_MISMATCH`` the span's offsets."""
    flag = stamp(flag, segment_id)
    if flag.code != codes.EXACT_SPAN_MISMATCH or not flag.details:
        return flag
    details = dict(flag.details)
    words = details.get("words")
    cue = details.get("cue", flag.cue)
    if isinstance(cue, int) and isinstance(words, (list, tuple)) and len(words) == 2:  # pyright: ignore[reportUnknownArgumentType]
        key = (cue, (int(words[0]), int(words[1])))  # pyright: ignore[reportUnknownArgumentType]
        span = spans.get(key)
        if span is not None:
            details.update(start=span.start, end=span.end)
            return dataclasses.replace(flag, details=details)
    return flag


def measured_error(store: Store, method_id: str) -> MeasuredError | None:
    """The aligner method's published error, from its current benchmark (None until one exists)."""
    try:
        bench = store.get_alignment_benchmark(method_id)
    except (ValueError, NarrationError):  # a method id the store cannot name a file for
        return None
    if bench is None:
        return None
    return MeasuredError(
        p50_s=bench.measured_error.p50_s,
        p95_s=bench.measured_error.p95_s,
        n=bench.measured_error.n,
        benchmark=bench.benchmark.id,
        by_kind=dict(bench.by_kind),
    )


def _cues(cues: Sequence[CueTiming], include_words: bool) -> tuple[CueTiming, ...]:
    return tuple(cues) if include_words else tuple(dataclasses.replace(c, words=()) for c in cues)


def take_result(
    store: Store,
    *,
    attempt: JobAttempt,
    take: TakeRecord,
    render: RenderRecord,
    analysis: AnalysisRecord | None,
    segment: SegmentIn,
    text: SegmentText,
    include_words: bool,
    include_transcripts: bool,
) -> TakeResult:
    """One take as ``get_results`` shows it (section 7.5)."""
    segment_id = text.segment_id
    spans = _exact_map(text)
    alignment: TakeAlignment | None = None
    qa: TakeQa | None = None
    cues: tuple[CueTiming, ...] = ()
    if analysis is not None:
        a = analysis.alignment
        cross = a.cross_check
        alignment = TakeAlignment(
            method=a.method,
            model=a.model or "",
            revision=a.revision or "",
            cross_check=f"{cross.model} word timestamps",
            max_disagreement_s=cross.max_disagreement_s,
            measured_error=measured_error(store, analysis.versions.aligner_method),
            flags=tuple(restamp_flag(f, segment_id, spans) for f in a.flags),
        )
        q = analysis.qa
        qa = TakeQa(
            verdict=q.verdict,
            flags=tuple(restamp_flag(f, segment_id, spans) for f in q.flags),
            wer_raw=q.metrics.wer_raw,
            wer_adj=q.metrics.wer_adj,
            exact_ok=q.metrics.exact_ok,
            exact=tuple(restamp_exact(r, spans) for r in q.exact),
            terms=q.terms,
            spk_sim_anchor=q.metrics.spk_sim_anchor,
            pace=PaceValue(spoken_wpm=q.metrics.spoken_wpm, articulation_cps=q.metrics.articulation_cps),
            pace_expected=PaceValue(
                spoken_wpm=q.metrics.expected_spoken_wpm, articulation_cps=q.metrics.expected_articulation_cps
            ),
            transcript=q.transcript if include_transcripts else None,
        )
        cues = _cues(a.cues, include_words)
    fit = fit_report(take.delivery.duration_s, segment.scene_seconds, segment.fit, segment_id=segment_id)
    flags = (*(stamp(f, segment_id) for f in take.flags), *(stamp(f, segment_id) for f in attempt.flags))
    return TakeResult(
        take_id=take.take_id,
        render_id=render.render_id,
        attempt=attempt.attempt,
        seed=attempt.seed,
        fresh=attempt.fresh,
        delivery=take.delivery,
        trim=take.trim,
        loudness=take.loudness,
        analysis_id=analysis.analysis_id if analysis is not None else None,
        cues=cues,
        alignment=alignment,
        qa=qa,
        fit=fit,
        flags=flags,
    )


def _suggestions(job: JobRecord) -> dict[str, tuple[str | None, Suggestion | None]]:
    out: dict[str, tuple[str | None, Suggestion | None]] = {}
    for entry in (job.result or {}).get("suggestions") or ():
        if not isinstance(entry, Mapping):
            continue
        suggestion: Suggestion | None = None
        raw = entry.get("suggestion")  # pyright: ignore[reportUnknownMemberType]
        if isinstance(raw, Mapping):
            try:
                suggestion = from_json(Suggestion, dict(raw))  # pyright: ignore[reportUnknownArgumentType]
            except ContractError:
                suggestion = None
        take_id = entry.get("take_id")  # pyright: ignore[reportUnknownMemberType]
        out[str(entry.get("segment_id"))] = (take_id if isinstance(take_id, str) else None, suggestion)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return out


def consistency_of(job: JobRecord) -> Consistency:
    """The job's consistency report (a report on its suggested takes, never a verdict), or an empty one."""
    raw = (job.result or {}).get("consistency")
    if isinstance(raw, Mapping):
        try:
            return from_json(Consistency, dict(raw))  # pyright: ignore[reportUnknownArgumentType]
        except ContractError:
            log.warning("job %s: its consistency report cannot be read", job.job_id)
    return Consistency(min=None, median=None)


def assemble(
    store: Store,
    text_planner: TextPlanner,
    job: JobRecord,
    *,
    include_words: bool,
    include_transcripts: bool,
) -> Assembled:
    """The job's segments, in request order, with their text echo and every take found (module docstring)."""
    segments = segments_of(job.request)
    texts = text_planner.plan_request(segments, hints_of(job.request), strict_text=False)
    items: dict[str, JobSegment] = {item.segment_id: item for item in job.items}
    suggestions = _suggestions(job)
    engine: EngineRef | None = None
    licence = Licence(voice_clip="synthetic")
    out: list[SegmentResult] = []
    for segment, text in zip(segments, texts, strict=True):
        item = items.get(text.segment_id)
        takes: list[TakeResult] = []
        for attempt in item.attempts if item is not None else ():
            found = _records(store, attempt)
            if found is None:
                continue
            render, take, analysis = found
            engine = engine or EngineRef(id=render.engine.engine_profile_id, hash=render.engine.engine_profile_hash)
            licence = _merge_licence(licence, render, analysis)
            takes.append(
                take_result(
                    store,
                    attempt=attempt,
                    take=take,
                    render=render,
                    analysis=analysis,
                    segment=segment,
                    text=text,
                    include_words=include_words,
                    include_transcripts=include_transcripts,
                )
            )
        suggested, suggestion = suggestions.get(text.segment_id, (None, None))
        out.append(
            SegmentResult(
                segment_id=text.segment_id,
                status=item.state if item is not None else "planned",
                suggested_take_id=suggested,
                suggestion=suggestion,
                text=SegmentResultText(spoken_chars=text.spoken_chars, cues=text.cues),
                takes=tuple(takes),
                flags=tuple(stamp(f, text.segment_id) for f in item.flags) if item is not None else text.warnings,
            )
        )
    return Assembled(segments=tuple(out), engine=engine, licence=licence)


def _records(store: Store, attempt: JobAttempt) -> tuple[RenderRecord, TakeRecord, AnalysisRecord | None] | None:
    """The attempt's render, take and analysis, or None when it has no take (unfinished) or its records left
    the cache (collected after the retention period)."""
    if attempt.take_id is None or attempt.render_id is None:
        return None
    render = store.get_render_by_id(attempt.render_id)
    take = store.get_take_by_id(attempt.take_id)
    if render is None or take is None:
        log.info("take %s of attempt %s is no longer in the cache", attempt.take_id, attempt.attempt)
        return None
    analysis = store.get_analysis_by_id(attempt.analysis_id) if attempt.analysis_id is not None else None
    return render, take, analysis


def _merge_licence(licence: Licence, render: RenderRecord, analysis: AnalysisRecord | None) -> Licence:
    """Section 18: the generation model's, the voice clip's, and the QA and alignment models' licences."""
    other = analysis.licence if analysis is not None else Licence()
    return Licence(
        generation_model=licence.generation_model or render.licence.generation_model,
        voice_clip=licence.voice_clip or render.licence.voice_clip,
        aligner=licence.aligner or other.aligner,
        asr=licence.asr or other.asr,
        sv=licence.sv or other.sv,
    )


def segments_json(segments: Sequence[SegmentResult]) -> list[dict[str, Any]]:
    """The segments as JSON."""
    return [to_json(s) for s in segments]


__all__ = [
    "Assembled",
    "assemble",
    "consistency_of",
    "measured_error",
    "restamp_exact",
    "restamp_flag",
    "segments_json",
    "stamp",
    "take_result",
]
