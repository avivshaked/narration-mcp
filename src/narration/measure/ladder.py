"""The length ladder's judgement (design section 3.2, R8): the pace trend, ``tol``, whether each rung passes,
``max_segment_chars``, ``max_segment_seconds`` and the pace curve.

Pure functions over what QA measured on each rung's takes (``SeedTake``: one per seed). The rules, as section
3.2 states them, with the choices it leaves open made explicit:

- **Pace** is spoken characters per second of speaking time: the voiced span with its pauses taken out (QA's
  ``articulation_cps``; ``narration.qa.pace``, ``names.PACE_METHOD``). Every take of the ladder is scored by
  the same QA as a caller's take, so the curve and a take's pace are one quantity. (Before WP47 it was spoken
  words per minute, which followed the corpus's word lengths rather than the voice.)
- **The trend** is a straight line of pace against length, fitted by ordinary least squares over the rungs of
  the trend band (target length at most ``trend_band_max_chars``, 300 by default), one point per rung: its
  spoken length and the median pace over its seeds. One point per rung, not per take, so a rung counts once
  and one odd seed cannot tilt the line. A band with a single length gives a flat line through it.
- **The seed spread** of a rung is (highest - lowest) / median of its seeds' paces. ``tol`` is the larger of
  ``pace_tol_min`` (0.10) and the largest spread in the band, so it is never below the spread the voice showed.
  It is derived exactly as it was in words per minute, on the new pace: a relative spread has no unit.
- **The speaking share** is the median over the band's takes of speaking time / voiced span (QA's
  ``spoken_cps`` / ``articulation_cps``, which have the same characters). It is only for duration estimates.
- **A rung passes** when all of these hold:
  - the median pace over its seeds is at most the trend at the rung's length x (1 + ``tol``);
  - the medians of ``wer_adj`` and of the word errors do not meet QA's fail rule (``wer_adj`` above 0.06 with
    at least 2 word errors; ``QaProfile``). That is what "the median ``wer_adj`` passes" means (the lead's
    ruling): the fail rule already holds a one-word allowance, so one slip in a short rung does not end the
    ladder, as section 11.1 argues for a short segment;
  - the median similarity to the anchor is at least the warn threshold, ``anchor_p5 - sim_warn_margin``;
  - no seed has an exact-span mismatch, a head insertion (``HEAD_INSERTION`` at any severity) or a token-cap
    hit.
  A rung with no measurable pace, ``wer_adj`` or similarity does not pass: it was not shown to read reliably.
- **``max_segment_chars``** is the spoken length of the longest rung in the run of passing rungs counted up
  from the shortest; the first failing rung ends the run. It is the paragraph's own spoken length (the rung
  ``ladder-150`` may be 151 characters), because that is what the voice read. **``max_segment_seconds``** is
  the median delivery duration over that rung's seeds. With no passing rung both are None.
- **The pace curve** is the median pace by spoken length over the passing run, which QA reads at a segment's
  length (``narration.qa.pace.expected_cps``).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from narration.contracts.models import LadderRung, LadderSeed, PacePoint, PaceTrend
from narration.contracts.names import Verdict
from narration.qa.profile import DEFAULT_PROFILE, QaProfile

DECIMALS: Final = 6
"""Rounding of the ladder's numbers in the measurement."""


@dataclass(frozen=True, slots=True)
class SeedTake:
    """One seed's take of a rung, as QA measured it. ``cps`` is its pace (QA's ``articulation_cps``);
    ``spoken_cps`` and ``wpm`` are over the whole voiced span (information, and the speaking share). ``flags``
    are its QA flag codes."""

    seed: int
    attempt: int
    take_id: str | None
    cps: float | None
    wer_adj: float | None
    word_errors: int | None
    sim: float | None
    verdict: Verdict
    duration_s: float | None
    exact_mismatch: bool = False
    head_insertion: bool = False
    token_cap: bool = False
    flags: tuple[str, ...] = ()
    spoken_cps: float | None = None
    wpm: float | None = None


@dataclass(frozen=True, slots=True)
class Rung:
    """One rung of the ladder: its paragraph, its target length (``length_ladder_spoken_chars``), the
    paragraph's spoken length, and its seeds' takes."""

    paragraph_id: str
    target: int
    chars: int
    seeds: tuple[SeedTake, ...]


@dataclass(frozen=True, slots=True)
class RungJudgement:
    """Whether a rung passes, why not, and the medians it was judged on."""

    passes: bool
    reasons: tuple[str, ...] = ()
    median_cps: float | None = None
    limit_cps: float | None = None
    median_wer_adj: float | None = None
    median_word_errors: float | None = None
    median_sim: float | None = None
    median_duration_s: float | None = None


@dataclass(frozen=True, slots=True)
class LadderOutcome:
    """The ladder's result: the passing run's length, the limit and the curve from it."""

    run: int
    """How many rungs, counted up from the shortest, pass before the first that fails."""
    max_segment_chars: int | None
    max_segment_seconds: float | None
    curve: tuple[PacePoint, ...] = field(default=())


def median(values: Sequence[float | None]) -> float | None:
    """The median of the values that are not None, or None when there are none."""
    present = [v for v in values if v is not None]
    return float(statistics.median(present)) if present else None


def in_band(rung: Rung, band_max_chars: int) -> bool:
    """Whether a rung is in the trend band: its target length is at most ``band_max_chars``."""
    return rung.target <= band_max_chars


def fit_trend(rungs: Sequence[Rung], band_max_chars: int) -> PaceTrend:
    """The pace trend over the band's rungs: least squares through (spoken length, median pace) per rung.

    Raises ValueError when no rung in the band has a measured pace."""
    points: list[tuple[int, float]] = []
    for rung in rungs:
        pace = median([s.cps for s in rung.seeds]) if in_band(rung, band_max_chars) else None
        if pace is not None:
            points.append((rung.chars, pace))
    if not points:
        raise ValueError("no rung in the trend band has a measured pace")
    xs = [float(x) for x, _ in points]
    ys = [y for _, y in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    var = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / var if var > 0 else 0.0
    return PaceTrend(
        intercept_cps=round(mean_y - slope * mean_x, DECIMALS),
        per_100_chars=round(slope * 100.0, DECIMALS),
        band_max_chars=band_max_chars,
    )


def trend_at(trend: PaceTrend, chars: int) -> float:
    """The trend's pace at a spoken length, in spoken characters per second of speaking time."""
    return trend.intercept_cps + trend.per_100_chars * chars / 100.0


def seed_spread(rung: Rung) -> float:
    """(highest - lowest) / median of the rung's seeds' paces; 0 with fewer than two."""
    paces = [s.cps for s in rung.seeds if s.cps is not None]
    if len(paces) < 2:
        return 0.0
    mid = statistics.median(paces)
    return (max(paces) - min(paces)) / mid if mid > 0 else 0.0


def pace_tol(rungs: Sequence[Rung], band_max_chars: int, tol_min: float) -> float:
    """``tol = max(pace_tol_min, the largest seed spread in the trend band)``."""
    spreads = [seed_spread(r) for r in rungs if in_band(r, band_max_chars)]
    return round(max([tol_min, *spreads]), DECIMALS)


def speaking_share(rungs: Sequence[Rung], band_max_chars: int) -> float:
    """The median over the band's takes of speaking time / voiced span (``spoken_cps / cps``: both have the
    rung's spoken characters), at most 1; 1.0 when no take of the band has both (then no pause is assumed)."""
    shares = [
        min(1.0, s.spoken_cps / s.cps)
        for r in rungs
        if in_band(r, band_max_chars)
        for s in r.seeds
        if s.spoken_cps is not None and s.cps is not None and s.cps > 0
    ]
    found = median(shares)
    return round(found, DECIMALS) if found is not None else 1.0


def judge(
    rung: Rung,
    *,
    trend: PaceTrend,
    tol: float,
    sim_warn: float,
    profile: QaProfile = DEFAULT_PROFILE,
) -> RungJudgement:
    """Whether the rung passes (the module docstring), with the reasons it does not."""
    reasons: list[str] = []
    cps = median([s.cps for s in rung.seeds])
    limit = trend_at(trend, rung.chars) * (1.0 + tol)
    if cps is None:
        reasons.append("no seed's pace could be measured")
    elif cps > limit:
        reasons.append(
            f"median pace {cps:.2f} characters per second of speaking time is above the trend x (1 + tol) = "
            f"{limit:.2f}"
        )
    wer = median([s.wer_adj for s in rung.seeds])
    errors = median([None if s.word_errors is None else float(s.word_errors) for s in rung.seeds])
    if wer is None or errors is None:
        reasons.append("no seed's wer_adj could be measured")
    elif wer > profile.wer_fail_above and errors >= profile.wer_fail_min_errors:
        reasons.append(
            f"median wer_adj {wer:.3f} with a median of {errors:g} word errors meets the fail rule "
            f"(above {profile.wer_fail_above} with at least {profile.wer_fail_min_errors})"
        )
    sim = median([s.sim for s in rung.seeds])
    if sim is None:
        reasons.append("no seed's similarity to the anchor could be measured")
    elif sim < sim_warn:
        reasons.append(f"median similarity {sim:.4f} is below the warn threshold {sim_warn}")
    for what, found in (
        ("an exact-span mismatch", [s for s in rung.seeds if s.exact_mismatch]),
        ("a head insertion", [s for s in rung.seeds if s.head_insertion]),
        ("a token-cap hit", [s for s in rung.seeds if s.token_cap]),
    ):
        if found:
            reasons.append(f"attempt {', '.join(str(s.attempt) for s in found)} had {what}")
    return RungJudgement(
        passes=not reasons,
        reasons=tuple(reasons),
        median_cps=_round(cps),
        limit_cps=_round(limit),
        median_wer_adj=_round(wer),
        median_word_errors=errors,
        median_sim=_round(sim),
        median_duration_s=_round(median([s.duration_s for s in rung.seeds])),
    )


def outcome(rungs: Sequence[Rung], judgements: Sequence[RungJudgement]) -> LadderOutcome:
    """The passing run counted up from the shortest rung, and ``max_segment_chars``, ``max_segment_seconds`` and
    the pace curve from it. ``rungs`` are in ladder order; ``judgements`` may stop early (the rungs above the
    first failing one are never judged)."""
    run = 0
    for judgement in judgements:
        if not judgement.passes:
            break
        run += 1
    if run == 0:
        return LadderOutcome(run=0, max_segment_chars=None, max_segment_seconds=None)
    passing = list(zip(rungs[:run], judgements[:run], strict=True))
    top_rung, top = passing[-1]
    curve = tuple(PacePoint(chars=r.chars, cps=j.median_cps) for r, j in passing if j.median_cps is not None)
    return LadderOutcome(
        run=run, max_segment_chars=top_rung.chars, max_segment_seconds=top.median_duration_s, curve=curve
    )


def ladder_rung(rung: Rung, judgement: RungJudgement) -> LadderRung:
    """The rung as the measurement's ladder table shows it (section 7.6: per rung and seed)."""
    return LadderRung(
        chars=rung.chars,
        paragraph_id=rung.paragraph_id,
        passes=judgement.passes,
        seeds=tuple(
            LadderSeed(
                seed=s.seed,
                attempt=s.attempt,
                take_id=s.take_id,
                cps=_round(s.cps),
                wer_adj=_round(s.wer_adj),
                sim=_round(s.sim),
                verdict=s.verdict,
                spoken_cps=_round(s.spoken_cps),
                wpm=_round(s.wpm),
                duration_s=_round(s.duration_s),
                flags=s.flags,
            )
            for s in rung.seeds
        ),
    )


def _round(value: float | None) -> float | None:
    return round(value, DECIMALS) if value is not None else None


__all__ = [
    "DECIMALS",
    "LadderOutcome",
    "Rung",
    "RungJudgement",
    "SeedTake",
    "fit_trend",
    "in_band",
    "judge",
    "ladder_rung",
    "median",
    "outcome",
    "pace_tol",
    "seed_spread",
    "speaking_share",
    "trend_at",
]
