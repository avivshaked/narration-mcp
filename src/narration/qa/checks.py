"""Signal, speaker, pace and cue-alignment checks for one take (design section 11.1 steps 1, 3, 8 and 9).

Each check has a small numeric core (``*_flags`` taking plain numbers) and a wrapper that derives those
numbers from the take's records, so every threshold edge can be tested directly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from narration.config import MeasurementConfig
from narration.contracts import codes
from narration.contracts.interfaces import SignalStats
from narration.contracts.models import Alignment, Anchor, Flag, Pace, SegmentText, SimilarityBaseline
from narration.text import words

from ._flags import make_flag
from .errors import QaUnavailable
from .profile import QaProfile

__all__ = [
    "PaceCheck",
    "SpeakerCheck",
    "alignment_flags",
    "cosine",
    "expected_wpm",
    "pace_check",
    "pace_flags",
    "signal_flags",
    "speaker_check",
    "speaker_flags",
    "spoken_words",
]


# ======================================================================== step 1: signal


def signal_flags(signal: SignalStats, hit_token_cap: bool, profile: QaProfile) -> tuple[Flag, ...]:
    """Token cap, non-finite samples and DC offset, clipping on the raw audio, and the longest internal silence."""
    flags: list[Flag] = []
    if hit_token_cap:
        flags.append(
            make_flag(
                codes.TOKEN_CAP_HIT,
                "fail",
                "generation stopped at max_new_tokens, so the model stopped mid-text and the take is cut short",
            )
        )
    if signal.raw_nonfinite:
        flags.append(
            make_flag(
                codes.SIGNAL_INVALID,
                "fail",
                "the raw audio has non-finite samples (NaN or infinity); the render is broken",
                details={"nonfinite": True},
            )
        )
    elif abs(signal.raw_dc_offset) > profile.dc_warn_above:
        flags.append(
            make_flag(
                codes.SIGNAL_INVALID,
                "warn",
                f"the raw audio has a DC offset of {signal.raw_dc_offset:.4f} (warn above {profile.dc_warn_above})",
                details={"dc_offset": signal.raw_dc_offset, "warn_above": profile.dc_warn_above},
            )
        )
    if signal.raw_clipping_fraction > profile.clipping_warn_above:
        flags.append(
            make_flag(
                codes.CLIPPING,
                "warn",
                f"{signal.raw_clipping_fraction:.4%} of the raw samples are at full scale "
                f"(warn above {profile.clipping_warn_above:.2%})",
                details={"fraction": signal.raw_clipping_fraction, "warn_above": profile.clipping_warn_above},
            )
        )
    silence = signal.longest_internal_silence_s
    details = {
        "longest_silence_s": silence,
        "warn_above_s": profile.silence_warn_above_s,
        "fail_above_s": profile.silence_fail_above_s,
    }
    if silence > profile.silence_fail_above_s:
        flags.append(
            make_flag(
                codes.SILENCE_LONG,
                "fail",
                f"a {silence:.2f} s silence inside the take (fail above {profile.silence_fail_above_s} s): "
                "a dropout or a hang",
                details=details,
            )
        )
    elif silence > profile.silence_warn_above_s:
        flags.append(
            make_flag(
                codes.SILENCE_LONG,
                "warn",
                f"a {silence:.2f} s silence inside the take (warn above {profile.silence_warn_above_s} s)",
                details=details,
            )
        )
    return tuple(flags)


# ======================================================================== step 8: speaker


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two embeddings; raises ValueError for different lengths or a zero vector."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"embeddings of different shapes: {x.shape} and {y.shape}")
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        raise ValueError("a zero embedding has no direction")
    return float(np.dot(x, y) / (nx * ny))


def speaker_flags(similarity: float | None, warn_below: float | None, fail_below: float) -> tuple[Flag, ...]:
    """``SPK_SIM_LOW``: fail below the absolute floor, warn below the voice's measured threshold."""
    if similarity is None:
        return ()
    details = {"similarity": similarity, "warn_below": warn_below, "fail_below": fail_below}
    if similarity < fail_below:
        return (
            make_flag(
                codes.SPK_SIM_LOW,
                "fail",
                f"similarity to the voice's anchor {similarity:.4f} is below the floor {fail_below}",
                details=details,
            ),
        )
    if warn_below is not None and similarity < warn_below:
        return (
            make_flag(
                codes.SPK_SIM_LOW,
                "warn",
                f"similarity to the voice's anchor {similarity:.4f} is below this voice's threshold {warn_below}",
                details=details,
            ),
        )
    return ()


@dataclass(frozen=True, slots=True)
class SpeakerCheck:
    similarity: float | None
    warn_below: float | None
    fail_below: float
    flags: tuple[Flag, ...]


def speaker_check(
    embedding: Sequence[float] | None,
    anchor: Anchor | None,
    baseline: SimilarityBaseline | None,
    config: MeasurementConfig,
) -> SpeakerCheck:
    """Similarity of the take's embedding to the voice's anchor, against thresholds from its measurement.

    The warn threshold is ``anchor_p5 - sim_warn_margin`` (rounded to 6 places), the fail threshold the
    absolute floor ``sim_fail_floor``. Without an anchor there is nothing to compare with and no flag (an
    audition); without a baseline only the floor applies.

    Raises ``QaUnavailable`` when there is an anchor but the take has no embedding, or the embeddings cannot be
    compared: the speaker check is called for and cannot run, and a take is never passed without it.
    """
    warn_below = round(baseline.anchor_p5 - config.sim_warn_margin, 6) if baseline is not None else None
    fail_below = config.sim_fail_floor
    if anchor is None:
        similarity = None
    elif embedding is None:
        raise QaUnavailable("the voice has an anchor but the take has no speaker embedding to compare with it")
    else:
        try:
            similarity = cosine(embedding, anchor.embedding)
        except ValueError as exc:
            raise QaUnavailable(f"the take's embedding cannot be compared with the anchor: {exc}") from exc
    return SpeakerCheck(
        similarity=similarity,
        warn_below=warn_below,
        fail_below=fail_below,
        flags=speaker_flags(similarity, warn_below, fail_below),
    )


# ======================================================================== step 9: pace


def spoken_words(text: str) -> int:
    """Spoken words, counted as the text pipeline counts words (``narration.text.words``: a token of
    punctuation only is not a word)."""
    return len(words(text))


def expected_wpm(pace: Pace, spoken_chars: int) -> float | None:
    """The voice's pace curve at this spoken length, in spoken words per minute.

    Inside the curve's range: straight lines between its points. Outside it: the nearest end point moved along
    the trend's slope (``per_100_chars``), so the value is continuous and follows the fitted trend. With no
    curve: the trend itself. None when the result is not a positive pace.
    """
    slope = pace.trend.per_100_chars / 100.0
    by_chars: dict[int, list[float]] = {}
    for point in pace.curve:
        by_chars.setdefault(point.chars, []).append(point.wpm)
    points = sorted((c, sum(v) / len(v)) for c, v in by_chars.items())
    if not points:
        value = pace.trend.intercept_wpm + slope * spoken_chars
    elif spoken_chars <= points[0][0]:
        value = points[0][1] + slope * (spoken_chars - points[0][0])
    elif spoken_chars >= points[-1][0]:
        value = points[-1][1] + slope * (spoken_chars - points[-1][0])
    else:
        value = points[0][1]
        for (c0, w0), (c1, w1) in pairwise(points):
            if c0 <= spoken_chars <= c1:
                value = w0 + (w1 - w0) * (spoken_chars - c0) / (c1 - c0)
                break
    return value if value > 0 else None


def pace_flags(
    wpm: float | None,
    expected: float | None,
    tol: float | None,
    profile: QaProfile,
) -> tuple[Flag, ...]:
    """``PACE_FAST`` warn above ``expected x (1 + tol)`` and fail above ``expected x (1 + 2 tol)``;
    ``PACE_SLOW`` warn below ``expected x (1 - tol)`` (default.v3 has no slow fail)."""
    if wpm is None or expected is None or tol is None:
        return ()
    fast_warn = expected * (1 + tol)
    fast_fail = expected * (1 + profile.pace_fail_tol_factor * tol)
    slow_warn = expected * (1 - tol)
    details = {
        "spoken_wpm": round(wpm, 3),
        "expected_spoken_wpm": round(expected, 3),
        "tol": tol,
        "fast_warn_above": round(fast_warn, 3),
        "fast_fail_above": round(fast_fail, 3),
        "slow_warn_below": round(slow_warn, 3),
    }
    if wpm > fast_fail:
        return (
            make_flag(
                codes.PACE_FAST,
                "fail",
                f"{wpm:.0f} spoken wpm against {expected:.0f} expected at this length (fail above {fast_fail:.0f})",
                details=details,
            ),
        )
    if wpm > fast_warn:
        return (
            make_flag(
                codes.PACE_FAST,
                "warn",
                f"{wpm:.0f} spoken wpm against {expected:.0f} expected at this length (warn above {fast_warn:.0f})",
                details=details,
            ),
        )
    if wpm < slow_warn:
        return (
            make_flag(
                codes.PACE_SLOW,
                "warn",
                f"{wpm:.0f} spoken wpm against {expected:.0f} expected at this length (warn below {slow_warn:.0f})",
                details=details,
            ),
        )
    return ()


@dataclass(frozen=True, slots=True)
class PaceCheck:
    spoken_wpm: float | None
    spoken_cps: float | None
    expected_wpm: float | None
    tol: float | None
    flags: tuple[Flag, ...]


def pace_check(
    segment: SegmentText,
    signal: SignalStats,
    pace: Pace | None,
    config: MeasurementConfig,
    profile: QaProfile,
) -> PaceCheck:
    """Spoken words per minute (and spoken characters per second) over the voiced span, against the curve.

    The tolerance is the measurement's ``tol``, never below ``pace_tol_min``. No voiced span means no pace;
    no curve means no comparison.
    """
    start, end = signal.voiced_start_s, signal.voiced_end_s
    span = end - start if start is not None and end is not None else None
    if span is None or span <= 0:
        wpm = cps = None
    else:
        wpm = spoken_words(segment.spoken_text) / span * 60.0
        cps = segment.spoken_chars / span
    expected = expected_wpm(pace, segment.spoken_chars) if pace is not None else None
    tol = max(pace.tol, config.pace_tol_min) if pace is not None else None
    return PaceCheck(
        spoken_wpm=wpm,
        spoken_cps=cps,
        expected_wpm=expected,
        tol=tol,
        flags=pace_flags(wpm, expected, tol, profile),
    )


# ======================================================================== step 3: cue alignment


def alignment_flags(alignment: Alignment, segment: SegmentText) -> tuple[Flag, ...]:
    """The aligner's flags, carried into QA, plus ``CUE_UNALIGNED`` for any cue left without times.

    Cue times are never interpolated: a cue with no timing, or null times, is unplaced, and QA makes sure it
    carries ``CUE_UNALIGNED`` (a retake trigger) even if the aligner did not say so. Each carried flag's
    ``retake_trigger`` is set again by the contract's rule, from its code, severity and details: a
    ``CUE_UNALIGNED`` whose ``details.reason`` is ``no_alignable_words`` is not a trigger (DC-12). Like every
    QA flag, these carry no segment id (``QaScorer.score``).
    """
    flags = [
        dataclasses.replace(f, segment_id=None, retake_trigger=codes.is_retake_trigger(f.code, f.severity, f.details))
        for f in alignment.flags
    ]
    placed = {c.index for c in alignment.cues if c.start_s is not None and c.end_s is not None}
    flagged = {f.cue for f in flags if f.code == codes.CUE_UNALIGNED}
    for cue in segment.cues:
        if cue.index not in placed and cue.index not in flagged:
            flags.append(
                make_flag(
                    codes.CUE_UNALIGNED,
                    "warn",
                    f"cue {cue.index} could not be placed in the audio; its times are null (never interpolated)",
                    cue=cue.index,
                )
            )
    return tuple(flags)
