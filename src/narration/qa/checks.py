"""Signal, speaker, pace and cue-alignment checks for one take (design section 11.1 steps 1, 3, 8 and 9).

Pace is spoken characters per second of speaking time, by the one rule the voice's measurement also uses
(``narration.qa.pace``; WP47).

Each check has a small numeric core (``*_flags`` taking plain numbers) and a wrapper that derives those
numbers from the take's records, so every threshold edge can be tested directly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from narration.config import MeasurementConfig
from narration.contracts import codes
from narration.contracts.interfaces import SignalStats
from narration.contracts.models import Alignment, Anchor, Flag, Pace, SegmentText, SimilarityBaseline
from narration.text import words

from ._flags import make_flag
from .errors import QaUnavailable
from .pace import expected_cps, expected_wpm_at, take_rate
from .profile import QaProfile

__all__ = [
    "PaceCheck",
    "SpeakerCheck",
    "alignment_flags",
    "cosine",
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


def pace_flags(
    cps: float | None,
    expected: float | None,
    tol: float | None,
    profile: QaProfile,
    *,
    pause_s: float | None = None,
) -> tuple[Flag, ...]:
    """``PACE_FAST`` warn above ``expected x (1 + tol)``; ``PACE_SLOW`` warn below ``expected x (1 - tol)``.

    ``cps`` is the take's pace and ``expected`` the voice's curve at its length, both in spoken characters per
    second of speaking time (``narration.qa.pace``). ``pause_s`` is the pause time taken out of the voiced span,
    reported in ``details`` (None: the silences were not measured, and the pace is over the whole span).

    ``default.v5`` has no pace fail (``QaProfile.pace_fail_tol_factor`` is None; DC-19): a fast take only warns,
    which is never a retake trigger, and ``details.fast_fail_above`` is null. A profile with a factor ``f``
    fails ``PACE_FAST`` above ``expected x (1 + f tol)`` (``default.v3``: f = 2). There is no slow fail."""
    if cps is None or expected is None or tol is None:
        return ()
    factor = profile.pace_fail_tol_factor
    fast_warn = expected * (1 + tol)
    fast_fail = expected * (1 + factor * tol) if factor is not None else None
    slow_warn = expected * (1 - tol)
    details = {
        "articulation_cps": round(cps, 3),
        "expected_articulation_cps": round(expected, 3),
        "tol": tol,
        "fast_warn_above": round(fast_warn, 3),
        "fast_fail_above": round(fast_fail, 3) if fast_fail is not None else None,
        "slow_warn_below": round(slow_warn, 3),
        "pause_s": round(pause_s, 3) if pause_s is not None else None,
        "basis": "speaking_time" if pause_s is not None else "voiced_span",
    }
    if fast_fail is not None and cps > fast_fail:
        return (
            make_flag(
                codes.PACE_FAST,
                "fail",
                f"{cps:.1f} characters per second of speaking against {expected:.1f} expected at this length "
                f"(fail above {fast_fail:.1f})",
                details=details,
            ),
        )
    if cps > fast_warn:
        return (
            make_flag(
                codes.PACE_FAST,
                "warn",
                f"{cps:.1f} characters per second of speaking against {expected:.1f} expected at this length "
                f"(warn above {fast_warn:.1f})",
                details=details,
            ),
        )
    if cps < slow_warn:
        return (
            make_flag(
                codes.PACE_SLOW,
                "warn",
                f"{cps:.1f} characters per second of speaking against {expected:.1f} expected at this length "
                f"(warn below {slow_warn:.1f})",
                details=details,
            ),
        )
    return ()


@dataclass(frozen=True, slots=True)
class PaceCheck:
    """A take's pace (``narration.qa.pace``) against the voice's curve: ``articulation_cps`` is judged,
    against ``expected_cps``; ``spoken_wpm`` and ``spoken_cps`` are over the whole voiced span (information);
    ``expected_wpm`` is ``spoken_wpm`` at the expected pace (``pace.expected_wpm_at``)."""

    spoken_wpm: float | None
    spoken_cps: float | None
    articulation_cps: float | None
    pause_s: float | None
    expected_cps: float | None
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
    """Spoken characters per second of speaking time (the voiced span less its pauses), against the curve.

    The tolerance is the measurement's ``tol``, never below ``pace_tol_min``. No voiced span means no pace;
    no curve means no comparison.
    """
    count = spoken_words(segment.spoken_text)
    rate = take_rate(segment.spoken_chars, count, signal)
    expected = expected_cps(pace, segment.spoken_chars) if pace is not None else None
    tol = max(pace.tol, config.pace_tol_min) if pace is not None else None
    return PaceCheck(
        spoken_wpm=rate.spoken_wpm,
        spoken_cps=rate.spoken_cps,
        articulation_cps=rate.articulation_cps,
        pause_s=rate.pause_s,
        expected_cps=expected,
        expected_wpm=expected_wpm_at(rate, count, expected, segment.spoken_chars),
        tol=tol,
        flags=pace_flags(rate.articulation_cps, expected, tol, profile, pause_s=rate.pause_s),
    )


# ======================================================================== step 3: cue alignment


def alignment_flags(alignment: Alignment, segment: SegmentText) -> tuple[Flag, ...]:
    """The aligner's flags, carried into QA, plus ``CUE_UNALIGNED`` for any cue left without times.

    Cue times are never interpolated: a cue with no timing, or null times, is unplaced, and QA makes sure it
    carries ``CUE_UNALIGNED`` (a retake trigger) even if the aligner did not say so; QA cannot tell why, so its
    ``details.reason`` is ``not_placed`` (``codes.CUE_NOT_PLACED``), and every ``CUE_UNALIGNED`` has a reason.
    Each carried flag's ``retake_trigger`` is set again by the contract's rule, from its code, severity and
    details: a ``CUE_UNALIGNED`` whose ``details.reason`` is ``no_alignable_words`` is not a trigger (DC-12).
    Like every QA flag, these carry no segment id (``QaScorer.score``).
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
                    f"cue {cue.index} could not be placed in the audio, and the aligner gave no reason; its times "
                    "are null (never interpolated)",
                    cue=cue.index,
                    details={"reason": codes.CUE_NOT_PLACED},
                )
            )
    return tuple(flags)
