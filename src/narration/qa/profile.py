"""The QA profile ``default.v5``: every fixed threshold of design section 11.1's table, in one record.

The speaker margins and the pace-tolerance floor are not here: they are configuration (``[measurement]`` in
section 16: ``sim_warn_margin``, ``sim_fail_floor``, ``pace_tol_min``), and the per-voice parts (``anchor_p5``,
the pace curve and ``tol``) come from the voice's measurement.

The values the design's table does not give (reference bleed) are marked ASSUME below. They are this work
package's placeholders, reported in every flag's ``details`` so a reader can see what was applied; the profile
name must change with any of them.
"""

from __future__ import annotations

from dataclasses import dataclass

from narration.contracts.names import QA_PROFILE

__all__ = ["DEFAULT_PROFILE", "QaProfile"]


@dataclass(frozen=True, slots=True, kw_only=True)
class QaProfile:
    """The fixed thresholds of one QA profile. Comparisons are strict where the design says "above"/"below"
    (a value exactly at a threshold does not trigger it) and inclusive where it says "at least"."""

    name: str = QA_PROFILE

    # Text match (section 11.1 step 4): warn when wer_adj > 0.02 with >= 1 word error,
    # fail when wer_adj > 0.06 with >= 2 word errors.
    wer_warn_above: float = 0.02
    wer_warn_min_errors: int = 1
    wer_fail_above: float = 0.06
    wer_fail_min_errors: int = 2

    # Terms (step 6): a letters-only fuzzy ratio of at least 0.75, or an alias, is ok.
    term_min_ratio: float = 0.75

    # Insertions (step 7): warn at >= 1 word, fail at >= 3 words (or on reference bleed, at the head).
    insertion_warn_words: int = 1
    insertion_fail_words: int = 3

    # Signal (step 1): longest internal silence warn > 1.2 s, fail > 2.5 s; clipping warn > 0.01 % of samples.
    silence_warn_above_s: float = 1.2
    silence_fail_above_s: float = 2.5
    clipping_warn_above: float = 0.0001

    # Pace (step 9): warn outside curve x (1 +/- tol), where pace is spoken characters per second of speaking
    # time (``narration.qa.pace``; ``default.v5``, WP47). ``default.v4`` and ``v5`` have no pace fail: the owner
    # decided (2026-09-27) that PACE_FAST warns only, until WP47's rate model is validated (DC-19), because the
    # words-per-minute pace model failed short paragraphs that listened fine. A factor f, when set, fails
    # PACE_FAST above curve x (1 + f tol); ``default.v3`` had f = 2.0. None means there is no fail line.
    pace_fail_tol_factor: float | None = None

    # ASSUME (not in the design's table; to be measured on planted renders in WP40): reference bleed at the
    # head is a fuzzy match (letters only, this ratio or more) of the head's extra words against as many letters
    # at the end, or the start, of the voice's transcript. It is judged only for a head of at least
    # ``bleed_min_words`` words or at least ``bleed_min_letters`` letters: a shorter head matches by chance
    # (one 4-letter word can match 3 letters of any transcript).
    bleed_min_ratio: float = 0.75
    bleed_min_words: int = 2
    bleed_min_letters: int = 8

    # Signal (step 1, SIGNAL_INVALID; plan.md DC-5 and DC-10): any non-finite sample fails; a DC offset of the
    # raw audio above this (full scale 1.0) warns.
    dc_warn_above: float = 0.001


DEFAULT_PROFILE = QaProfile()
"""``default.v5``: design section 11.1's table, except that ``PACE_FAST`` never fails (DC-19), with pace in
spoken characters per second of speaking time (DC-18)."""
