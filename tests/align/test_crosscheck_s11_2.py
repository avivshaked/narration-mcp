"""The Whisper cross-check (design section 11.2 step 6): compared where the neighbouring words match, names
excluded; ``CUE_ALIGNMENT_DISAGREE`` above the threshold; ``max_disagreement_s`` reported."""

from __future__ import annotations

import pytest

from narration.align import Boundary, cross_check, interval_distance, plain
from narration.contracts import codes
from narration.contracts.models import Hint

from ._support import RATE, asr, audio, codes_of, reply, segment
from .test_resolve_s11_2 import ALIGNER, TWO_AUDIO, TWO_CUES, TWO_WORDS


def _resolve_with(asr_words, *, cues=TWO_CUES, hints=()):
    seg = segment(*cues, hints=hints)
    t = ALIGNER.build_transcript(seg, list(hints))
    return ALIGNER.resolve(t, reply(t, TWO_WORDS, samples=TWO_AUDIO.shape[0]), TWO_AUDIO, RATE, asr_words, None)


def test_boundaries_in_the_same_pause_agree_s11_2() -> None:
    # The service's boundary is the pause 1.40–1.90; Whisper stretched "turns" into it, as it often does.
    words = asr(("Then", 0.2, 0.5), ("it", 0.56, 0.8), ("turns.", 0.86, 1.62), ("For", 1.62, 2.2), ("home.", 2.26, 2.8))
    a = _resolve_with(words)
    assert a.cross_check.max_disagreement_s == 0.0
    assert codes_of(a) == []


def test_disagreement_above_the_threshold_is_flagged_s11_2() -> None:
    words = asr(("Then", 0.2, 0.5), ("it", 0.56, 0.8), ("turns.", 0.86, 2.3), ("For", 2.3, 2.5), ("home.", 2.5, 2.8))
    a = _resolve_with(words)
    assert a.cross_check.max_disagreement_s == 0.4
    [flag] = a.flags
    assert (flag.code, flag.severity, flag.cue, flag.retake_trigger) == (
        codes.CUE_ALIGNMENT_DISAGREE,
        "warn",
        0,
        False,
    )
    assert flag.details == {
        "next_cue": 1,
        "disagreement_s": 0.4,
        "threshold_s": 0.25,
        "boundary_s": [1.4, 1.9],
        "asr_boundary_s": [2.3, 2.3],
    }


def test_disagreement_at_or_below_the_threshold_is_reported_not_flagged_s11_2() -> None:
    words = asr(("Then", 0.2, 0.5), ("it", 0.56, 0.8), ("turns.", 0.86, 2.1), ("For", 2.15, 2.5), ("home", 2.5, 2.8))
    a = _resolve_with(words)
    assert a.cross_check.max_disagreement_s == 0.2
    assert codes_of(a) == []


def test_names_are_excluded_from_the_cross_check_s11_2() -> None:
    hint = Hint(term="turns")
    words = asr(("Then", 0.2, 0.5), ("it", 0.56, 0.8), ("turns.", 0.86, 2.3), ("For", 2.3, 2.5), ("home.", 2.5, 2.8))
    a = _resolve_with(words, hints=[hint])
    assert a.cross_check.max_disagreement_s is None
    assert codes_of(a) == []


def test_neighbours_whisper_did_not_hear_together_are_not_compared_s11_2() -> None:
    misheard = asr(("Then", 0.2, 0.5), ("it", 0.56, 0.8), ("terns", 0.86, 2.3), ("For", 2.3, 2.5), ("home", 2.5, 2.8))
    assert _resolve_with(misheard).cross_check.max_disagreement_s is None
    inserted = asr(("Then", 0.2, 0.5), ("it", 0.56, 0.8), ("turns", 0.86, 1.2), ("uh", 1.3, 1.5), ("for", 2.3, 2.5))
    assert _resolve_with(inserted).cross_check.max_disagreement_s is None
    untimed = asr(("Then", 0.2, 0.5), ("it", 0.56, 0.8), ("turns", 0.86, None), ("For", 2.3, 2.5), ("home", 2.5, 2.8))
    assert _resolve_with(untimed).cross_check.max_disagreement_s is None


def test_no_asr_words_means_no_cross_check_s11_2() -> None:
    assert _resolve_with([]).cross_check.max_disagreement_s is None


def test_unplaced_cues_are_not_cross_checked_s11_2() -> None:
    speech = audio(3.0, [(0.20, 1.40), (1.90, 2.80)])
    t = ALIGNER.build_transcript(segment(*TWO_CUES), [])
    r = reply(t, TWO_WORDS, samples=speech.shape[0], scores={1: 0.1})
    words = asr(("Then", 0.2, 0.5), ("it", 0.56, 0.8), ("turns", 0.86, 2.3), ("For", 2.3, 2.5), ("home", 2.5, 2.8))
    a = ALIGNER.resolve(t, r, speech, RATE, words, None)
    assert a.cross_check.max_disagreement_s is None
    assert codes.CUE_ALIGNMENT_DISAGREE not in [f.code for f in a.flags]


def test_plain_form_for_matching_s11_2() -> None:
    assert plain(" Turns,") == "turns"
    assert plain("Café") == "cafe"
    assert plain("guide's") == "guides"
    assert plain("3,050") == "3050"
    assert plain("—") == ""


def test_interval_distance_s11_2() -> None:
    assert interval_distance((1.4, 1.9), (1.62, 1.62)) == 0.0
    assert interval_distance((1.4, 1.9), (2.3, 2.3)) == pytest.approx(0.4)
    assert interval_distance((1.4, 1.9), (1.0, 1.2)) == pytest.approx(0.2)
    assert interval_distance((1.9, 1.4), (1.5, 1.6)) == 0.0


def test_cross_check_skips_words_with_no_plain_form_s11_2() -> None:
    boundary = Boundary(0, 1, 1.0, 1.2, (0, 0), (1, 0))
    spoken = [(0, 0, "…"), (1, 0, "…")]
    assert cross_check([boundary], spoken, asr(("…", 0.1, 0.9), ("…", 1.3, 1.5)), ()) == ()
