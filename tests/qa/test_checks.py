"""Signal, speaker, pace and cue-alignment checks (design section 11.1 steps 1, 3, 8, 9): every default.v3 edge,
just below, at and just above."""

from __future__ import annotations

import dataclasses

import pytest

from narration.config import MeasurementConfig
from narration.contracts import codes
from narration.contracts import errors as contract_errors
from narration.contracts.models import CueTiming, Flag, SimilarityBaseline
from narration.qa._flags import verdict_of
from narration.qa.checks import (
    alignment_flags,
    cosine,
    expected_wpm,
    pace_check,
    pace_flags,
    signal_flags,
    speaker_check,
    speaker_flags,
    spoken_words,
)
from narration.qa.errors import QaUnavailable
from narration.qa.profile import DEFAULT_PROFILE

from .builders import ANCHOR, alignment, pace, segment, signal, with_similarity

P = DEFAULT_PROFILE
CONFIG = MeasurementConfig()
EPS = 1e-9


def found(flags: tuple[Flag, ...], code: str) -> list[str]:
    return [f.severity for f in flags if f.code == code]


# ======================================================================== signal (step 1)


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(P.clipping_warn_above - EPS, []), (P.clipping_warn_above, []), (P.clipping_warn_above + EPS, ["warn"])],
)
def test_clipping_edge_s11_1(fraction: float, expected: list[str]) -> None:
    assert P.clipping_warn_above == 0.0001  # 0.01 % of samples at full scale
    assert found(signal_flags(signal(clipping=fraction), False, P), codes.CLIPPING) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (1.2 - EPS, []),
        (1.2, []),
        (1.2 + EPS, ["warn"]),
        (2.5 - EPS, ["warn"]),
        (2.5, ["warn"]),
        (2.5 + EPS, ["fail"]),
    ],
)
def test_longest_silence_edges_s11_1(seconds: float, expected: list[str]) -> None:
    assert found(signal_flags(signal(silence_s=seconds), False, P), codes.SILENCE_LONG) == expected


def test_token_cap_fails_and_is_a_retake_trigger_s11_1() -> None:
    (flag,) = signal_flags(signal(), True, P)
    assert (flag.code, flag.severity, flag.retake_trigger, flag.segment_id) == (codes.TOKEN_CAP_HIT, "fail", True, None)
    assert signal_flags(signal(), False, P) == ()


def test_non_finite_samples_fail_s11_1() -> None:
    (flag,) = signal_flags(signal(nonfinite=True), False, P)
    assert (flag.code, flag.severity, flag.retake_trigger) == (codes.SIGNAL_INVALID, "fail", True)


@pytest.mark.parametrize(
    ("dc", "expected"),
    [(0.001 - EPS, []), (0.001, []), (0.001 + EPS, ["warn"]), (-(0.001 + EPS), ["warn"])],
)
def test_dc_offset_edge_s11_1(dc: float, expected: list[str]) -> None:
    assert P.dc_warn_above == 0.001  # section 11.1's table since revision 5.4 (plan.md DC-10)
    assert found(signal_flags(signal(dc=dc), False, P), codes.SIGNAL_INVALID) == expected


# ======================================================================== speaker (step 8)


@pytest.mark.parametrize(
    ("similarity", "expected"),
    [
        (0.965 + EPS, []),  # just above the voice's warn threshold (anchor_p5 0.975 - 0.01)
        (0.965, []),  # at it
        (0.965 - EPS, ["warn"]),  # just below
        (0.90 + EPS, ["warn"]),  # just above the floor
        (0.90, ["warn"]),  # at the floor
        (0.90 - EPS, ["fail"]),  # just below the floor
    ],
)
def test_speaker_similarity_edges_s11_1(similarity: float, expected: list[str]) -> None:
    flags = speaker_flags(similarity, round(0.975 - CONFIG.sim_warn_margin, 6), CONFIG.sim_fail_floor)
    assert found(flags, codes.SPK_SIM_LOW) == expected
    assert all(f.retake_trigger == (f.severity == "fail") for f in flags)


def test_speaker_thresholds_come_from_the_voices_measurement_s11_1() -> None:
    # The probe's d4 read 0.966-0.969 against its clip: a fixed 0.975 would flag every take, a
    # d4-calibrated threshold flags none.
    baseline = SimilarityBaseline(anchor_p5=0.9663, anchor_p50=0.968, consistency_p5=0.98)
    check = speaker_check(with_similarity(0.9663), ANCHOR, baseline, CONFIG)
    assert check.warn_below == 0.9563
    assert check.fail_below == 0.90
    assert check.similarity == pytest.approx(0.9663)
    assert check.flags == ()


def test_speaker_without_a_baseline_checks_only_the_floor_s11_1() -> None:
    check = speaker_check(with_similarity(0.93), ANCHOR, None, CONFIG)
    assert (check.warn_below, check.flags) == (None, ())
    low = speaker_check(with_similarity(0.85), ANCHOR, None, CONFIG)
    assert found(low.flags, codes.SPK_SIM_LOW) == ["fail"]


def test_speaker_without_an_anchor_is_not_checked_s11_1() -> None:
    # An audition: no voice facts, nothing to compare with.
    check = speaker_check(with_similarity(0.5), None, None, CONFIG)
    assert (check.similarity, check.flags) == (None, ())
    assert speaker_check(None, None, None, CONFIG).flags == ()


def test_an_anchor_without_an_embedding_is_qa_unavailable_s11_1() -> None:
    # The speaker check is called for and cannot run: never a silent pass.
    baseline = SimilarityBaseline(anchor_p5=0.98, anchor_p50=0.99, consistency_p5=0.98)
    with pytest.raises(QaUnavailable, match="no speaker embedding") as caught:
        speaker_check(None, ANCHOR, baseline, CONFIG)
    assert caught.value.code == codes.QA_UNAVAILABLE
    assert QaUnavailable is contract_errors.QaUnavailable  # the contract's class, not a second one
    with pytest.raises(QaUnavailable, match="cannot be compared"):
        speaker_check((1.0, 0.0), ANCHOR, baseline, CONFIG)


def test_speaker_margins_come_from_config_s16() -> None:
    strict = MeasurementConfig(sim_warn_margin=0.0, sim_fail_floor=0.95)
    baseline = SimilarityBaseline(anchor_p5=0.97, anchor_p50=0.98, consistency_p5=0.98)
    check = speaker_check(with_similarity(0.94), ANCHOR, baseline, strict)
    assert (check.warn_below, check.fail_below, found(check.flags, codes.SPK_SIM_LOW)) == (0.97, 0.95, ["fail"])


def test_cosine_rejects_mismatched_or_zero_embeddings() -> None:
    assert cosine((1.0, 0.0), (2.0, 0.0)) == 1.0
    with pytest.raises(ValueError, match="shapes"):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="zero"):
        cosine((0.0, 0.0), (1.0, 0.0))


# ======================================================================== pace (step 9)


def test_expected_pace_follows_the_curve_at_this_length_s11_1() -> None:
    record = pace(((80, 120.0), (300, 150.0)), per_100=13.0)
    assert expected_wpm(record, 80) == 120.0
    assert expected_wpm(record, 190) == pytest.approx(135.0)
    assert expected_wpm(record, 300) == 150.0
    # Outside the curve: the end point moved along the trend's slope (13 wpm per 100 characters).
    assert expected_wpm(record, 40) == pytest.approx(120.0 - 5.2)
    assert expected_wpm(record, 400) == pytest.approx(163.0)
    assert expected_wpm(pace((), intercept=110.0, per_100=10.0), 200) == pytest.approx(130.0)


EXPECTED, TOL = 150.0, 0.10
FAST_WARN = EXPECTED * (1 + TOL)  # warn above curve x (1 + tol)
FAST_FAIL = EXPECTED * (1 + 2 * TOL)  # fail above curve x (1 + 2 tol)
SLOW_WARN = EXPECTED * (1 - TOL)  # warn below curve x (1 - tol); default.v3 has no slow fail


@pytest.mark.parametrize(
    ("wpm", "expected"),
    [
        (FAST_WARN - 1e-6, []),
        (FAST_WARN, []),
        (FAST_WARN + 1e-6, [(codes.PACE_FAST, "warn")]),
        (FAST_FAIL - 1e-6, [(codes.PACE_FAST, "warn")]),
        (FAST_FAIL, [(codes.PACE_FAST, "warn")]),
        (FAST_FAIL + 1e-6, [(codes.PACE_FAST, "fail")]),
        (SLOW_WARN + 1e-6, []),
        (SLOW_WARN, []),
        (SLOW_WARN - 1e-6, [(codes.PACE_SLOW, "warn")]),
        (40.0, [(codes.PACE_SLOW, "warn")]),
    ],
)
def test_pace_edges_s11_1(wpm: float, expected: list[tuple[str, str]]) -> None:
    flags = pace_flags(wpm, EXPECTED, TOL, P)
    assert [(f.code, f.severity) for f in flags] == expected
    assert all(f.retake_trigger == (f.severity == "fail") for f in flags)


def test_pace_counts_spoken_words_over_the_voiced_span_s11_1() -> None:
    seg = segment("Twenty four boats — all of them — came in.")
    assert spoken_words(seg.spoken_text) == 8
    check = pace_check(seg, signal(voiced=(1.0, 5.0)), pace(), CONFIG, P)
    assert check.spoken_wpm == pytest.approx(8 / 4 * 60)
    assert check.spoken_cps == pytest.approx(seg.spoken_chars / 4)


def test_pace_tolerance_never_below_pace_tol_min_s3_2() -> None:
    seg = segment("Before dawn the lamp was lit.")
    assert pace_check(seg, signal(), pace(tol=0.05), CONFIG, P).tol == 0.10
    assert pace_check(seg, signal(), pace(tol=0.17), CONFIG, P).tol == 0.17


def test_no_voiced_span_or_no_curve_means_no_pace_check_s11_1() -> None:
    seg = segment("Before dawn the lamp was lit.")
    silent = pace_check(seg, signal(voiced=(None, None)), pace(), CONFIG, P)
    assert (silent.spoken_wpm, silent.flags) == (None, ())
    unmeasured = pace_check(seg, signal(), None, CONFIG, P)
    assert (unmeasured.expected_wpm, unmeasured.tol, unmeasured.flags) == (None, None, ())


# ======================================================================== cue alignment (step 3)


def test_alignment_flags_are_carried_with_their_retake_rule_s11_2() -> None:
    seg = segment("One cue.", "Two cue.")
    low = Flag(code=codes.CUE_LOW_CONFIDENCE, severity="warn", message="low", cue=1)
    flags = alignment_flags(alignment(seg, flags=[low]), seg)
    assert [(f.code, f.segment_id, f.retake_trigger) for f in flags] == [(codes.CUE_LOW_CONFIDENCE, None, False)]
    # A segment id an aligner put on its flag is dropped too: QA results carry none (QaScorer.score).
    stamped = Flag(code=codes.CUE_LOW_CONFIDENCE, severity="warn", message="low", cue=1, segment_id="p01")
    assert alignment_flags(alignment(seg, flags=[stamped]), seg)[0].segment_id is None


def test_an_unplaced_cue_always_gets_cue_unaligned_s11_2() -> None:
    seg = segment("One cue.", "Two cue.")
    cues = (
        CueTiming(index=0, start_s=0.1, end_s=1.0, confidence=0.9),
        CueTiming(index=1, start_s=None, end_s=None, confidence=None),
    )
    flags = alignment_flags(alignment(seg, cues=cues), seg)
    assert [(f.code, f.severity, f.cue, f.retake_trigger) for f in flags] == [(codes.CUE_UNALIGNED, "warn", 1, True)]
    assert verdict_of(flags) == "warn"
    # Not added twice when the aligner already said so.
    said = Flag(code=codes.CUE_UNALIGNED, severity="warn", message="unplaced", cue=1)
    assert len(alignment_flags(alignment(seg, cues=cues, flags=[said]), seg)) == 1


def test_a_cue_with_no_alignable_words_is_not_a_retake_trigger_s11_1() -> None:
    # DC-12: the cue's text gives the aligner nothing to place, so every retake would fail the same way.
    seg = segment("One cue.", "Two cue.")
    cues = (
        CueTiming(index=0, start_s=0.1, end_s=1.0, confidence=0.9),
        CueTiming(index=1, start_s=None, end_s=None, confidence=None),
    )
    reason = {"reason": codes.CUE_NO_ALIGNABLE_WORDS}
    nothing = Flag(code=codes.CUE_UNALIGNED, severity="warn", message="unplaced", cue=1, details=reason)
    flags = alignment_flags(alignment(seg, cues=cues, flags=[nothing]), seg)
    assert [(f.code, f.cue, f.retake_trigger, f.details) for f in flags] == [(codes.CUE_UNALIGNED, 1, False, reason)]
    # Any other reason, or none, is still a trigger, whatever the aligner set.
    other = dataclasses.replace(nothing, details={"reason": "low_confidence"}, retake_trigger=False)
    assert alignment_flags(alignment(seg, cues=cues, flags=[other]), seg)[0].retake_trigger is True


def test_a_cue_missing_from_the_alignment_is_unplaced_s11_2() -> None:
    seg = segment("One cue.", "Two cue.")
    flags = alignment_flags(alignment(seg, cues=()), seg)
    assert [(f.code, f.cue) for f in flags] == [(codes.CUE_UNALIGNED, 0), (codes.CUE_UNALIGNED, 1)]


def test_verdict_is_the_worst_severity_and_info_never_counts_s11_1() -> None:
    info = Flag(code=codes.CUE_BOUNDARY_NO_PAUSE, severity="info", message="i")
    warn = Flag(code=codes.TERM_UNVERIFIED, severity="warn", message="w")
    fail = Flag(code=codes.WER_HIGH, severity="fail", message="f")
    assert verdict_of([]) == verdict_of([info]) == "pass"
    assert verdict_of([info, warn]) == "warn"
    assert verdict_of([warn, fail, info]) == "fail"
