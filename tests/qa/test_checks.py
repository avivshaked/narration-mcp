"""Signal, speaker, pace and cue-alignment checks (design section 11.1 steps 1, 3, 8, 9): every default.v5 edge,
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
    pace_check,
    pace_flags,
    signal_flags,
    speaker_check,
    speaker_flags,
    spoken_words,
)
from narration.qa.errors import QaUnavailable
from narration.qa.pace import MIN_PAUSE_S, expected_cps, pause_seconds
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
# Pace is spoken characters per second of speaking time: the voiced span less every pause inside it (WP47,
# DC-18; ``narration.qa.pace``). Every number and sentence here is invented.


def test_expected_pace_follows_the_curve_and_holds_its_ends_flat_s11_1_dc20() -> None:
    record = pace(((80, 16.0), (300, 17.1)), per_100=0.5)
    assert expected_cps(record, 80) == 16.0
    assert expected_cps(record, 190) == pytest.approx(16.55)
    assert expected_cps(record, 300) == 17.1
    # Outside the curve: the end value, held flat. The trend's slope is never extended (DC-20).
    assert expected_cps(record, 40) == 16.0
    assert expected_cps(record, 400) == 17.1
    # With no curve (no rung passed): the voice's level. The trend is not read.
    assert expected_cps(pace((), level=15.0, intercept=10.0, per_100=1.0), 200) == 15.0


# The review of WP47's first version (DC-20): a curve that falls a little from its first point to its last, and
# a trend that falls with it. Extended beyond the curve, that slope would expect 14.5 at 600 characters.
FALLING = pace(((80, 16.4), (300, 16.0)), level=16.2, intercept=16.8, per_100=-0.5)


def test_a_take_at_the_bands_rate_beyond_the_curves_end_gets_no_pace_flag_dc20() -> None:
    long = segment(*["The ferry left the quay at dawn and came back at noon."] * 11)
    assert long.spoken_chars >= 600
    take = signal(duration_s=40.0, voiced=(0.1, 0.1 + long.spoken_chars / 16.2), silences=())
    check = pace_check(long, take, FALLING, CONFIG, P)
    assert check.expected_cps == 16.0  # the curve's last value, held flat
    assert check.articulation_cps == pytest.approx(16.2)
    assert check.flags == ()
    # the extended slope would have expected 14.5 there, and flagged this take fast
    assert [f.code for f in pace_flags(16.2, 14.5, TOL, P)] == [codes.PACE_FAST]


def test_a_take_shorter_than_the_curves_first_point_expects_that_points_value_dc20() -> None:
    short = segment("Gulls rose over the quay.")
    assert short.spoken_chars < 80
    assert expected_cps(FALLING, short.spoken_chars) == 16.4
    assert expected_cps(FALLING, 30) == 16.4
    take = signal(voiced=(0.1, 0.1 + short.spoken_chars / 16.4), silences=())
    check = pace_check(short, take, FALLING, CONFIG, P)
    assert (check.expected_cps, check.flags) == (16.4, ())


EXPECTED, TOL = 16.5, 0.10
FAST_WARN = EXPECTED * (1 + TOL)  # warn above curve x (1 + tol)
V3_FAST_FAIL = EXPECTED * (1 + 2 * TOL)  # default.v3 failed above curve x (1 + 2 tol); v4 and v5 never fail
SLOW_WARN = EXPECTED * (1 - TOL)  # warn below curve x (1 - tol); no profile has a slow fail


@pytest.mark.parametrize(
    ("cps", "expected"),
    [
        (FAST_WARN - 1e-6, []),
        (FAST_WARN, []),
        (FAST_WARN + 1e-6, [(codes.PACE_FAST, "warn")]),
        (V3_FAST_FAIL - 1e-6, [(codes.PACE_FAST, "warn")]),
        (V3_FAST_FAIL, [(codes.PACE_FAST, "warn")]),
        (V3_FAST_FAIL + 1e-6, [(codes.PACE_FAST, "warn")]),
        (EXPECTED * 1.5, [(codes.PACE_FAST, "warn")]),
        (EXPECTED * 4, [(codes.PACE_FAST, "warn")]),
        (SLOW_WARN + 1e-6, []),
        (SLOW_WARN, []),
        (SLOW_WARN - 1e-6, [(codes.PACE_SLOW, "warn")]),
        (4.0, [(codes.PACE_SLOW, "warn")]),
    ],
)
def test_pace_edges_s11_1(cps: float, expected: list[tuple[str, str]]) -> None:
    flags = pace_flags(cps, EXPECTED, TOL, P)
    assert [(f.code, f.severity) for f in flags] == expected
    assert all(f.retake_trigger is False for f in flags)  # a pace warning is never a retake trigger


def test_pace_fast_warns_and_never_fails_in_default_v5_dc19() -> None:
    """The owner's decision (2026-09-27): PACE_FAST warns only, so it never triggers a retake. default.v5 keeps
    it."""
    assert (P.name, P.pace_fail_tol_factor) == ("default.v5", None)
    (flag,) = pace_flags(EXPECTED * 1.5, EXPECTED, TOL, P, pause_s=0.4)
    assert (flag.code, flag.severity, flag.retake_trigger) == (codes.PACE_FAST, "warn", False)
    assert not codes.is_retake_trigger(flag.code, flag.severity, flag.details)
    assert flag.details is not None
    assert flag.details["articulation_cps"] == round(EXPECTED * 1.5, 3)
    assert flag.details["expected_articulation_cps"] == EXPECTED
    assert flag.details["fast_warn_above"] == round(FAST_WARN, 3)
    assert flag.details["fast_fail_above"] is None  # there is no fail line
    assert flag.details["slow_warn_below"] == round(SLOW_WARN, 3)
    assert (flag.details["pause_s"], flag.details["basis"]) == (0.4, "speaking_time")
    assert "fail" not in flag.message


def test_a_profile_with_a_pace_fail_factor_fails_above_it_dc19() -> None:
    """The factor is still honoured when a profile sets one (default.v3 had 2)."""
    v3 = dataclasses.replace(P, name="default.v3", pace_fail_tol_factor=2.0)
    assert [f.severity for f in pace_flags(V3_FAST_FAIL, EXPECTED, TOL, v3)] == ["warn"]
    (flag,) = pace_flags(V3_FAST_FAIL + 1e-6, EXPECTED, TOL, v3)
    assert (flag.code, flag.severity, flag.retake_trigger) == (codes.PACE_FAST, "fail", True)
    assert flag.details is not None and flag.details["fast_fail_above"] == round(V3_FAST_FAIL, 3)
    assert [f.severity for f in pace_flags(4.0, EXPECTED, TOL, v3)] == ["warn"]  # still no slow fail


def test_pace_is_spoken_characters_per_second_of_speaking_time_s11_1() -> None:
    """The pauses (silences of at least MIN_PAUSE_S) come out of the voiced span; shorter silences, such as a
    stop's closure, are part of speaking."""
    seg = segment("Twenty four boats — all of them — came in.")
    assert spoken_words(seg.spoken_text) == 8
    check = pace_check(seg, signal(voiced=(1.0, 5.0), silences=(0.1, 0.5, 0.24)), pace(), CONFIG, P)
    assert check.pause_s == pytest.approx(0.5)
    assert check.articulation_cps == pytest.approx(seg.spoken_chars / 3.5)
    assert check.spoken_cps == pytest.approx(seg.spoken_chars / 4)  # over the whole voiced span, information
    assert check.spoken_wpm == pytest.approx(8 / 4 * 60)


def test_a_pause_is_a_silence_of_at_least_min_pause_s_s11_1() -> None:
    assert pause_seconds(signal(silences=(MIN_PAUSE_S,))) == MIN_PAUSE_S
    assert pause_seconds(signal(silences=(0.24, 0.02))) == 0.0  # 12 frames of 20 ms: still speaking
    assert pause_seconds(signal(silences=(0.26, 0.9, 0.1))) == pytest.approx(1.16)
    assert pause_seconds(signal(silences=())) == 0.0
    assert pause_seconds(signal(silences=None)) is None  # not measured


def test_the_same_speaking_rate_with_or_without_pauses_between_sentences_is_one_pace_s11_1() -> None:
    """A one-sentence segment has no pause between sentences; a paragraph of three has two. Spoken at the same
    rate, both read as the same pace, so the one-sentence segment is not taken for fast (the owner's first job:
    single sentences flagged against a curve measured on paragraphs)."""
    rate = 16.5
    flat = pace(((30, rate), (300, rate)), per_100=0.0)
    one = segment("The ferry left the quay at dawn.")
    one_signal = signal(voiced=(0.1, 0.1 + one.spoken_chars / rate), silences=(0.06,))  # a closure, no pause
    three = segment(
        "The ferry left the quay at dawn.", "Gulls followed it out past the light.", "By noon it was a speck."
    )
    pauses = (0.55, 0.62)
    three_voiced = three.spoken_chars / rate + sum(pauses)
    three_signal = signal(voiced=(0.1, 0.1 + three_voiced), silences=(0.08, pauses[0], 0.12, pauses[1]))

    one_check = pace_check(one, one_signal, flat, CONFIG, P)
    three_check = pace_check(three, three_signal, flat, CONFIG, P)
    assert one_check.articulation_cps == pytest.approx(rate)
    assert three_check.articulation_cps == pytest.approx(rate)
    assert one_check.flags == () and three_check.flags == ()
    # Over the whole voiced span (the rule before WP47) the one sentence reads fast against the paragraph.
    assert one_check.spoken_cps is not None and three_check.spoken_cps is not None
    assert [f.code for f in pace_flags(one_check.spoken_cps, three_check.spoken_cps, TOL, P)] == [codes.PACE_FAST]


def test_a_voice_with_a_flat_rate_passes_short_words_that_read_fast_in_words_per_minute_s11_1() -> None:
    """Characters per second, not words per minute: a segment of short words has many more words a minute at the
    same rate, and passes (D2)."""
    rate = 16.0
    flat = pace(((80, rate), (300, rate)), per_100=0.0)
    short = segment("We sat by the red fire and ate figs.")
    long = segment("Extraordinary luminescence illuminated everything.")
    checks = [
        pace_check(s, signal(voiced=(0.1, 0.1 + s.spoken_chars / rate), silences=()), flat, CONFIG, P)
        for s in (short, long)
    ]
    assert [c.flags for c in checks] == [(), ()]
    assert checks[0].spoken_wpm is not None and checks[1].spoken_wpm is not None
    assert checks[0].spoken_wpm > 2.5 * checks[1].spoken_wpm  # 240 against 77 words a minute


def test_a_take_a_quarter_faster_warns_and_fails_only_under_a_fail_factor_s11_1_dc19() -> None:
    seg = segment("The ferry left the quay at dawn.", "Gulls followed it out past the light.")
    curve = pace(((30, EXPECTED), (300, EXPECTED)), per_100=0.0, tol=TOL)
    pause = 0.5
    fast = signal(voiced=(0.1, 0.1 + seg.spoken_chars / (1.25 * EXPECTED) + pause), silences=(0.1, pause))
    check = pace_check(seg, fast, curve, CONFIG, P)
    assert check.articulation_cps == pytest.approx(1.25 * EXPECTED)
    assert [(f.code, f.severity, f.retake_trigger) for f in check.flags] == [(codes.PACE_FAST, "warn", False)]
    v3 = dataclasses.replace(P, name="default.v3", pace_fail_tol_factor=2.0)
    assert [f.severity for f in pace_check(seg, fast, curve, CONFIG, v3).flags] == ["fail"]
    slow = signal(voiced=(0.1, 0.1 + seg.spoken_chars / (0.8 * EXPECTED) + pause), silences=(pause,))
    assert [(f.code, f.severity) for f in pace_check(seg, slow, curve, CONFIG, P).flags] == [(codes.PACE_SLOW, "warn")]


def test_without_measured_silences_the_pace_is_over_the_voiced_span_s11_1() -> None:
    seg = segment("Before dawn the lamp was lit.")
    check = pace_check(seg, signal(voiced=(1.0, 2.0), silences=None), pace(((30, 5.0), (300, 5.0))), CONFIG, P)
    assert check.pause_s is None
    assert check.articulation_cps == check.spoken_cps == pytest.approx(seg.spoken_chars / 1.0)
    (flag,) = check.flags
    assert flag.details is not None and (flag.details["pause_s"], flag.details["basis"]) == (None, "voiced_span")


def test_expected_wpm_is_the_takes_own_words_at_the_expected_pace_s11_1() -> None:
    """Information only: the words a minute this take would have if it spoke at the curve's pace, its pauses as
    they are."""
    seg = segment("Twenty four boats — all of them — came in.")
    record = pace(((30, 14.0), (300, 14.0)), per_100=0.0)
    check = pace_check(seg, signal(voiced=(1.0, 5.0), silences=(0.5,)), record, CONFIG, P)
    assert check.expected_cps == pytest.approx(14.0)
    assert check.expected_wpm == pytest.approx(8 / (seg.spoken_chars / 14.0 + 0.5) * 60)


def test_pace_tolerance_never_below_pace_tol_min_s3_2() -> None:
    seg = segment("Before dawn the lamp was lit.")
    assert pace_check(seg, signal(), pace(tol=0.05), CONFIG, P).tol == 0.10
    assert pace_check(seg, signal(), pace(tol=0.17), CONFIG, P).tol == 0.17


def test_no_voiced_span_or_no_curve_means_no_pace_check_s11_1() -> None:
    seg = segment("Before dawn the lamp was lit.")
    silent = pace_check(seg, signal(voiced=(None, None), silences=()), pace(), CONFIG, P)
    assert (silent.spoken_wpm, silent.articulation_cps, silent.pause_s, silent.flags) == (None, None, None, ())
    unmeasured = pace_check(seg, signal(), None, CONFIG, P)
    assert (unmeasured.expected_cps, unmeasured.expected_wpm, unmeasured.tol, unmeasured.flags) == (
        None,
        None,
        None,
        (),
    )


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


def test_qas_own_cue_unaligned_says_the_cue_was_not_placed_and_is_a_retake_trigger_s11_2_dc12() -> None:
    # Every CUE_UNALIGNED carries a details.reason (DC-12). QA's fallback cannot know why the aligner left a cue
    # without times, so it says only that: not_placed, which is still a retake trigger.
    seg = segment("One cue.", "Two cue.")
    cues = (
        CueTiming(index=0, start_s=0.1, end_s=1.0, confidence=0.9),
        CueTiming(index=1, start_s=None, end_s=None, confidence=None),
    )
    flags = alignment_flags(alignment(seg, cues=cues), seg)
    assert [(f.code, f.cue, f.retake_trigger, f.details) for f in flags] == [
        (codes.CUE_UNALIGNED, 1, True, {"reason": codes.CUE_NOT_PLACED})
    ]
    missing = alignment_flags(alignment(seg, cues=()), seg)
    assert [f.details for f in missing] == [{"reason": codes.CUE_NOT_PLACED}] * 2


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
    other = dataclasses.replace(nothing, details={"reason": codes.CUE_UNPLACED_LOW_CONFIDENCE}, retake_trigger=False)
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
