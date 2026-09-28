"""The scorer (design section 11.1): one take's verdict from that take and the request's inputs for it."""

from __future__ import annotations

import itertools

import pytest

from narration.config import MeasurementConfig
from narration.contracts import codes
from narration.contracts.interfaces import QaInputs, QaScorer
from narration.contracts.models import CueTiming, Flag, Hint, SimilarityBaseline
from narration.contracts.names import NUMBER_READER, QA_PROFILE
from narration.qa import QaUnavailable, Scorer
from narration.qa.checks import expected_wpm, spoken_words

from .builders import ANCHOR, Cue, alignment, inputs, measurement, pace, planned, segment, signal, with_similarity

SEG = segment(
    "Before dawn the Velmoranth lamp was lit on the old quay.",
    Cue("By noon some three thousand two hundred gulls had come back to the harbour.", exact=((3, 7),)),
)
HINTS = (Hint(term="Velmoranth"),)
MEASURED = measurement(anchor_p5=0.975, pace_record=pace(((80, 150.0), (300, 170.0)), tol=0.12))
CLEAN = inputs(
    SEG,
    hints=HINTS,
    embedding=with_similarity(0.985),
    measurement_record=MEASURED,
    signal_stats=signal(voiced=(0.1, 9.9)),
)


def test_scorer_is_a_qa_scorer_with_the_pinned_versions() -> None:
    scorer = Scorer()
    assert isinstance(scorer, QaScorer)
    assert scorer.profile_version == QA_PROFILE
    assert scorer.number_reader == NUMBER_READER


def test_a_clean_take_passes_with_every_metric_s11_1() -> None:
    result = Scorer().score(CLEAN)
    assert result.verdict == "pass"
    assert result.flags == ()
    assert result.transcript == SEG.spoken_text
    m = result.metrics
    assert (m.wer_raw, m.wer_adj, m.word_errors, m.exact_ok) == (0.0, 0.0, 0, True)
    assert m.spk_sim_anchor == pytest.approx(0.985)
    assert m.spoken_wpm == pytest.approx(25 / 9.8 * 60)  # 11 + 14 spoken words over 9.8 s voiced
    assert m.expected_spoken_wpm is not None
    assert (m.head_insertion_words, m.end_insertion_words, m.longest_silence_s) == (0, 0, 0.5)
    assert result.thresholds.spk_warn == 0.965
    assert result.thresholds.spk_fail == 0.90
    assert result.thresholds.pace_tol == 0.12
    assert [(e.expected, e.match) for e in result.exact] == [("3200", "same")]
    assert [(t.term, t.ok) for t in result.terms] == [("Velmoranth", True)]


def test_a_take_half_again_as_fast_as_its_curve_warns_and_is_no_retake_dc19() -> None:
    """default.v4 (the owner's decision, 2026-09-27): a take at +50% of the expected pace, with every other check
    clean, gets a PACE_FAST warning and a warn verdict. It is not failed and not retaken."""
    expected = expected_wpm(MEASURED.pace, SEG.spoken_chars)
    assert expected is not None
    span = spoken_words(SEG.spoken_text) / (1.5 * expected) * 60.0
    take = inputs(
        SEG,
        hints=HINTS,
        embedding=with_similarity(0.985),
        measurement_record=MEASURED,
        signal_stats=signal(voiced=(0.1, 0.1 + span)),
    )
    result = Scorer().score(take)
    assert result.metrics.spoken_wpm == pytest.approx(1.5 * expected)
    assert [(f.code, f.severity, f.retake_trigger) for f in result.flags] == [(codes.PACE_FAST, "warn", False)]
    assert result.verdict == "warn"
    assert not any(codes.is_retake_trigger(f.code, f.severity, f.details) for f in result.flags)


def test_a_slow_take_still_only_warns_s11_1() -> None:
    expected = expected_wpm(MEASURED.pace, SEG.spoken_chars)
    assert expected is not None
    span = spoken_words(SEG.spoken_text) / (0.5 * expected) * 60.0
    take = inputs(
        SEG,
        hints=HINTS,
        embedding=with_similarity(0.985),
        measurement_record=MEASURED,
        signal_stats=signal(voiced=(0.1, 0.1 + span)),
    )
    result = Scorer().score(take)
    assert [(f.code, f.severity, f.retake_trigger) for f in result.flags] == [(codes.PACE_SLOW, "warn", False)]
    assert result.verdict == "warn"


def test_every_flag_carries_its_segment_and_retake_rule_s11_1() -> None:
    bad = inputs(
        SEG,
        "Before dawn the lamp was lit on the old quay. By noon some 3000 gulls had come back to the harbour.",
        hints=HINTS,
        hit_token_cap=True,
    )
    result = Scorer().score(bad)
    assert result.verdict == "fail"
    assert {f.segment_id for f in result.flags} == {None}  # the job assembler stamps the request's
    for flag in result.flags:
        assert flag.retake_trigger == codes.is_retake_trigger(flag.code, flag.severity, flag.details)
    assert {f.code for f in result.flags} >= {codes.TOKEN_CAP_HIT, codes.EXACT_SPAN_MISMATCH, codes.TERM_UNVERIFIED}


def test_a_ladder_take_is_judged_on_the_calibration_anchor_without_pace_s11_1() -> None:
    baseline = SimilarityBaseline(anchor_p5=0.975, anchor_p50=0.98, consistency_p5=0.98)
    take = inputs(SEG, hints=HINTS, embedding=with_similarity(0.95), anchor=ANCHOR, similarity=baseline)
    result = Scorer().score(take)
    assert [(f.code, f.severity) for f in result.flags] == [(codes.SPK_SIM_LOW, "warn")]
    assert (result.thresholds.spk_warn, result.thresholds.pace_tol) == (0.965, None)
    assert result.metrics.expected_spoken_wpm is None


def test_an_audition_has_no_speaker_or_pace_check_s11_1() -> None:
    result = Scorer().score(inputs(SEG, hints=HINTS, embedding=with_similarity(0.5)))
    assert result.verdict == "pass"
    assert result.metrics.spk_sim_anchor is None
    assert (result.thresholds.spk_warn, result.thresholds.pace_tol) == (None, None)


def test_the_measurement_wins_over_partial_voice_facts_s11_1() -> None:
    low = SimilarityBaseline(anchor_p5=0.5, anchor_p50=0.6, consistency_p5=0.5)
    take = inputs(SEG, hints=HINTS, embedding=with_similarity(0.95), measurement_record=MEASURED, similarity=low)
    assert Scorer().score(take).thresholds.spk_warn == 0.965


def test_an_unplaced_cue_warns_and_triggers_a_retake_s11_2() -> None:
    cues = (
        CueTiming(index=0, start_s=0.1, end_s=4.0, confidence=0.9),
        CueTiming(index=1, start_s=None, end_s=None, confidence=None),
    )
    result = Scorer().score(inputs(SEG, hints=HINTS, align=alignment(SEG, cues=cues)))
    assert result.verdict == "warn"
    (flag,) = result.flags
    assert (flag.code, flag.cue, flag.retake_trigger) == (codes.CUE_UNALIGNED, 1, True)


def test_an_alignment_error_fails_the_take_s11_2() -> None:
    error = Flag(code=codes.ALIGNMENT_ERROR, severity="fail", message="frames < tokens + repeats")
    unplaced = tuple(CueTiming(index=c.index, start_s=None, end_s=None, confidence=None) for c in SEG.cues)
    result = Scorer().score(inputs(SEG, hints=HINTS, align=alignment(SEG, cues=unplaced, flags=[error])))
    assert result.verdict == "fail"
    assert [f.code for f in result.flags] == [codes.ALIGNMENT_ERROR, codes.CUE_UNALIGNED, codes.CUE_UNALIGNED]


# ======================================================================== planted non-faults (WP40's list)


def test_per_cent_heard_as_percent_passes_s11_3() -> None:
    seg = segment(Cue("About fourteen per cent of the boats stayed out.", exact=((1, 4),)))
    result = Scorer().score(inputs(seg, "About 14 percent of the boats stayed out."))
    assert result.verdict == "pass"
    assert [(e.expected, e.heard) for e in result.exact] == [("14%", "14%")]


def test_nought_heard_as_zero_passes_s11_3() -> None:
    seg = segment(Cue("The pump lifts nought point seven two litres a second.", exact=((3, 7),)))
    result = Scorer().score(inputs(seg, "The pump lifts zero point seven two litres a second."))
    assert result.verdict == "pass"


def test_one_slip_in_a_three_word_title_warns_but_does_not_fail_s11_1() -> None:
    result = Scorer().score(inputs(segment("The Quiet Harbour"), "The Quite Harbour"))
    assert result.metrics.wer_adj == pytest.approx(1 / 3)
    assert [(f.code, f.severity) for f in result.flags] == [(codes.WER_HIGH, "warn")]
    assert result.verdict == "warn"


# ======================================================================== independence (section 11.1, 10.2)

OTHERS = [
    inputs(SEG, "", hints=HINTS, hit_token_cap=True),
    inputs(
        SEG,
        "hello there " + SEG.spoken_text + " thank you all",
        hints=HINTS,
        embedding=with_similarity(0.7),
        measurement_record=MEASURED,
        signal_stats=signal(silence_s=3.0, clipping=0.01),
    ),
    inputs(segment("A different paragraph entirely."), "A different paragraph entirely."),
]


def test_a_takes_verdict_never_depends_on_another_take_s11_1() -> None:
    alone = Scorer().score(CLEAN)
    for order in itertools.permutations([CLEAN, *OTHERS]):
        scorer = Scorer()
        results = {id(i): scorer.score(i) for i in order}
        assert results[id(CLEAN)] == alone
    # Nor on what is done with the takes afterwards: suggestions and the consistency report.
    scorer = Scorer()
    scorer.consistency([("tk_a", with_similarity(0.99)), ("tk_b", with_similarity(0.5))], MEASURED)
    assert scorer.score(CLEAN) == alone


def test_a_hint_not_used_in_the_segment_changes_nothing_s11_1() -> None:
    # QaInputs.hints are the hints used in the segment, the set the analysis key covers. A hint whose term is
    # not in the segment must not map its aliases, in wer_adj or in an exact span, or it would change a
    # cached verdict that its key does not cover.
    seg = segment(
        "The children cried kill a star as the Velmoranth lamp was lit.",
        Cue("Then twelve boats came in.", exact=((1, 2),)),
    )
    unrelated = (
        Hint(term="Quillastre", asr_aliases=("kill a star",)),
        Hint(term="Tessarella", asr_aliases=("twelve",)),
    )
    alone = Scorer().score(inputs(seg, hints=HINTS))
    assert alone.verdict == "pass"
    assert Scorer().score(inputs(seg, hints=(*HINTS, *unrelated))) == alone
    assert Scorer().score(inputs(seg, hints=(*unrelated, *HINTS))) == alone


def test_a_result_holds_nothing_the_analysis_key_does_not_cover_s10_2() -> None:
    # Two requests that differ only in the segment id and in how the cue was sent (so its exact span's
    # offsets differ, but not its words) have the same analysis key, so they must get the same result.
    one = planned(("By noon some three thousand gulls came back.", [(13, 27)]), hints=HINTS, segment_id="p01")
    two = planned(("By noon  some three thousand gulls came back.", [(14, 28)]), hints=HINTS, segment_id="dusk-07")
    assert one.cues[0].exact[0].start != two.cues[0].exact[0].start
    assert one.cues[0].exact[0].words == two.cues[0].exact[0].words
    heard = "By noon some 3000 gulls came back, thank you."  # a fail, so there are flags to compare
    first = Scorer().score(inputs(one, heard, hints=HINTS))
    assert first.verdict == "fail" and first.exact
    assert Scorer().score(inputs(two, heard, hints=HINTS)) == first
    assert {f.segment_id for f in first.flags} == {None}
    (exact,) = first.exact
    assert (exact.words, exact.start, exact.end) == ((3, 5), 13, 27)  # offsets in the cue's spoken text


def test_an_anchor_without_an_embedding_raises_qa_unavailable_s11_1() -> None:
    with pytest.raises(QaUnavailable):
        Scorer().score(inputs(SEG, hints=HINTS, embedding=None, measurement_record=MEASURED))
    ladder = inputs(SEG, hints=HINTS, anchor=ANCHOR, similarity=MEASURED.similarity)
    with pytest.raises(QaUnavailable):
        Scorer().score(ladder)


def test_the_same_inputs_give_the_same_result_s10_2() -> None:
    assert Scorer().score(CLEAN) == Scorer().score(CLEAN)


def test_scorer_settings_are_fixed_at_construction_s16() -> None:
    lenient = Scorer(MeasurementConfig(sim_fail_floor=0.5, sim_warn_margin=0.2))
    take = inputs(SEG, hints=HINTS, embedding=with_similarity(0.85), measurement_record=MEASURED)
    assert Scorer().score(take).verdict == "fail"
    assert lenient.score(take).verdict == "pass"
    assert isinstance(OTHERS[0], QaInputs)
