"""Text match (design sections 11.1 steps 4–7 and 11.3): wer_raw, wer_adj and its word-count rule, exact spans,
terms and head/end insertions, on the tests' own text."""

from __future__ import annotations

import dataclasses

import pytest

from narration.contracts import codes
from narration.contracts.models import ExactWords, Hint, HintApplied, SegmentText
from narration.qa.normaliser import NumberReader
from narration.qa.profile import DEFAULT_PROFILE
from narration.qa.textmatch import TextMatch, align, find_spans, find_terms, match_text

from .builders import VOICE_TRANSCRIPT, Cue, planned, segment, words

READER = NumberReader()
VELMORANTH = Hint(term="Velmoranth")
QUILLASTRE = Hint(term="Quillastre", asr_aliases=("kill a star",))
TAVIMOCREL = Hint(term="Tavimocrel")
KURDELL = Hint(term="Kurdell vesatrimens")


def run(seg: SegmentText, transcript: str, hints: tuple[Hint, ...] = (), voice: str = VOICE_TRANSCRIPT) -> TextMatch:
    return match_text(seg, hints, transcript, voice, READER, DEFAULT_PROFILE)


def codes_of(match: TextMatch) -> list[tuple[str, str]]:
    return [(f.code, f.severity) for f in match.flags]


def severities(match: TextMatch, code: str) -> list[str]:
    return [f.severity for f in match.flags if f.code == code]


# ======================================================================== locating terms and spans


def test_terms_match_whole_words_case_sensitive_longest_first_s9_1() -> None:
    seg = segment("Tessarel met Tessarella's kin, and tessarella slept.", "Tessarella Minor woke.")
    hints = [Hint(term="Tessarella"), Hint(term="Tessarella Minor"), Hint(term="Tessarel")]
    found = [(t.cue, t.term, seg.spoken_text[t.start : t.end], t.possessive) for t in find_terms(seg, hints)]
    assert found == [
        (0, "Tessarel", "Tessarel", False),
        (0, "Tessarella", "Tessarella's", True),
        (1, "Tessarella Minor", "Tessarella Minor", False),
    ]


def test_terms_never_match_across_a_cue_boundary_s9_1() -> None:
    seg = segment("They called it Tessarella", "Minor at last.")
    assert find_terms(seg, [Hint(term="Tessarella Minor")]) == ()


def test_terms_come_from_the_pipelines_hints_applied_s9_1() -> None:
    # The pipeline records where it applied each hint; QA takes those places, and the canonical term.
    hint = Hint(term="Tessarella  Minor", respell="Tes-sarella Minor")
    seg = planned("They saw Tessarella Minor at dawn.", hints=[hint])
    assert [(a.term, a.offset) for a in seg.cues[0].hints_applied] == [("Tessarella Minor", 9)]
    (occ,) = find_terms(seg, [hint])
    assert (occ.term, seg.spoken_text[occ.start : occ.end]) == ("Tessarella Minor", "Tessarella Minor")
    # A cue with no record is searched for the canonical term, never the hint's raw spelling.
    (occ,) = find_terms(segment("They saw Tessarella Minor at dawn."), [hint])
    assert (occ.term, occ.start) == ("Tessarella Minor", 9)


def test_a_recorded_term_offset_that_does_not_hold_the_term_is_refused() -> None:
    seg = segment("They saw Tessarella at dawn.")
    wrong = dataclasses.replace(seg.cues[0], hints_applied=(HintApplied(term="Tessarella", respell=None, offset=3),))
    with pytest.raises(ValueError, match="hints_applied"):
        find_terms(dataclasses.replace(seg, cues=(wrong,)), [Hint(term="Tessarella")])


def test_exact_span_word_ranges_map_into_the_join_s11_3() -> None:
    seg = segment("A calm morning.", Cue("Then forty-eight boats came in.", exact=((1, 2),)))
    (span,) = find_spans(seg)
    assert seg.spoken_text[span.char_start : span.char_end] == "forty-eight"
    assert (span.cue, span.words, span.start, span.end) == (1, (1, 2), 5, 16)


def test_exact_span_words_skip_punctuation_only_tokens_s7_2() -> None:
    # narration.text.words: a lone dash is not a word, so word 1 is "two".
    (span,) = find_spans(segment(Cue("One — two, three.", exact=((1, 2),))))
    assert (span.start, span.end) == (6, 9)


def test_a_word_range_outside_its_cue_is_refused() -> None:
    good = segment(Cue("Two words.", exact=((1, 2),)))
    bad_cue = dataclasses.replace(good.cues[0], exact=(ExactWords(start=4, end=10, words=(1, 3)),))
    seg = dataclasses.replace(good, cues=(bad_cue,))
    with pytest.raises(ValueError, match="do not fit"):
        find_spans(seg)


# ======================================================================== WER


def test_a_perfect_transcript_scores_zero_s11_1() -> None:
    seg = segment("Before dawn the lamp was lit.", "The boat left at noon.")
    match = run(seg, "before dawn, the lamp was lit; the boat left at noon")
    assert (match.wer_raw, match.wer_adj, match.word_errors) == (0.0, 0.0, 0)
    assert match.flags == ()


def test_wer_raw_is_whisper_normalised_word_error_rate_s11_1() -> None:
    seg = segment("The guide's boat left at noon.")
    match = run(seg, "The guides boat left at noon.")
    # Whisper's normaliser reads "guide's" as "guide is": seven reference words, one substitution and one deletion.
    assert match.wer_raw == pytest.approx(2 / 7)


def test_wer_adj_removes_apostrophes_s11_1() -> None:
    seg = segment("The guide's boat left at noon.")
    for heard in ("The guides boat left at noon.", "The guide’s boat left at noon."):
        match = run(seg, heard)
        assert (match.wer_adj, match.flags) == (0.0, ()), heard


def test_wer_adj_reads_nought_as_zero_s11_1() -> None:
    match = run(segment("The pump lifts nought point seven two litres."), "The pump lifts zero point seven two litres.")
    assert match.wer_adj == 0.0


@pytest.mark.parametrize(
    ("spoken", "heard"),
    [
        ("In the year two thousand, forty ships sailed.", "In 2000, 40 ships sailed."),
        ("In the year two thousand, forty ships sailed.", "In the year two thousand, forty ships sailed."),
        ("By nineteen ninety, five hundred ships had gone.", "By 1990, 500 ships had gone."),
        ("She counted one, two, three and stopped.", "She counted one, two, three and stopped."),
    ],
)
def test_numbers_never_merge_across_punctuation_in_wer_adj_s11_3(spoken: str, heard: str) -> None:
    # Reader @2 (plan.md DC-7): "two thousand, forty" is 2000 and 40 on both sides, never 2040.
    match = run(segment(spoken), heard.replace("In 2000", "In the year 2000"))
    assert (match.wer_adj, match.flags) == (0.0, ())


@pytest.mark.parametrize(
    ("n_words", "errors", "expected"),
    [
        (51, 1, None),  # 0.0196: just below the warn rate
        (50, 1, None),  # 0.0200: at the warn rate, not above it
        (49, 1, "warn"),  # 0.0204: just above
        (10, 1, "warn"),  # 0.1000 is above the fail rate, but one error is under the fail count
        (34, 2, "warn"),  # 0.0588: just below the fail rate
        (50, 3, "warn"),  # 0.0600: at the fail rate, not above it
        (33, 2, "fail"),  # 0.0606: just above, with the two errors the fail needs
    ],
)
def test_wer_word_count_rule_edges_s11_1(n_words: int, errors: int, expected: str | None) -> None:
    text = words(n_words)
    heard = text.split()
    for i in range(errors):
        heard[2 * i + 1] = "stormy"
    match = run(segment(text), " ".join(heard))
    assert match.word_errors == errors
    assert match.wer_adj == pytest.approx(errors / n_words)
    assert severities(match, codes.WER_HIGH) == ([] if expected is None else [expected])


# ======================================================================== terms in wer_adj (section 11.1)


def test_wer_adj_collapses_a_found_term_to_one_token_s11_1() -> None:
    seg = segment("Before dawn the Velmoranth lamp was lit.")
    for heard in ("velmorant", "vel morant", "Velmoranth"):
        match = run(seg, f"Before dawn the {heard} lamp was lit.", (VELMORANTH,))
        assert match.wer_adj == 0.0, heard
        assert match.terms[0].ok, heard
        assert match.terms[0].heard == heard.lower()


def test_a_term_not_found_costs_exactly_one_substitution_s11_1() -> None:
    # "bottle" and "fell more" are ratios 0.25 and 0.56 against "velmoranth": not the term. The words heard in
    # its place collapse to one token that is not the term's: one substitution, however many words.
    seg = segment("Before dawn the Velmoranth lamp was lit on the quay.")
    for heard in ("bottle", "fell more"):
        match = run(seg, f"Before dawn the {heard} lamp was lit on the quay.", (VELMORANTH,))
        assert match.word_errors == 1, heard
        assert (match.terms[0].heard, match.terms[0].ok) == (heard, False)
        assert codes_of(match) == [(codes.WER_HIGH, "warn"), (codes.TERM_UNVERIFIED, "warn")]


def test_unrelated_words_around_a_term_not_found_are_errors_of_their_own_s11_1() -> None:
    # Only the best-matching words collapse; the others are insertions, so this take fails.
    seg = segment("Then a Velmoranth swam past the quay.")
    match = run(seg, "Then a big fat bottle swam past the quay.", (VELMORANTH,))
    assert match.word_errors == 3
    assert severities(match, codes.WER_HIGH) == ["fail"]
    assert match.terms[0].heard is not None and len(match.terms[0].heard.split()) == 1


def test_a_name_split_in_two_is_one_substitution_s11_1() -> None:
    # "Tavimocrel" heard as "tavvy mokrell" (0.73, not found) is one error, and the aligned words after it
    # are never reported as heard in its place.
    seg = segment("The Tavimocrel and a crab swam past the quay.")
    match = run(seg, "The tavvy mokrell and a crab swam past the quay.", (TAVIMOCREL,))
    assert (match.word_errors, match.terms[0].heard, match.terms[0].ok) == (1, "tavvy mokrell", False)


def test_a_term_not_found_never_takes_in_an_end_insertion_s11_1() -> None:
    seg = segment("The keeper called it Velmoranth.")
    match = run(seg, "The keeper called it bell moran, thank you all.", (VELMORANTH,))
    heard = match.terms[0].heard
    assert heard is not None and "thank" not in heard
    assert severities(match, codes.END_INSERTION) == ["fail"]
    tail = run(segment("They saw the Tavimocrel."), "They saw the tavvy mokrell and a crab.", (TAVIMOCREL,))
    assert tail.terms[0].heard is not None and "crab" not in tail.terms[0].heard
    assert severities(tail, codes.END_INSERTION) == ["fail"]


def test_a_term_not_found_never_takes_in_a_head_insertion_s11_1() -> None:
    seg = segment("Velmoranth rose early over the quay.")
    match = run(seg, "Well then, as always, bottle rose early over the quay.", (VELMORANTH,))
    assert match.terms[0].heard == "bottle"
    assert severities(match, codes.HEAD_INSERTION) == ["fail"]


OSSAVINE = Hint(term="Ossavine")
SHRIMP = segment(f"{words(14)} and the Ossavine shrimp swam past {words(12, stem='quiet')}")  # 32 words
DAWN = segment(f"{words(10)} at dawn Kurdell vesatrimens woke on the shore {words(10, stem='quiet')}")  # 28 words


@pytest.mark.parametrize(
    ("seg", "name", "heard", "hint", "errors", "severity"),
    [
        # Words the model or the ASR put inside a name: each costs one word error, and the name's pieces
        # around them stay one window (review F1, the lead's ruling A as amended).
        (SHRIMP, "Ossavine", "Oss of vine", OSSAVINE, 1, "warn"),
        (SHRIMP, "Ossavine", "Oss the vine", OSSAVINE, 1, "warn"),
        (DAWN, "Kurdell vesatrimens", "kurdell ves of trimens", KURDELL, 1, "warn"),
        (segment("Before dawn the Velmoranth lamp was lit."), "Velmoranth", "vel in moranth", VELMORANTH, 1, "warn"),
        (segment("Before dawn the Velmoranth lamp was lit."), "Velmoranth", "velmo went ranth", VELMORANTH, 1, "warn"),
        # An inserted run is never hidden: it costs its words.
        (
            segment("At dawn Kurdell vesatrimens woke on the shore."),
            "Kurdell vesatrimens",
            "kurdell is now vesatrimens",
            KURDELL,
            2,
            "fail",
        ),
        (
            segment("At dawn Kurdell vesatrimens woke on the shore."),
            "Kurdell vesatrimens",
            "kurdell the the vesatrimens",
            KURDELL,
            2,
            "fail",
        ),
    ],
)
def test_words_inside_a_name_cost_one_error_each_s11_1(
    seg: SegmentText, name: str, heard: str, hint: Hint, errors: int, severity: str
) -> None:
    # "of" (-0.11 for "Oss of vine"), "is" (-0.045), "now" (-0.069), "the" (-0.066), "in" (-0.091) and "went" (-0.167)
    # lower the window's match with the name, so they are not part of it; the pieces around them are.
    match = run(seg, seg.spoken_text.replace(name, heard), (hint,))
    assert (match.terms[0].heard, match.terms[0].ok) == (heard.lower(), True)
    assert match.word_errors == errors
    assert severities(match, codes.WER_HIGH) == [severity]


@pytest.mark.parametrize(
    ("seg", "heard", "hint"),
    [
        (
            segment("At dawn Kurdell vesatrimens woke on the shore."),
            "At dawn kurdell vesa trimens woke on the shore.",
            KURDELL,
        ),
        (segment("They named the reef Quillastre at last."), "They named the reef quill a stre at last.", QUILLASTRE),
    ],
)
def test_real_fragments_of_a_term_are_part_of_its_window_s11_1(seg: SegmentText, heard: str, hint: Hint) -> None:
    # "vesa" and "a" raise the match, so the name heard in pieces is the name.
    match = run(seg, heard, (hint,))
    assert (match.wer_adj, match.flags) == (0.0, ())
    assert match.terms[0].heard in ("kurdell vesa trimens", "quill a stre")


def test_a_term_not_found_takes_its_best_match_not_the_aligned_word_s11_1() -> None:
    # The review's case: in a 43-word segment, "Tavimocrel" heard as "tavvy mokrell too" is one substitution
    # plus one insertion ("too"): 2 errors, a warning, and "too" is never offered as what was heard.
    seg = segment(f"{words(20)} and the Tavimocrel swam by {words(18, stem='quiet')}")
    heard = seg.spoken_text.replace("Tavimocrel", "tavvy mokrell too")
    match = run(seg, heard, (TAVIMOCREL,))
    assert (match.ref_words, match.word_errors) == (43, 2)
    assert severities(match, codes.WER_HIGH) == ["warn"]
    assert (match.terms[0].heard, match.terms[0].ok) == ("tavvy mokrell", False)
    (flag,) = [f for f in match.flags if f.code == codes.TERM_UNVERIFIED]
    assert "'too'" not in flag.message


def test_an_edge_word_joins_a_window_that_already_matches_s11_1() -> None:
    # Lead ruling: "bell" raises "moranth" (already a match, 0.82) by +0.034, so "bell moranth" is the name,
    # and as the first words of the segment it is not a head insertion.
    match = run(
        segment("Velmoranth rose early over the quay."), "bell moranth rose early over the quay.", (VELMORANTH,)
    )
    assert (match.terms[0].heard, match.terms[0].ok, match.head_count, match.wer_adj) == ("bell moranth", True, 0, 0.0)
    # "the" (-0.13) and "at" (-0.087) lower the match, so they stay outside the window.
    head = run(segment("Velmoranth rose early."), "the velmoranth rose early.", (VELMORANTH,))
    assert (head.terms[0].heard, head.head_count) == ("velmoranth", 1)
    tail = run(segment("The keeper called it Velmoranth."), "The keeper called it moranth at.", (VELMORANTH,))
    assert (tail.terms[0].heard, tail.terms[0].ok, tail.end_count) == ("moranth", True, 1)


def test_no_edge_word_can_lift_a_window_to_a_match_s11_1() -> None:
    # "bell moran" is 0.737; "thank" would lift it to 0.75 with a gain of only 0.013.
    match = run(
        segment("The keeper called it Velmoranth."), "The keeper called it bell moran thank you all.", (VELMORANTH,)
    )
    assert (match.terms[0].heard, match.terms[0].ok) == ("bell moran", False)
    assert severities(match, codes.END_INSERTION) == ["fail"]


def test_a_term_with_nothing_heard_costs_one_deletion_s11_1() -> None:
    seg = segment("Before dawn the Velmoranth lamp was lit on the quay.")
    match = run(seg, "Before dawn the lamp was lit on the quay.", (VELMORANTH,))
    assert match.word_errors == 1  # the term's token, deleted
    assert match.wer_adj == pytest.approx(1 / 10)
    assert (match.terms[0].heard, match.terms[0].ok) == (None, False)


def test_wer_adj_maps_asr_aliases_s11_1() -> None:
    seg = segment("They named the reef Quillastre at last.")
    without = run(seg, "They named the reef kill a star at last.", (Hint(term="Quillastre"),))
    assert not without.terms[0].ok  # letters-only ratio 0.74, just under 0.75
    with_alias = run(seg, "They named the reef kill a star at last.", (QUILLASTRE,))
    assert with_alias.terms[0].ok
    assert with_alias.terms[0].heard == "kill a star"
    assert with_alias.wer_adj == 0.0


def test_an_alias_counts_only_where_its_term_was_to_be_spoken_s11_1() -> None:
    # "bran" is Brannoc's alias, and also an ordinary word of this cue: only the term's place reads it as
    # Brannoc. Without the alias, "bran" would not match the term (0.73).
    brannoc = Hint(term="Brannoc", asr_aliases=("bran",))
    seg = segment(Cue("Brannoc took the bran into the sea.", exact=((3, 4),)))
    match = run(seg, "Brannoc took the bran into the sea.", (brannoc,))
    assert (match.wer_adj, match.flags) == (0.0, ())
    assert [(r.expected, r.heard, r.match) for r in match.exact] == [("bran", "bran", "same")]
    heard_as_alias = run(seg, "Bran took the bran into the sea.", (brannoc,))
    assert (heard_as_alias.wer_adj, heard_as_alias.terms[0].ok) == (0.0, True)


def test_a_term_is_read_before_it_is_compared_s11_1() -> None:
    seg = segment("They sailed past the Seven Sisters at dawn.")
    match = run(seg, "They sailed past the 7 sisters at dawn.", (Hint(term="Seven Sisters"),))
    assert (match.terms[0].ok, match.terms[0].heard, match.wer_adj) == (True, "7 sisters", 0.0)


# ======================================================================== exact spans (section 11.3)

SUNRISE = segment(Cue("By sunrise, some three thousand two hundred of them are back.", exact=((3, 7),)))


@pytest.mark.parametrize(
    "heard",
    [
        "some 3,200 of them",
        "some 3200 of them",
        "some thirty-two hundred of them",
        "some three thousand two hundred of them",
    ],
)
def test_exact_span_same_number_passes_s11_3(heard: str) -> None:
    match = run(SUNRISE, f"By sunrise, {heard} are back.")
    (result,) = match.exact
    assert (result.cue, result.words, result.start, result.end) == (0, (3, 7), 17, 43)
    assert (result.expected, result.heard, result.match) == ("3200", "3200", "same")
    assert match.exact_ok
    assert not severities(match, codes.EXACT_SPAN_MISMATCH)


def test_exact_span_different_number_fails_s11_3() -> None:
    match = run(SUNRISE, "By sunrise, some 3000 of them are back.")
    (result,) = match.exact
    assert (result.expected, result.heard, result.match) == ("3200", "3000", "different")
    assert not match.exact_ok
    (flag,) = [f for f in match.flags if f.code == codes.EXACT_SPAN_MISMATCH]
    assert (flag.severity, flag.retake_trigger, flag.cue, flag.segment_id) == ("fail", True, 0, None)
    assert flag.details == {"cue": 0, "words": [3, 7], "expected": "3200", "heard": "3000", "match": "different"}


def test_exact_span_missing_fails_s11_3() -> None:
    match = run(SUNRISE, "By sunrise, some of them are back.")
    (result,) = match.exact
    assert (result.heard, result.match) == (None, "missing")
    assert severities(match, codes.EXACT_SPAN_MISMATCH) == ["fail"]


def test_exact_span_fails_whatever_wer_adj_says_s11_1() -> None:
    # One wrong number in a long paragraph is under the WER warn rate, but the span still fails.
    text = words(60) + " and then twenty-four boats came in."
    n = len(text.split())
    seg = segment(Cue(text, exact=((n - 4, n - 3),)))
    match = run(seg, text.replace("twenty-four", "twenty-five"))
    assert match.wer_adj is not None and match.wer_adj < DEFAULT_PROFILE.wer_warn_above
    assert [f.code for f in match.flags] == [codes.EXACT_SPAN_MISMATCH]


def test_exact_span_nought_heard_as_zero_passes_s11_3() -> None:
    seg = segment(Cue("The pump lifts nought point seven two litres.", exact=((3, 7),)))
    for heard in ("zero point seven two", "0.72", "nought point seven two"):
        (result,) = run(seg, f"The pump lifts {heard} litres.").exact
        assert (result.expected, result.heard, result.match) == ("0.72", "0.72", "same"), heard


@pytest.mark.parametrize(
    ("text", "span", "heard"),
    [
        ("In the year two thousand, forty ships sailed.", (5, 6), "In the year two thousand, forty ships sailed."),
        ("In the year two thousand, forty ships sailed.", (5, 6), "In the year 2000, 40 ships sailed."),
        ("She counted one, two, three and stopped.", (3, 4), "She counted one, two, three and stopped."),
        ("By nineteen ninety, five hundred ships had gone.", (3, 5), "By 1990, 500 ships had gone."),
    ],
)
def test_numbers_never_merge_across_punctuation_in_exact_spans_s11_3(
    text: str, span: tuple[int, int], heard: str
) -> None:
    # Reader @1 read "two thousand, forty" as 2040, failing the span "forty" on a perfect take (a retake loop).
    match = run(segment(Cue(text, exact=(span,))), heard)
    assert [r.match for r in match.exact] == ["same"]
    assert match.flags == ()


def test_exact_span_reads_its_words_with_their_punctuation_s11_3() -> None:
    # "38%" is one word whose core is "38"; the span is read whole, so the per cent is part of the number.
    seg = segment(Cue("The tide rose to 38% of the harbour wall.", exact=((4, 5),)))
    for heard in ("38% of the harbour wall", "thirty-eight per cent of the harbour wall"):
        (result,) = run(seg, f"The tide rose to {heard}.").exact
        assert (result.expected, result.heard, result.match) == ("38%", "38%", "same"), heard
        assert (result.start, result.end) == (17, 19)  # the word's core in the cue
    assert run(seg, "The tide rose to 38 of the harbour wall.").exact[0].match == "different"


@pytest.mark.parametrize(
    ("text", "span", "heard"),
    [
        ("The ferry left at four thirty sharp.", (4, 6), "The ferry left at 4:30 sharp."),
        ("The ferry left at ten thirty sharp.", (4, 6), "The ferry left at 10:30 sharp."),
        ("A pint cost three pounds fifty that year.", (3, 6), "A pint cost £3.50 that year."),
        ("The water fell to minus five degrees.", (4, 6), "The water fell to -5 degrees."),
        ("The lake was five degrees that morning.", (3, 5), "The lake was 5° that morning."),
        ("It was five degrees celsius at noon.", (2, 5), "It was 5°C at noon."),
    ],
)
def test_written_times_money_and_minus_read_as_spoken_s11_3(text: str, span: tuple[int, int], heard: str) -> None:
    # Reader @2 (lead ruling on the re-review): a perfect take never fails on how the ASR wrote these.
    match = run(segment(Cue(text, exact=(span,))), heard)
    assert [r.match for r in match.exact] == ["same"]
    assert (match.wer_adj, match.flags) == (0.0, ())


def test_exact_span_apostrophes_are_set_aside_s11_3() -> None:
    # Curly and straight apostrophes read alike, and "Brannoc's" and "Brannocs" cannot be told apart by ear.
    seg = segment(Cue("They found Brannoc’s boat on the shore.", exact=((2, 3),)))
    for heard in ("They found Brannoc's boat on the shore.", "They found Brannocs boat on the shore."):
        match = run(seg, heard)
        assert [r.match for r in match.exact] == ["same"], heard
        assert match.flags == (), heard


def test_exact_span_includes_words_inserted_inside_it_s11_3() -> None:
    seg = segment(Cue("The flags were red and blue that day.", exact=((3, 6),)))
    (result,) = run(seg, "The flags were red and green blue that day.").exact
    assert (result.expected, result.heard, result.match) == ("red and blue", "red and green blue", "different")


def test_exact_span_non_number_words_compare_word_for_word_s11_3() -> None:
    seg = segment(Cue("The flags were red and blue that day.", exact=((3, 6),)))
    assert run(seg, "The flags were red and blue that day.").exact[0].match == "same"
    assert run(seg, "The flags were red and black that day.").exact[0].match == "different"


def test_a_term_inside_an_exact_span_matches_only_by_spelling_or_alias_s11_3() -> None:
    seg = segment(Cue("They named the reef Quillastre at last.", exact=((4, 5),)))
    fuzzy = run(seg, "They named the reef Quilastre at last.", (QUILLASTRE,))
    assert fuzzy.terms[0].ok  # the term check is fuzzy ...
    assert fuzzy.exact[0].match == "different"  # ... the exact span is not
    alias = run(seg, "They named the reef kill a star at last.", (QUILLASTRE,))
    assert alias.exact[0].match == "same"


def test_exact_spans_in_several_cues_s11_3() -> None:
    seg = segment(Cue("Nine boats left.", exact=((0, 1),)), Cue("Eleven came back.", exact=((0, 1),)))
    match = run(seg, "Nine boats left. Twelve came back.")
    assert [(r.cue, r.expected, r.heard, r.match) for r in match.exact] == [
        (0, "9", "9", "same"),
        (1, "11", "12", "different"),
    ]


# ======================================================================== terms (section 11.1 step 6)


def test_term_possessive_is_set_aside_s11_1() -> None:
    seg = segment("They named Quillastre's garden.")
    for heard in ("Quillastre's", "quillastres", "kill a stars"):
        match = run(seg, f"They named {heard} garden.", (QUILLASTRE,))
        assert match.terms[0].ok, heard


def test_every_occurrence_of_a_term_is_reported_s11_1() -> None:
    seg = segment("Velmoranth rose early.", "Later Velmoranth slept.")
    match = run(seg, "Velmoranth rose early. Later bottle slept.", (VELMORANTH,))
    assert [(t.cue, t.ok) for t in match.terms] == [(0, True), (1, False)]
    (flag,) = [f for f in match.flags if f.code == codes.TERM_UNVERIFIED]
    assert (flag.severity, flag.cue, flag.retake_trigger) == ("warn", 1, False)


def test_a_missing_term_is_unverified_with_nothing_heard_s11_1() -> None:
    match = run(segment("Before dawn the Velmoranth lamp was lit."), "Before dawn the lamp was lit.", (VELMORANTH,))
    assert (match.terms[0].heard, match.terms[0].ok) == (None, False)
    assert codes_of(match).count((codes.TERM_UNVERIFIED, "warn")) == 1


def test_term_ratio_edge_s11_1() -> None:
    # "quillastre" against "killastar": 7 letters in common of 10 + 9, a ratio of 14/19 = 0.737 < 0.75;
    # against "quilastar": 8 in common of 10 + 9, 16/19 = 0.842 >= 0.75.
    seg = segment("They named the reef Quillastre at last.")
    assert not run(seg, "They named the reef killastar at last.", (Hint(term="Quillastre"),)).terms[0].ok
    assert run(seg, "They named the reef quilastar at last.", (Hint(term="Quillastre"),)).terms[0].ok


# ======================================================================== insertions (section 11.1 step 7)

LAMP = segment("Before dawn the lamp was lit.")


@pytest.mark.parametrize(
    ("head", "expected"),
    [("", None), ("yes", "warn"), ("yes indeed", "warn"), ("yes indeed sir", "fail")],
)
def test_head_insertion_edges_s11_1(head: str, expected: str | None) -> None:
    match = run(LAMP, f"{head} before dawn the lamp was lit".strip())
    assert match.head_count == len(head.split())
    assert severities(match, codes.HEAD_INSERTION) == ([] if expected is None else [expected])


@pytest.mark.parametrize("bleed", ["the boats", "for the boats", "wall for the boats"])
def test_head_insertion_matching_the_voice_transcript_fails_s11_1(bleed: str) -> None:
    match = run(LAMP, f"{bleed}. Before dawn the lamp was lit.")
    assert match.bleed
    (flag,) = [f for f in match.flags if f.code == codes.HEAD_INSERTION]
    assert (flag.severity, flag.retake_trigger, flag.cue) == ("fail", True, 0)
    assert flag.details is not None and flag.details["bleed"] is True


def test_head_insertion_of_the_transcripts_opening_is_bleed_s11_1() -> None:
    assert run(LAMP, "Morning comes before dawn the lamp was lit").bleed


def test_a_head_word_from_inside_the_transcript_is_not_bleed_s11_1() -> None:
    # Bleed is the transcript's end (or start) carried into the take; "stone wall" is only in its middle.
    match = run(LAMP, "stone wall before dawn the lamp was lit")
    assert not match.bleed
    assert severities(match, codes.HEAD_INSERTION) == ["warn"]


@pytest.mark.parametrize(
    ("head", "voice", "bleed"),
    [
        ("boats", VOICE_TRANSCRIPT, False),  # one word of 5 letters: too short to tell from chance
        ("oats", VOICE_TRANSCRIPT, False),  # a single 4-letter word matching the end is never bleed
        ("the boats", VOICE_TRANSCRIPT, True),  # two words
        ("lighthouse", "The keepers climbed up to the lighthouse.", True),  # one word of 10 letters
        ("moorings", "The boats rest at their moorings", True),  # one word of 8 letters: at the minimum
        ("mooring", "The boat rests at its mooring", False),  # 7 letters: under it
    ],
)
def test_bleed_needs_two_words_or_eight_letters_s11_1(head: str, voice: str, bleed: bool) -> None:
    match = run(LAMP, f"{head} before dawn the lamp was lit", voice=voice)
    assert match.bleed is bleed
    assert severities(match, codes.HEAD_INSERTION) == ["fail" if bleed else "warn"]


@pytest.mark.parametrize(
    ("tail", "expected"),
    [("", None), ("thanks", "warn"), ("thank you", "warn"), ("thank you all", "fail")],
)
def test_end_insertion_edges_s11_1(tail: str, expected: str | None) -> None:
    match = run(LAMP, f"before dawn the lamp was lit {tail}".strip())
    assert match.end_count == len(tail.split())
    found = [(f.severity, f.cue) for f in match.flags if f.code == codes.END_INSERTION]
    assert found == ([] if expected is None else [(expected, 0)])


def test_a_hallucinated_tail_after_a_term_is_still_an_end_insertion_s11_1() -> None:
    seg = segment("The keeper called it Velmoranth")
    match = run(seg, "The keeper called it velmoranth thank you", (VELMORANTH,))
    assert match.terms[0].ok
    assert match.terms[0].heard == "velmoranth"
    assert match.end_words == ("thank", "you")


def test_a_split_term_at_the_end_is_not_an_insertion_s11_1() -> None:
    seg = segment("The keeper called it Velmoranth")
    match = run(seg, "The keeper called it vel moranth", (VELMORANTH,))
    assert (match.end_count, match.wer_adj, match.terms[0].heard) == (0, 0.0, "vel moranth")


def test_text_flags_carry_no_segment_id_s11_1() -> None:
    match = run(SUNRISE, "Well, by sunrise, some 3000 of them are back, thank you all.")
    assert match.flags and {f.segment_id for f in match.flags} == {None}


# ======================================================================== alignment helper


def test_align_handles_empty_sides() -> None:
    assert align([], []) == ()
    assert [c.kind for c in align(["a"], [])] == ["delete"]
    assert [c.kind for c in align([], ["a"])] == ["insert"]
