"""Pronunciation hints (design section 9.1 steps 3.1 and 5; section 9.3's second bullet).

Terms match case-sensitively, longest first, as whole words, per cue; only the term is replaced; each
application is recorded with its offset in the cue's spoken text; a term never applies across a cue.
"""

from __future__ import annotations

import pytest

from narration.contracts.errors import NarrationError
from narration.contracts.models import CueIn, Hint, HintApplied, SegmentIn, SegmentText
from narration.text import TextPipeline

OSSAVINE = Hint(term="Ossavine", respell="Oss-a-veen")
GASTRELLA = Hint(term="Gastrella", respell="Gas-trella")


def _plan(*cues: str, hints: tuple[Hint, ...]) -> SegmentText:
    segment = SegmentIn(segment_id="s", cues=tuple(CueIn(text=c) for c in cues))
    return TextPipeline().plan_request([segment], hints, strict_text=False)[0]


def test_hint_replaces_the_term_in_the_engine_text_only_s9_1() -> None:
    planned = _plan("Before dawn, the reef belongs to the Ossavine shrimp.", hints=(OSSAVINE,))
    (cue,) = planned.cues
    assert cue.spoken == "Before dawn, the reef belongs to the Ossavine shrimp."
    assert cue.engine == "Before dawn, the reef belongs to the Oss-a-veen shrimp."
    assert cue.hints_applied == (HintApplied(term="Ossavine", respell="Oss-a-veen", offset=37),)


def test_hint_matches_whole_words_only_s9_1() -> None:
    (cue,) = _plan("Gastrellas and Gastrellan waters hold one Gastrella.", hints=(GASTRELLA,)).cues
    assert cue.engine == "Gastrellas and Gastrellan waters hold one Gas-trella."
    assert [h.offset for h in cue.hints_applied] == [42]


def test_hint_matches_case_sensitively_s9_1() -> None:
    (cue,) = _plan("the gastrella and the GASTRELLA", hints=(GASTRELLA,)).cues
    assert cue.engine == cue.spoken
    assert cue.hints_applied == ()


def test_hint_longest_term_first_s9_1() -> None:
    hints = (Hint(term="Velmora", respell="Vell-mora"), Hint(term="Velmora Pass", respell="Vell-mora Pahss"))
    (cue,) = _plan("Velmora Pass is shut, but Velmora is open.", hints=hints).cues
    assert cue.engine == "Vell-mora Pahss is shut, but Vell-mora is open."
    assert [(h.term, h.offset) for h in cue.hints_applied] == [("Velmora Pass", 0), ("Velmora", 26)]


def test_hint_longest_first_does_not_depend_on_request_order_s9_1() -> None:
    long_first = (Hint(term="Ossavine shrimp", respell="Oss-a-veen shrimp"), Hint(term="the Ossavine", respell="X"))
    short_first = tuple(reversed(long_first))
    one = _plan("Here the Ossavine shrimp rest.", hints=long_first).cues[0]
    two = _plan("Here the Ossavine shrimp rest.", hints=short_first).cues[0]
    assert one.engine == two.engine == "Here the Oss-a-veen shrimp rest."
    assert one.hints_applied == two.hints_applied


def test_hint_possessive_matches_term_only_s9_1() -> None:
    (cue,) = _plan("Gastrella's fins are long.", hints=(GASTRELLA,)).cues
    assert cue.engine == "Gas-trella's fins are long."
    assert cue.hints_applied == (HintApplied(term="Gastrella", respell="Gas-trella", offset=0),)


def test_hint_possessive_with_curly_apostrophe_s9_1() -> None:
    (cue,) = _plan("Gastrella’s fins are long.", hints=(GASTRELLA,)).cues
    assert cue.engine == "Gas-trella’s fins are long."


def test_hint_bounded_by_hyphen_and_other_punctuation_s9_1() -> None:
    (cue,) = _plan("The Gastrella-side reef (Gastrella) and “Gastrella”.", hints=(GASTRELLA,)).cues
    assert cue.engine == "The Gas-trella-side reef (Gas-trella) and “Gas-trella”."
    assert [h.offset for h in cue.hints_applied] == [4, 25, 41]


def test_hint_not_bounded_by_a_combining_mark_or_digit_s9_1() -> None:
    # U+20DD COMBINING ENCLOSING CIRCLE has no precomposed form, so it stays after the "a" under NFC.
    (cue,) = _plan("Gastrella⃝ and Gastrella2", hints=(GASTRELLA,)).cues
    assert cue.hints_applied == ()


def test_hint_multiword_term_matches_after_canonical_form_s9_1() -> None:
    hints = (Hint(term="Velmora  Pass", respell="Vell-mora Pahss"),)
    (cue,) = _plan("We crossed Velmora   Pass at noon.", hints=hints).cues
    assert cue.engine == "We crossed Vell-mora Pahss at noon."
    assert cue.hints_applied == (HintApplied(term="Velmora Pass", respell="Vell-mora Pahss", offset=11),)


def test_hint_offsets_are_per_cue_in_the_spoken_text_s9_1() -> None:
    planned = _plan("Ossavine rowed out,", "and the Ossavine came back.", hints=(OSSAVINE,))
    assert [[h.offset for h in c.hints_applied] for c in planned.cues] == [[0], [8]]
    assert planned.engine_text == "Oss-a-veen rowed out, and the Oss-a-veen came back."
    assert [c.engine_span for c in planned.cues] == [(0, 21), (22, 51)]
    assert [c.spoken_span for c in planned.cues] == [(0, 19), (20, 47)]


def test_hint_never_across_a_cue_boundary_s9_1() -> None:
    hints = (Hint(term="Velmora Pass", respell="Vell-mora Pahss"),)
    planned = _plan("We climbed towards Velmora", "Pass before noon.", hints=hints)
    assert planned.engine_text == planned.spoken_text == "We climbed towards Velmora Pass before noon."
    assert all(c.hints_applied == () for c in planned.cues)
    (flag,) = planned.warnings
    assert (flag.code, flag.severity, flag.segment_id, flag.cue) == ("TERM_SPLIT_ACROSS_CUES", "warn", "s", 0)
    assert flag.details == {"term": "Velmora Pass", "cues": [0, 1], "offset": 19}


def test_hint_split_across_cues_is_flagged_even_when_it_also_applies_inside_one_s9_1() -> None:
    hints = (Hint(term="Velmora Pass", respell="Vell-mora Pahss"),)
    planned = _plan("Velmora Pass, then Velmora", "Pass again.", hints=hints)
    assert [len(c.hints_applied) for c in planned.cues] == [1, 0]
    assert [f.code for f in planned.warnings] == ["TERM_SPLIT_ACROSS_CUES"]


def test_hint_split_across_three_cues_names_them_s9_1() -> None:
    hints = (Hint(term="Wendle Cross Fell", respell="Wen-dle Cross Fell"),)
    (flag,) = _plan("We took the Wendle", "Cross", "Fell Road home.", hints=hints).warnings
    assert flag.details == {"term": "Wendle Cross Fell", "cues": [0, 2], "offset": 12}
    assert "cues 0 to 2" in flag.message


def test_hint_that_matches_nowhere_is_not_an_error_s9_1() -> None:
    planned = _plan("Nothing here.", hints=(OSSAVINE,))
    assert planned.cues[0].hints_applied == ()
    assert planned.warnings == ()


def test_hint_without_respelling_is_recorded_and_leaves_the_engine_text_s9_1() -> None:
    (cue,) = _plan("Ossavine rowed out.", hints=(Hint(term="Ossavine"),)).cues
    assert cue.engine == cue.spoken
    assert cue.hints_applied == (HintApplied(term="Ossavine", respell=None, offset=0),)


def test_hints_do_not_change_spoken_length_s7_2() -> None:
    planned = _plan("Ossavine rowed out.", hints=(OSSAVINE,))
    assert planned.spoken_chars == len("Ossavine rowed out.")
    assert len(planned.engine_text) == planned.spoken_chars + 2


def test_hint_duplicate_term_is_refused_s9_1() -> None:
    with pytest.raises(NarrationError) as caught:
        _plan("x", hints=(OSSAVINE, Hint(term="Ossavine ", respell="Oh-sa-vine")))
    assert (caught.value.code, caught.value.field) == ("INVALID_ARGUMENT", "hints[1].term")


def test_hint_empty_respelling_is_refused_s9_1() -> None:
    with pytest.raises(NarrationError) as caught:
        _plan("Ossavine rowed.", hints=(Hint(term="Ossavine", respell="  "),))
    assert (caught.value.code, caught.value.field) == ("INVALID_ARGUMENT", "hints[0].respell")


def test_hint_term_of_whitespace_only_is_refused_s9_1() -> None:
    with pytest.raises(NarrationError) as caught:
        _plan("x", hints=(Hint(term=" \t"),))
    assert (caught.value.code, caught.value.field) == ("INVALID_ARGUMENT", "hints[0].term")


def test_respelled_term_raises_no_text_warning_of_its_own_s9_1() -> None:
    # The engine is given the respelling, not the digits, so the term's digits are not a warning.
    (cue,) = _plan("Then R2-D2 rolled in.", hints=(Hint(term="R2-D2", respell="Artoo-Deetoo"),)).cues
    assert cue.warnings == ()
    (unhinted,) = _plan("Then R2-D2 rolled in.", hints=(Hint(term="R2-D2"),)).cues
    assert [w.details["token"] for w in unhinted.warnings if w.details] == ["2", "2"]


def test_a_respelling_is_checked_and_reported_at_its_term_s9_1() -> None:
    (cue,) = _plan("Meet Ossavine now.", hints=(Hint(term="Ossavine", respell="Oss 4 veen"),)).cues
    (warning,) = cue.warnings
    assert warning.details == {"token": "4", "kind": "digit", "offset": 5, "respell_of": "Ossavine"}
    assert "respelling of 'Ossavine'" in warning.message
