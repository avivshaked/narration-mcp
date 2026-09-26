"""The aligner's transcript and its guard (design section 11.2 steps 1 and 3), on the service's own texts
(``material/alignment/alignment-en.v1``) and short made-up ones."""

from __future__ import annotations

import pytest

from narration.align import CtcAligner, build_transcript, ctc_frames, guard, repeats, wildcard_runs
from narration.contracts.interfaces import AlignTranscript
from narration.contracts.models import Hint
from narration.text import words

from ._support import REVISION, segment


def _spelled_words(transcript: AlignTranscript) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for token, owner in zip(transcript.tokens, transcript.token_words, strict=True):
        if owner is not None:
            out[owner] = out.get(owner, "") + token
    return out


def test_token_map_uses_the_text_pipelines_word_indices_s11_2() -> None:
    seg = segment("The morning ferry to Quenby Reach carries", "about forty passengers and a few cars.")
    t = build_transcript(seg, ())
    expected = [(cue.index, w.index, w.text) for cue in seg.cues for w in words(cue.spoken)]
    assert list(t.words) == expected
    assert _spelled_words(t) == {(c, w): text.upper().rstrip(".") for c, w, text in expected}
    assert t.dropped == ()


def test_separator_only_between_words_and_across_cues_s11_2() -> None:
    t = build_transcript(segment("Then it turns", "for home."), ())
    assert "".join(t.tokens) == "THEN|IT|TURNS|FOR|HOME"
    assert all(owner is None for token, owner in zip(t.tokens, t.token_words, strict=True) if token == "|")
    assert t.tokens[0] != "|" and t.tokens[-1] != "|"


def test_word_outside_the_alphabet_becomes_a_wildcard_s11_2_dc11() -> None:
    t = build_transcript(segment("We counted 12 herons and 3,050 gulls."), ())
    assert "".join(t.tokens) == "WE|COUNTED|*|HERONS|AND|*|GULLS"
    assert t.dropped == ((0, 2, "12"), (0, 5, "3,050"))
    assert (0, 2, "12") in t.words  # still a word of the cue, with null times later
    assert [t.token_words[i] for i, token in enumerate(t.tokens) if token == "*"] == [(0, 2), (0, 5)]
    assert wildcard_runs(t) == {11: ((0, 2),), 24: ((0, 5),)}


def test_a_run_of_left_out_words_is_one_wildcard_s11_2_dc11() -> None:
    t = build_transcript(segment("It ran 12 7 & 365 days."), ())
    assert "".join(t.tokens) == "IT|RAN|*|DAYS"
    [(index, run)] = wildcard_runs(t).items()
    assert t.tokens[index] == "*" and run == tuple((0, w) for w in range(2, len(t.words) - 1))


def test_a_wildcard_run_never_crosses_a_cue_s11_2_dc11() -> None:
    t = build_transcript(segment("We saw 12", "13 more."), ())
    assert "".join(t.tokens) == "WE|SAW|*|*|MORE"
    assert list(wildcard_runs(t).values()) == [((0, 2),), ((1, 0),)]


def test_a_cue_of_digits_is_one_wildcard_s11_2_dc11() -> None:
    t = build_transcript(segment("Rain.", "12, 13.", "Sun."), ())
    assert "".join(t.tokens) == "RAIN|*|SUN"
    assert list(wildcard_runs(t).values()) == [((1, 0), (1, 1))]


def test_punctuation_only_token_is_not_a_word_s11_2() -> None:
    t = build_transcript(segment("Rain — then sun."), ())
    assert [text for _, _, text in t.words] == ["Rain", "then", "sun"]
    assert "".join(t.tokens) == "RAIN|THEN|SUN"


def test_cue_without_words_still_counted_s11_2() -> None:
    t = build_transcript(segment("Rain.", "—", "Then sun."), ())
    assert isinstance(t, AlignTranscript)
    assert t.cue_count == 3
    assert {cue for cue, _, _ in t.words} == {0, 2}


def test_hinted_term_uses_spoken_letters_never_the_respelling_s11_2() -> None:
    hint = Hint(term="Ossavine", respell="Oss-a-veen")
    seg = segment("Before dawn, the reef belongs to the Ossavine shrimp.", hints=[hint])
    assert seg.cues[0].engine.endswith("Oss-a-veen shrimp.")
    assert "".join(build_transcript(seg, [hint]).tokens).endswith("|OSSAVINE|SHRIMP")


def test_align_as_replaces_the_terms_letters_s11_2() -> None:
    hint = Hint(term="Ossavine", respell="Oss-a-veen", align_as="ossaveen")
    seg = segment("Before dawn, the reef belongs to the Ossavine shrimp.", hints=[hint])
    t = build_transcript(seg, [hint])
    assert _spelled_words(t)[(0, 7)] == "OSSAVEEN"
    assert t.words[7] == (0, 7, "Ossavine")  # the word keeps its spoken text


def test_align_as_keeps_the_rest_of_a_possessive_word_s11_2() -> None:
    hint = Hint(term="Brindlewick", align_as="BRINDLWIK")
    t = build_transcript(segment("The Brindlewick's shell.", hints=[hint]), [hint])
    assert _spelled_words(t)[(0, 1)] == "BRINDLWIK'S"


def test_align_as_after_a_hyphen_keeps_the_prefix_s11_2() -> None:
    hint = Hint(term="Brindlewick", align_as="BRINDLWIK")
    t = build_transcript(segment("A non-Brindlewick path.", hints=[hint]), [hint])
    assert "".join(t.tokens) == "A|NON|BRINDLWIK|PATH"
    assert t.token_words[t.tokens.index("N")] == (0, 1)


def test_align_as_parts_go_to_the_terms_words_in_order_s11_2() -> None:
    two = Hint(term="Maren Tollis", align_as="Marren Tolis")
    t = build_transcript(segment("Maren Tollis walks.", hints=[two]), [two])
    assert _spelled_words(t) == {(0, 0): "MARREN", (0, 1): "TOLIS", (0, 2): "WALKS"}

    extra = Hint(term="Ossavine", align_as="oss a veen")
    t = build_transcript(segment("An Ossavine shrimp.", hints=[extra]), [extra])
    assert "".join(t.tokens) == "AN|OSS|A|VEEN|SHRIMP"
    assert {t.token_words[i] for i in range(3, 11) if t.tokens[i] != "|"} == {(0, 1)}

    fewer = Hint(term="Maren Tollis", align_as="Marentollis")
    t = build_transcript(segment("Maren Tollis walks.", hints=[fewer]), [fewer])
    assert _spelled_words(t) == {(0, 0): "MARENTOLLIS", (0, 2): "WALKS"}
    assert t.dropped == ((0, 1, "Tollis"),)
    assert "*" not in t.tokens  # MARENTOLLIS already spells Tollis's speech: no wildcard (DC-11)


def test_align_as_only_for_the_hints_used_s11_2() -> None:
    hint = Hint(term="Quenby Reach", align_as="KWENBY REECH")
    seg = segment("The ferry to Quenby Reach.", hints=[hint])
    assert "".join(build_transcript(seg, []).tokens) == "THE|FERRY|TO|QUENBY|REACH"
    assert "".join(build_transcript(seg, [hint]).tokens) == "THE|FERRY|TO|KWENBY|REECH"


def test_transcript_does_not_depend_on_the_order_of_the_hints_s10_2() -> None:
    # Hints are a set: the analysis key sorts them, so the transcript it covers must not see their order.
    hints = [
        Hint(term="Maren", align_as="MARREN"),
        Hint(term="Maren Tollis", align_as="Marren Tolis"),
        Hint(term="Quenby", respell="Kwen-bee"),
        Hint(term="Ossavine", align_as="oss a veen"),
    ]
    cues = ("Maren Tollis walks to Quenby.", "Maren waits by the Ossavine.")
    first = build_transcript(segment(*cues, hints=hints), hints)
    for order in (hints[::-1], [hints[2], hints[0], hints[3], hints[1]]):
        assert build_transcript(segment(*cues, hints=order), order) == first
    assert "".join(first.tokens) == "MARREN|TOLIS|WALKS|TO|QUENBY|MARREN|WAITS|BY|THE|OSS|A|VEEN"


def test_one_term_with_two_align_as_is_refused_s11_2() -> None:
    hints = [Hint(term="Ossavine", align_as="ossaveen"), Hint(term="Ossavine", align_as="ossavine")]
    seg = segment("An Ossavine shrimp.", hints=hints[:1])
    with pytest.raises(ValueError, match="more than one align_as"):
        build_transcript(seg, hints)
    same = [Hint(term="Ossavine", align_as="ossaveen"), Hint(term="Ossavine", align_as="ossaveen")]
    assert "".join(build_transcript(seg, same).tokens) == "AN|OSSAVEEN|SHRIMP"


def test_term_words_listed_for_the_cross_check_s11_2() -> None:
    hint = Hint(term="Maren Tollis")
    t = build_transcript(segment("Here, Maren Tollis walks.", "Maren waits.", hints=[hint]), [hint])
    assert t.term_words == ((0, 1), (0, 2))


def test_repeats_count_equal_neighbours_s11_2() -> None:
    assert repeats(tuple("BELL")) == 1
    assert repeats(tuple("SEE|EELS")) == 2
    assert repeats(()) == 0


def test_guard_needs_frames_for_tokens_plus_repeats_s11_2() -> None:
    t = build_transcript(segment("A bell."), ())  # A | B E L L: 6 tokens, 1 repeat
    assert (len(t.tokens), repeats(t.tokens)) == (6, 1)
    assert guard(t, 7)
    assert not guard(t, 6)
    assert not guard(build_transcript(segment("12."), ()), 1000)  # a wildcard alone: no letter to align


def test_ctc_frames_matches_the_worker_s11_2() -> None:
    # KNOW from spike (b): two bake-off paragraphs (n01 and n08 of one take), 474240 and 157440 samples
    # at 24 kHz, gave 987 and 327 frames from the model.
    assert ctc_frames(474240, 24000) == 987
    assert ctc_frames(157440, 24000) == 327
    assert ctc_frames(48000, 48000) == 49
    assert ctc_frames(1200, 48000) == 1  # 400 samples at 16 kHz: one window
    assert ctc_frames(1197, 48000) == 0
    with pytest.raises(ValueError):
        ctc_frames(1, 0)


def test_too_short_audio_raises_the_guard_not_a_crash_s11_2() -> None:
    aligner = CtcAligner(revision=REVISION)
    t = aligner.build_transcript(segment("Winds came over the ridge.", "The gulls stayed low."), ())
    frames = ctc_frames(4800, 48000)  # 0.1 s
    assert not aligner.guard(t, frames)
    details = aligner.guard_details(t, frames)
    assert details == {"reason": "too_short", "frames": 4, "tokens": len(t.tokens), "repeats": repeats(t.tokens)}
