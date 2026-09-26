"""Property-style tests of the rule "text is spoken as sent" (design sections 7.2, 9.1; AGENTS.md section 7).

Random cues are drawn from a pool of awkward characters (every kind of whitespace canonical form collapses,
decomposed and composed letters, combining marks, digits, symbols, astral characters, format characters),
with a fixed seed so every run checks the same cases. For each, the planner may change only whitespace and
Unicode form; hints change only the engine text, and only at the places they record; warnings change
nothing; word indices survive canonical form.
"""

from __future__ import annotations

import random
import unicodedata

import pytest

from narration.contracts.models import CueIn, Hint, SegmentIn, SegmentText
from narration.text import TextPipeline, canonical_form, words
from narration.text.rules import WHITESPACE

SEED = 20260926
CASES = 300

_POOL = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") * 4
    + list(" " * 20)
    + list("\t\n\r   　  ")
    + list(".,;:!?'’‘\"“”()-–—…")
    + ["é", "é", "Å", "Å", "́", "⃝", "q̇", "ễ"]
    + list("0123456789²٣")
    + list("%&/×°·#£€$@*+=<>{}~^")
    + ["𐌷", "𐍃", "\U0001f600", "​", "­", "µ", "μ", "Ω", "ß", "ﬁ"]
)


def _random_text(rng: random.Random) -> str:
    while True:
        text = "".join(rng.choice(_POOL) for _ in range(rng.randint(1, 60)))
        if canonical_form(text):
            return text


def _strip_whitespace(text: str) -> str:
    return "".join(ch for ch in text if ch not in WHITESPACE)


def _cases() -> list[list[str]]:
    rng = random.Random(SEED)
    return [[_random_text(rng) for _ in range(rng.randint(1, 4))] for _ in range(CASES)]


def _plan(cues: list[str], hints: tuple[Hint, ...] = ()) -> SegmentText:
    segment = SegmentIn(segment_id="p", cues=tuple(CueIn(text=c) for c in cues))
    return TextPipeline().plan_request([segment], hints, strict_text=False)[0]


@pytest.mark.parametrize("cues", _cases()[:100])
def test_planner_changes_only_whitespace_and_unicode_form_s7_2(cues: list[str]) -> None:
    planned = _plan(cues)
    for cue, sent in zip(planned.cues, cues, strict=True):
        assert cue.received == sent
        assert _strip_whitespace(cue.spoken) == _strip_whitespace(unicodedata.normalize("NFC", sent))
        assert unicodedata.is_normalized("NFC", cue.spoken)
        assert cue.spoken == cue.spoken.strip(" ") and "  " not in cue.spoken
        assert not any(ch in WHITESPACE and ch != " " for ch in cue.spoken)
        assert cue.engine == cue.spoken, "without hints, and whatever the warnings, the engine text is the spoken text"
        assert planned.spoken_text[cue.spoken_span[0] : cue.spoken_span[1]] == cue.spoken
        assert len(words(sent)) == len(words(cue.spoken)), "word indices survive canonical form"
    assert planned.spoken_text == " ".join(c.spoken for c in planned.cues)
    assert planned.engine_text == planned.spoken_text
    assert planned.spoken_chars == len(planned.spoken_text)


@pytest.mark.parametrize("cues", _cases()[100:200])
def test_planning_the_spoken_text_again_changes_nothing_s7_2(cues: list[str]) -> None:
    once = _plan(cues)
    twice = _plan([c.spoken for c in once.cues])
    assert [c.spoken for c in twice.cues] == [c.spoken for c in once.cues]
    assert [c.warnings for c in twice.cues] == [c.warnings for c in once.cues]


@pytest.mark.parametrize("cues", _cases()[200:])
def test_warnings_locate_their_token_in_the_spoken_text_s9_1(cues: list[str]) -> None:
    for cue in _plan(cues).cues:
        for warning in cue.warnings:
            assert warning.details is not None
            token, offset = warning.details["token"], warning.details["offset"]
            assert cue.spoken[offset : offset + len(token)] == token


@pytest.mark.parametrize("seed", range(40))
def test_hints_change_only_the_engine_text_at_recorded_places_s9_1(seed: int) -> None:
    rng = random.Random(SEED + seed)
    names = ["Ossavine", "Gastrella", "Velmora Pass", "Velmora", "Tarn"]
    fillers = ["the", "reef", "and", "Tarnholm", "Velmoran", "’s", "'s", "-side", ",", ".", "(", ")", "—"]
    cues = [
        " ".join(rng.choice(names + fillers) for _ in range(rng.randint(1, 10)))
        .replace(" ’s", "’s")
        .replace(" 's", "'s")
        for _ in range(rng.randint(1, 3))
    ]
    hints = tuple(Hint(term=n, respell=None if rng.random() < 0.3 else n.upper() + "-X") for n in names)
    plain, hinted = _plan(cues), _plan(cues, hints)
    assert [c.spoken for c in hinted.cues] == [c.spoken for c in plain.cues], "hints never change the spoken text"
    assert hinted.spoken_chars == plain.spoken_chars
    respell = {h.term: h.respell for h in hints}
    for cue in hinted.cues:
        rebuilt, pos = [], 0
        for applied in cue.hints_applied:
            assert cue.spoken[applied.offset : applied.offset + len(applied.term)] == applied.term
            assert applied.respell == respell[applied.term]
            assert applied.offset >= pos, "applications never overlap"
            rebuilt.append(cue.spoken[pos : applied.offset])
            rebuilt.append(applied.respell if applied.respell is not None else applied.term)
            pos = applied.offset + len(applied.term)
        rebuilt.append(cue.spoken[pos:])
        assert "".join(rebuilt) == cue.engine
