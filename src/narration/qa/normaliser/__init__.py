"""The number reader of design section 11.3: Whisper's English text normaliser, read phrase by phrase.

``NumberReader.read`` is ``names.NUMBER_READER``, version 2 (plan.md DC-7):

1. The curly apostrophes ’ ‘ and the modifier letter ʼ are read as the straight one, so "Brannoc’s" and
   "Brannoc's" read alike.
2. Bracketed annotations ("[music]", "<noise>") are removed, as Whisper's normaliser removes them. The text
   pipeline refuses square brackets in spoken text, so only a transcript can hold one.
3. "nought" is read as "zero" (Whisper's normaliser alone misses it).
4. Written forms are made to read as their spoken forms do:
   - a clock time, a digit, a colon and two digits, is joined ("4:30" reads 430, as "four thirty" does);
   - a hyphen-minus (or the minus sign) at the start of a word, directly before a digit, is read as
     "minus" ("-5" reads -5, as "minus five" does, instead of losing its sign);
   - the degree sign, which the normaliser drops, is read as a word: "°C" and "℃" as "degrees celsius",
     "°F" and "℉" as "degrees fahrenheit", any other "°" as "degrees" ("5°C" reads as "five degrees
     celsius" does).
5. The text is split into **phrases at punctuation**, and each phrase goes through Whisper's English
   normaliser on its own: lower case, punctuation and symbols removed except those numbers need,
   contractions expanded, number words turned into digits, British spellings mapped to American ones.
6. In what the normaliser returns, an amount of money with pence or cents is split at its period ("£3.50"
   reads £3 50, as "three pounds fifty" does). This step comes after the normaliser, not before it, because
   the normaliser itself writes "three dollars and fifty cents" as $3.50.

Steps 4 to 6 are what version 2 adds. Whisper's normaliser deletes commas before it reads number words, so on
its own it reads "two thousand, forty" as 2040 and "one, two, three" as one number. Read in phrases, number
words never merge across punctuation: "two thousand, forty" is ``2000 40`` on both sides. Punctuation that
belongs to a number or a word does not split a phrase: the apostrophe, ``%``, a hyphen between letters or
digits ("forty-eight"), and a period or comma between digits ("3.5", "3,200").

A known trade-off: a comma inside one spoken number splits it too, so "three thousand, two hundred" reads
``3000 200`` (and a transcript's "3,200" reads 3200). Spoken text rarely puts a comma there; a caller who
does can leave it out of an exact span or write the number without the comma.

One consequence: Whisper's normaliser deletes words in parentheses, but parentheses split phrases here, so
words in parentheses are read. The service speaks text as sent (section 9.1), parentheses included, so
those words are in the audio and must be in the comparison.

Both sides of an exact-span comparison are read this way, so a span confirms **which number** was spoken, not
how it was worded ("thirty-two hundred" and "three thousand two hundred" are both ``3200``). The version
enters the analysis key (section 10.2): any change to this behaviour is a new version of that name.

``NumberReader.whisper`` is the normaliser alone, on the whole text, with none of the steps above.
``wer_raw`` uses it, so the raw word error rate stays comparable with the bake-off's figures.

The normaliser and its spelling map are vendored from openai/whisper (MIT) in ``_whisper/`` (plan.md P2). The
map, ``english.json``, is the one Hugging Face ships as ``normalizer.json`` beside openai/whisper-large-v3,
less one entry that only Hugging Face's copy has, "mm" → "hmm". Whisper's normaliser deletes a standalone "mm"
before it maps spellings, so the entry matters only for "mm" split off digits (written "5mm", which spoken text
never contains).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from narration.contracts.names import NUMBER_READER

from ._whisper.english import EnglishTextNormalizer

__all__ = [
    "APOSTROPHES",
    "NumberReader",
    "fold_apostrophes",
    "letters_and_digits",
    "letters_only",
    "phrases",
    "remove_apostrophes",
]

APOSTROPHES: Final = frozenset("'’‘ʼ")
"""The characters ``remove_apostrophes`` deletes: the ASCII apostrophe, the curly ones and the modifier letter."""

_FOLD: Final = str.maketrans(dict.fromkeys("’‘ʼ", "'"))
_ANNOTATION: Final = re.compile(r"[<\[][^>\]]*[>\]]")
_NOUGHT: Final = re.compile(r"\bnought\b", re.IGNORECASE)
_HYPHENS: Final = frozenset("-‐‑")  # hyphen-minus, hyphen, non-breaking hyphen
_SPACES: Final = re.compile(r"\s+")
_CLOCK: Final = re.compile(r"(?<!\d)(\d{1,2}):(\d\d)(?!\d)")
"""A clock time: "4:30", "10:30"."""
_MINUS: Final = re.compile(r"(?<![\w.,:\-−])[-−](?=\d)")
"""A hyphen-minus or minus sign at the start of a word, directly before a digit: "-5", "(-5)"."""
_MONEY: Final = re.compile(r"([$£€])(\d+)\.(\d\d)(?!\d)")
"""An amount with pence or cents, as the normaliser writes it: "$3.50", "£3.50"."""
_DEGREES: Final = (
    (re.compile(r"(?:°C|℃)(?![A-Za-z])"), " degrees celsius "),
    (re.compile(r"(?:°F|℉)(?![A-Za-z])"), " degrees fahrenheit "),
    (re.compile(r"°"), " degrees "),
)
"""The degree sign, which the normaliser drops: "5°C" reads as "five degrees celsius" does."""


def fold_apostrophes(text: str) -> str:
    """Read ’ ‘ and ʼ as the straight apostrophe."""
    return text.translate(_FOLD)


def _splits(text: str, i: int) -> bool:
    """Whether the character at ``i`` ends a phrase: punctuation (category P*) that is not part of a number or
    a word."""
    ch = text[i]
    if not unicodedata.category(ch).startswith("P") or ch in "'%":
        return False
    before = text[i - 1] if i > 0 else ""
    after = text[i + 1] if i + 1 < len(text) else ""
    if ch in _HYPHENS and before.isalnum() and after.isalnum():
        return False
    if ch in ".,:" and before.isdigit() and after.isdigit():
        return False
    return not (ch == "." and after.isdigit() and not before.isalnum())


def phrases(text: str) -> tuple[str, ...]:
    """``text`` split at punctuation that ends a phrase (see the module docstring), empty phrases dropped."""
    out: list[str] = []
    start = 0
    for i in range(len(text)):
        if _splits(text, i):
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return tuple(p for p in out if p.strip())


class NumberReader:
    """The number reader ``names.NUMBER_READER`` (see the module docstring).

    One instance loads the spelling map once; reading text changes no state, so an instance can be shared.
    """

    version: Final = NUMBER_READER

    def __init__(self) -> None:
        self._normaliser = EnglishTextNormalizer()

    def read(self, text: str) -> str:
        """``text`` normalised for comparison, phrase by phrase.

        >>> NumberReader().read("nought point seven two")
        '0.72'
        >>> NumberReader().read("In the year two thousand, forty ships sailed.")
        'in the year 2000 40 ships sailed'
        """
        text = _NOUGHT.sub("zero", _ANNOTATION.sub(" ", fold_apostrophes(text)))
        text = _MINUS.sub("minus ", _CLOCK.sub(r"\1\2", text))
        for pattern, spoken in _DEGREES:
            text = pattern.sub(spoken, text)
        read = " ".join(_MONEY.sub(r"\1\2 \3", self._normaliser(p)) for p in phrases(text))
        return _SPACES.sub(" ", read).strip()

    def words(self, text: str) -> tuple[str, ...]:
        """``read(text)`` split into words."""
        return tuple(self.read(text).split())

    def whisper(self, text: str) -> str:
        """Whisper's English normaliser alone, on the whole text, stripped of leading and trailing spaces."""
        return self._normaliser(text).strip()


def remove_apostrophes(text: str) -> str:
    """Delete every apostrophe (``APOSTROPHES``), so "guide's" and "guides" become the same word."""
    return "".join(ch for ch in text if ch not in APOSTROPHES)


def letters_and_digits(text: str) -> str:
    """The letters and digits of ``text``, case-folded, with accents dropped: what the term check compares once
    both sides are read (section 11.1), so a number inside a term counts ("Seven Sisters" reads "7 sisters")."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if ch.isalnum() and not unicodedata.combining(ch)).casefold()


def letters_only(text: str) -> str:
    """The letters of ``text``, case-folded, with accents dropped: what the term check compares (section 11.1)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if ch.isalpha() and not unicodedata.combining(ch)).casefold()
