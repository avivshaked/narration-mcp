"""The aligner's alphabet (design section 11.2 step 1): a word's letters, spelled for wav2vec2.

The pinned model (``facebook/wav2vec2-large-960h-lv60-self``) spells English in 26 capital letters and the
apostrophe, with ``|`` between words. A word of the spoken text is spelled into those tokens:

- a letter is folded to A–Z: accents are dropped (Unicode NFKD, combining marks removed), and a few letters
  with no decomposition are folded by ``LETTER_FOLDS`` (``ß`` → ``SS``, ``æ`` → ``AE``, ``ø`` → ``O``, …);
- an apostrophe (``'``, ``’``, ``‘`` or the modifier letter ``ʼ``) is the apostrophe token;
- a hyphen or dash inside a word is a word separator, ``|`` (``forty-eight`` → ``FORTY|EIGHT``);
- the other punctuation a reader voices as prosody (``narration.text.rules.PROSODY_PUNCTUATION``: full
  stops, commas, quotes, brackets, the ellipsis) is silent inside a word and is dropped (``U.S`` → ``US``);
- anything else, a digit or a symbol, is outside the alphabet, and the whole word cannot be spelled. Those
  are exactly the characters the text checks already warn about (section 9.1: ``digit``, ``symbol``), so
  a word is unspellable only when its cue already carries a text warning. The transcript puts the
  wildcard, ``*``, in place of each run of such words (plan.md DC-11; ``transcript``).
"""

from __future__ import annotations

import unicodedata
from typing import Final

from narration.text.rules import PROSODY_PUNCTUATION

LETTERS: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
APOSTROPHE: Final = "'"
WORD_SEPARATOR: Final = "|"
ALPHABET: Final[frozenset[str]] = frozenset(LETTERS + APOSTROPHE + WORD_SEPARATOR)
"""The model's own tokens: its letters, the apostrophe and the word separator."""
WILDCARD: Final = "*"
"""The token for a run of words outside the alphabet (plan.md DC-11). It is not the model's: the worker aligns
it through an extra emission column that can absorb any speech but not silence."""

APOSTROPHES: Final[frozenset[str]] = frozenset("'’‘ʼ")
"""Spelled as the apostrophe token: ' ’ ‘ and ʼ (MODIFIER LETTER APOSTROPHE, a letter by category)."""
DASHES: Final[frozenset[str]] = frozenset(ch for ch in PROSODY_PUNCTUATION if unicodedata.category(ch) == "Pd")
"""Hyphens and dashes of the prosody list (``-``, ``–``, ``—``): a word separator inside a word."""
SILENT: Final[frozenset[str]] = PROSODY_PUNCTUATION - APOSTROPHES - DASHES
"""The rest of the prosody punctuation: nothing is spoken for it inside a word."""

LETTER_FOLDS: Final[dict[str, str]] = {
    "ß": "SS",
    "ẞ": "SS",
    "Æ": "AE",
    "æ": "AE",
    "Œ": "OE",
    "œ": "OE",
    "Ø": "O",
    "ø": "O",
    "Đ": "D",
    "đ": "D",
    "Ð": "D",
    "ð": "D",
    "Ħ": "H",
    "ħ": "H",
    "Ł": "L",
    "ł": "L",
    "Ŀ": "L",
    "ŀ": "L",
    "Þ": "TH",
    "þ": "TH",
}
"""Latin letters that Unicode does not decompose into A–Z, with the letters an English reader would say."""


def fold_letter(ch: str) -> str | None:
    """A letter's spelling in A–Z (one or more letters), or None if it has none (another script)."""
    if ch in LETTER_FOLDS:
        return LETTER_FOLDS[ch]
    decomposed = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.category(c).startswith("M"))
    folded = decomposed.upper()
    if folded and all(c in LETTERS for c in folded):
        return folded
    return None


def spell(text: str) -> tuple[str, ...] | None:
    """The tokens of one word (its text with the leading and trailing punctuation set aside).

    Letters and apostrophes become tokens; a run of hyphens or dashes between letters becomes one ``|``;
    silent punctuation and combining marks are dropped. Returns None when a character is outside the
    alphabet (a digit, a symbol, a letter of another script), or when no letter is left: the word is then
    left out of the transcript, and the words around it still align (section 11.2 step 1).
    """
    tokens: list[str] = []
    for ch in text:
        if ch in APOSTROPHES:
            tokens.append(APOSTROPHE)
        elif ch in DASHES:
            if tokens and tokens[-1] != WORD_SEPARATOR:
                tokens.append(WORD_SEPARATOR)
        elif ch in SILENT or unicodedata.category(ch).startswith("M"):
            continue
        elif unicodedata.category(ch).startswith("L"):
            folded = fold_letter(ch)
            if folded is None:
                return None
            tokens.extend(folded)
        else:
            return None
    while tokens and tokens[-1] == WORD_SEPARATOR:
        tokens.pop()
    if not any(t in LETTERS for t in tokens):
        return None
    return tuple(tokens)
