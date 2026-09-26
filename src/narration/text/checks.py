"""The four text checks (design section 9.1 step 3.2). They change nothing; each finding is a warning.

- ``digit``: a run of digits (``str.isdigit``: decimal digits of every script, superscripts and the like);
- ``symbol``: a run of characters that are not a letter, a space, a digit, or the punctuation a reader
  voices as prosody (``rules.PROSODY_PUNCTUATION``). A combining mark takes the class of the character it
  follows, so ``e`` + a combining accent is a letter; a digit is reported as ``digit`` only, never twice;
- ``unit_like``: a word (a token with its leading and trailing punctuation set aside) in the service's
  generic unit list (``rules.UNIT_SYMBOLS``), or a lone ``m``, ``s`` or ``g``;
- ``letter`` (info): any other lone letter except ``a``, ``A`` and ``I``.

The checks look at what the engine will be given: the cue's spoken text with each respelled term replaced
by its respelling. A term that has a respelling is not read by the engine as written, so its own
characters raise nothing; the respelling is checked instead. Every finding is located in the cue's spoken
text: a finding inside a respelling is reported at its term's offset, with ``respell_of`` naming the term.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from narration.contracts.names import TextWarningKind

from .canonical import is_whitespace, tokens
from .hints import Origin
from .rules import LETTER_EXCEPTIONS, PROSODY_PUNCTUATION, UNIT_LETTERS, UNIT_SYMBOLS

KIND_ORDER: Final[dict[TextWarningKind, int]] = {"digit": 0, "symbol": 1, "unit_like": 2, "letter": 3}


@dataclass(frozen=True, slots=True)
class Finding:
    """One text-check finding: ``offset`` is in code points into the cue's spoken text."""

    kind: TextWarningKind
    token: str
    offset: int
    respell_of: str | None = None


def _classes(text: str) -> list[str]:
    out: list[str] = []
    previous = "space"
    for ch in text:
        if is_whitespace(ch):
            cls = "space"
        elif ch in PROSODY_PUNCTUATION:
            cls = "prosody"
        elif ch.isdigit():
            cls = "digit"
        else:
            major = unicodedata.category(ch)[0]
            if major == "L":
                cls = "letter"
            elif major == "M":
                cls = previous if previous in ("letter", "digit", "symbol") else "symbol"
            else:
                cls = "symbol"
        out.append(cls)
        previous = cls
    return out


def _is_lone_letter(core: str) -> bool:
    return (
        unicodedata.category(core[0]).startswith("L")
        and all(unicodedata.category(ch).startswith("M") for ch in core[1:])
        and core not in LETTER_EXCEPTIONS
    )


def check(engine: str, origin: Sequence[Origin], respelled_terms: Sequence[str] = ()) -> tuple[Finding, ...]:
    """Run the four checks over a cue's engine text; ``origin`` maps each engine code point to the spoken
    text (``hints.apply_hints``). Findings come in order of offset, then kind (digit, symbol, unit_like,
    letter), then position in the engine text."""

    def source(i: int) -> int | None:
        return origin[i][1]

    def finding(kind: TextWarningKind, start: int, end: int) -> tuple[int, int, int, Finding]:
        offset, app = origin[start]
        term = None if app is None else respelled_terms[app]
        return (offset, KIND_ORDER[kind], start, Finding(kind, engine[start:end], offset, term))

    found: list[tuple[int, int, int, Finding]] = []
    classes = _classes(engine)
    i, n = 0, len(engine)
    while i < n:
        cls = classes[i]
        if cls not in ("digit", "symbol"):
            i += 1
            continue
        start = i
        while i < n and classes[i] == cls and source(i) == source(start):
            i += 1
        found.append(finding("digit" if cls == "digit" else "symbol", start, i))

    for token in tokens(engine):
        if not token.has_core:
            continue
        core = engine[token.core_start : token.core_end]
        if core in UNIT_SYMBOLS or core in UNIT_LETTERS:
            found.append(finding("unit_like", token.core_start, token.core_end))
        elif _is_lone_letter(core):
            found.append(finding("letter", token.core_start, token.core_end))

    found.sort(key=lambda f: f[:3])
    return tuple(f[3] for f in found)
