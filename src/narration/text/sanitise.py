"""Sanitising text before anything else (design section 9.1 step 1, section 17 items 5 and 6).

Refused, with ``TEXT_REFUSED`` and every offender listed:

- the configured markup strings (``[text] refuse``: by default ``[``, ``]``, ``<|`` and ``|>``), which the
  engine could take as instructions;
- control characters (category Cc), except tab, line feed and carriage return, which are whitespace and
  which canonical form collapses (lead ruling, 2026-09-26);
- lone surrogates (category Cs), which are not text: they cannot be encoded as UTF-8, so they could be
  neither hashed into a key nor sent to a worker.

Format characters (category Cf: zero-width space, joiners, bidirectional marks, the byte order mark, the
soft hyphen) are **not** refused. They are text a caller may have pasted without seeing, so the ``symbol``
check reports each one with its code point, and the text is spoken as sent.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

from .rules import KEPT_CONTROLS

OffenderReason = Literal["markup", "control", "surrogate"]


@dataclass(frozen=True, slots=True)
class Offender:
    """One refused character or markup string, at ``offset`` code points into the text as sent."""

    text: str
    offset: int
    reason: OffenderReason

    @property
    def codepoints(self) -> str:
        """The offender's code points in ``U+XXXX`` form, so an invisible one can be found."""
        return " ".join(f"U+{ord(ch):04X}" for ch in self.text)


def find_offenders(text: str, refuse: tuple[str, ...]) -> tuple[Offender, ...]:
    """Every refused string and character in ``text``, in order of offset (then text).

    Each markup string is found at every position it occurs, so ``<|>`` lists both ``<|`` and ``|>``.
    """
    found: list[Offender] = []
    for markup in refuse:
        i = text.find(markup)
        while i != -1:
            found.append(Offender(markup, i, "markup"))
            i = text.find(markup, i + 1)
    for i, ch in enumerate(text):
        category = unicodedata.category(ch)
        if category == "Cc" and ch not in KEPT_CONTROLS:
            found.append(Offender(ch, i, "control"))
        elif category == "Cs":
            found.append(Offender(ch, i, "surrogate"))
    found.sort(key=lambda o: (o.offset, o.text))
    return tuple(found)
