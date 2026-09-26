"""The text rules of design section 9.1, as data (``names.TEXT_CHECKS_VERSION``, text-1.1.0).

Everything the text pipeline decides by is defined here, and ``rules_document`` puts all of it into one
JSON object whose sha256 is the "rules hash" that goes into every sidecar (section 9). A change to any rule
below changes that hash; a change to what a rule *means* also needs a new ``TEXT_CHECKS_VERSION``, which is
a contract (AGENTS.md section 3).

Character classes use Python's Unicode database (``unicodedata.unidata_version``), which is part of the
rules document, so the hash also records which Unicode version classified the text.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Final

from narration.contracts.names import TEXT_CHECKS_VERSION

# ---------------------------------------------------------------- canonical form (section 7.2)
WHITESPACE: Final[frozenset[str]] = frozenset(
    "\t\n\r"  # the only control characters that count as whitespace (lead ruling, 2026-09-26)
    "   "
    "           "
    "    　"
)
"""What canonical form collapses: every run of these becomes one space (U+0020), and the ends are trimmed.

These are the Unicode ``White_Space`` characters, less the control characters VT, FF and NEL: tab, line
feed and carriage return are whitespace, and every other control character is refused (lead ruling,
2026-09-26). No-break, thin and ideographic spaces are whitespace too, so they become a plain space.
"""

KEPT_CONTROLS: Final[frozenset[str]] = frozenset("\t\n\r")
"""Control characters (category Cc) that are not refused, because they are whitespace."""

# ---------------------------------------------------------------- refused characters (section 9.1 step 1)
DEFAULT_REFUSE: Final[tuple[str, ...]] = ("[", "]", "<|", "|>")
"""The markup the engine could take as instructions (section 16 ``[text] refuse``; config may set it)."""

# ---------------------------------------------------------------- text checks (section 9.1 step 3.2)
PROSODY_PUNCTUATION: Final[frozenset[str]] = frozenset(
    [".", ",", ";", ":", "!", "?", "'", "’", "‘", '"', "“", "”", "(", ")", "-", "–", "—", "…"]
)
"""The punctuation a reader voices as prosody, exactly the design's list; it raises no ``symbol`` finding."""

LETTER_EXCEPTIONS: Final[frozenset[str]] = frozenset({"a", "A", "I"})
"""Lone letters that are ordinary words, so they raise no ``letter`` finding."""

UNIT_LETTERS: Final[frozenset[str]] = frozenset({"m", "s", "g"})
"""The lone lower-case unit letters: ``unit_like`` (warn), not ``letter`` (info)."""

SI_PREFIXES: Final[tuple[str, ...]] = (
    "Q", "R", "Y", "Z", "E", "P", "T", "G", "M", "k", "h", "da",
    "d", "c", "m", "µ", "μ", "n", "p", "f", "a", "z", "y", "r", "q",
)  # fmt: skip
"""Every SI prefix symbol (2022), with micro as both MICRO SIGN (U+00B5) and GREEK SMALL MU (U+03BC)."""

PREFIXED_UNITS: Final[tuple[str, ...]] = (
    # the SI base units (the kilogram through its gram)
    "m", "g", "s", "A", "K", "mol", "cd",
    # the SI derived units with special names (the ohm as U+03A9, its NFC form)
    "Hz", "N", "Pa", "J", "W", "C", "V", "F", "Ω", "S", "Wb", "T", "H", "lm", "lx", "Bq", "Gy", "Sv", "kat",
    # units accepted for use with the SI that take prefixes
    "L", "l", "t", "eV",
    # common non-SI units written with SI prefixes
    "Wh",
)  # fmt: skip
"""Units that appear with every SI prefix in ``UNIT_SYMBOLS`` (and on their own when two or more characters)."""

COMMON_UNIT_ABBREVIATIONS: Final[tuple[str, ...]] = (
    # length, speed, area, volume
    "in", "ft", "yd", "mph", "kph", "km/h", "m/s", "nmi", "kn", "ly", "pc", "au",
    "m²", "m³", "km²", "cm²", "cm³", "mm²", "mm³",
    # mass, pressure, energy, power, sound
    "lb", "lbs", "oz", "qt", "pt", "psi", "atm", "mbar", "mmHg", "cal", "kcal", "hp", "bhp", "cc", "dB", "dBA",
    # time, rate, electricity
    "min", "hr", "hrs", "yr", "yrs", "rpm", "ppm", "ppb", "Ah", "mAh", "kVA",
    # temperature
    "°C", "°F",
    # data
    "kB", "KB", "MB", "GB", "TB", "PB", "EB", "KiB", "MiB", "GiB", "TiB",
    "kb", "Kb", "Mb", "Gb", "Tb", "bps", "kbps", "Mbps", "Gbps",
)  # fmt: skip
"""Common unit abbreviations outside the prefixed SI set (design: "such as km, kg, mph, ft")."""

UNIT_EXCLUSIONS: Final[frozenset[str]] = frozenset(
    {
        # lower case: ordinary English words (the design names "in" and "am"; the lead added "as" and "at")
        "in", "am", "as", "at", "hm", "dam", "dag", "dal", "al", "au",
        # title case: ordinary words or names in that case, e.g. at the start of a sentence
        "Ah", "Pa", "Ms", "Mm", "El", "Em", "Et",
        # all capitals: everyday initialisms, which are words in that case
        "TV", "PC", "PA", "PS", "MA", "MC", "MS", "TA", "QA", "YA", "EV", "PT",
    }
)  # fmt: skip
"""Tokens left out of the unit list because they are ordinary English words in that case (section 9.1).

Every entry is a token the list would otherwise hold (a test checks it), so an exclusion is never
decorative. The choice is a judgement, recorded here and hashed with the rules.
"""


def _unit_symbols() -> frozenset[str]:
    generated = {prefix + unit for prefix in SI_PREFIXES for unit in PREFIXED_UNITS}
    generated.update(unit for unit in PREFIXED_UNITS if len(unit) >= 2)
    generated.update(COMMON_UNIT_ABBREVIATIONS)
    return frozenset(symbol for symbol in generated if len(symbol) >= 2) - UNIT_EXCLUSIONS


UNIT_SYMBOLS: Final[frozenset[str]] = _unit_symbols()
"""The service's generic list of unit symbols of two or more characters, matched case-sensitively against a
standalone token (a word with its leading and trailing punctuation set aside)."""


def rules_document(refuse: tuple[str, ...]) -> dict[str, Any]:
    """Every rule the text pipeline applies, as one JSON object (hashed into ``rules_sha256``).

    ``refuse`` is the configured markup list (``[text] refuse``); everything else is fixed by this module.
    Identifiers name each rule tersely; the code that implements them is ``narration.text``.
    """
    return {
        "version": TEXT_CHECKS_VERSION,
        "unicode": unicodedata.unidata_version,
        "canonical_form": {
            "normalization": "NFC",
            "whitespace": sorted(f"U+{ord(ch):04X}" for ch in WHITESPACE),
            "collapse": "run->U+0020,trim",
        },
        "refuse": {
            "strings": list(refuse),
            "control": "Cc-except-U+0009,U+000A,U+000D",
            "surrogate": "Cs",
        },
        "word": {"split": "whitespace", "edges": "category-P"},
        "hints": {
            "match": "case-sensitive,nfc,whole-word",
            "boundary": "start|end|U+0020|category-P",
            "order": "longest-first,then-term",
            "cross_cue": "not-applied,TERM_SPLIT_ACROSS_CUES",
            "empty_respell": "refused",
            "duplicate_term": "refused",
        },
        "checks": {
            "on": "engine-text,offsets-in-spoken-text",
            "digit": {"class": "str.isdigit", "token": "run"},
            "symbol": {
                "class": "not-letter,not-space,not-prosody,not-digit",
                "prosody": sorted(PROSODY_PUNCTUATION),
                "marks": "category-M-takes-preceding-class",
                "token": "run",
            },
            "unit_like": {"symbols": sorted(UNIT_SYMBOLS), "lone": sorted(UNIT_LETTERS), "token": "word"},
            "letter": {"severity": "info", "except": sorted(LETTER_EXCEPTIONS), "token": "word"},
        },
        "strict_text": "refuses-warn-severity-only",
    }
