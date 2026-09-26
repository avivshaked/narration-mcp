"""The text pipeline (design section 9.1; WP10): the service speaks the text it is given, and says exactly
what it did to it.

``TextPipeline`` implements ``narration.contracts.interfaces.TextPlanner``. The helpers are the service's
single definitions of canonical form, words, the join and spoken length (section 7.2), which the aligner
and QA use too, and of the over-long segment warning (section 3.2).
"""

from __future__ import annotations

from .canonical import Token, Word, canonical_form, join, spoken_length, tokens, words
from .length import segment_too_long
from .planner import TextPipeline, rules_sha256

__all__ = [
    "TextPipeline",
    "Token",
    "Word",
    "canonical_form",
    "join",
    "rules_sha256",
    "segment_too_long",
    "spoken_length",
    "tokens",
    "words",
]
