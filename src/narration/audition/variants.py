"""What an audition asks for, read from its request (design sections 7.6, 9.1 and 10.3): one segment per variant.

``audition_pronunciation`` renders a term with up to four respelling variants, in a carrier sentence or alone. The
front-end keeps the request by value (``narration.backend.steps.audition_request``); the handler plans it, and
``get_results`` reads it again, both through this module, so the two cannot plan a variant differently.

**A variant is a segment.** Variant ``i`` (from 0) is the segment ``variant-<i+1>``: the carrier as sent (or the
term alone, without one), with one hint, the variant's (``variant_hint``). The text pipeline applies it as it
applies any hint (section 9.1): the engine text takes the respelling where the term stands as whole words; the
spoken text stays as sent. So each variant's take is keyed, seeded and cached exactly as a ``submit_job`` take of
that text with that hint (sections 10.2, 10.3): no key of its own.

**The hint** is the term with the variant's respelling, and the respelling as the term's one ``asr_alias``. QA
collapses a hinted term to one token in ``wer_adj`` and counts it found when the recogniser wrote something close
to the term, or one of its aliases (section 11.1 steps 4 and 6). A respelling is written to be said, and what the
recogniser writes for a take that says it may read as the respelling rather than the term ("Tor-vin" heard as
"Torvin" for the term "Thorvyn"): the alias counts that take as saying the term, so ``wer_adj`` judges the
carrier's words and the respelling, not the term's spelling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from narration.contracts.models import Hint, SegmentIn, TermResult

MAX_VARIANTS: Final = 4
"""``variants`` holds 1 to 4 respellings (section 7.6)."""
SEGMENT_PREFIX: Final = "variant-"


class AuditionRequestError(ValueError):
    """A stored audition request cannot be read (it should have passed the input schema at submit)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Variant:
    """One respelling variant: the caller's label, echoed back, and the respelling."""

    label: str
    respell: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditionAsk:
    """An ``audition_pronunciation`` request as the handler and ``get_results`` read it (the voice is read apart,
    as ``narration.jobs.plan.VoiceSpec``)."""

    term: str
    carrier: str | None
    variants: tuple[Variant, ...]

    @classmethod
    def parse(cls, request: Mapping[str, Any]) -> AuditionAsk:
        """Read a stored request; ``AuditionRequestError`` for one without the input schema's shape."""
        term = request.get("term")
        carrier = request.get("carrier")
        raw = request.get("variants")
        if not isinstance(term, str) or not term:
            raise AuditionRequestError("the audition's term must be a non-empty string")
        if carrier is not None and not isinstance(carrier, str):
            raise AuditionRequestError("the audition's carrier must be a string")
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_VARIANTS:  # pyright: ignore[reportUnknownArgumentType]
            raise AuditionRequestError(f"the audition needs 1 to {MAX_VARIANTS} variants")
        variants: list[Variant] = []
        for i, item in enumerate(raw):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            label = item.get("label") if isinstance(item, Mapping) else None  # pyright: ignore[reportUnknownMemberType]
            respell = item.get("respell") if isinstance(item, Mapping) else None  # pyright: ignore[reportUnknownMemberType]
            if not isinstance(label, str) or not label or not isinstance(respell, str) or not respell:
                raise AuditionRequestError(f"variants[{i}] needs a label and a respelling")
            variants.append(Variant(label=label, respell=respell))
        return cls(term=term, carrier=carrier or None, variants=tuple(variants))

    @property
    def spoken(self) -> str:
        """What each variant speaks, as sent: the carrier, or the term alone."""
        return self.carrier if self.carrier else self.term

    def segments(self) -> tuple[tuple[SegmentIn, Hint], ...]:
        """Each variant's segment and hint, in the request's order (the module docstring)."""
        return tuple(
            (SegmentIn(segment_id=segment_id(i), text=self.spoken), variant_hint(self.term, v.respell))
            for i, v in enumerate(self.variants)
        )


def segment_id(index: int) -> str:
    """The segment id of variant ``index`` (from 0): ``variant-1`` .. ``variant-4``. It enters no key or seed."""
    return f"{SEGMENT_PREFIX}{index + 1}"


def variant_hint(term: str, respell: str) -> Hint:
    """A variant's hint: the term, the respelling for the engine, and the respelling as the term's alias for QA."""
    return Hint(term=term, respell=respell, asr_aliases=(respell,))


def heard_term(terms: Sequence[TermResult]) -> str | None:
    """What the recogniser heard in the term's place, from a variant take's ``qa.terms``: its first occurrence's
    ``heard`` (a variant's segment has one hint, so every result is the term's). None when nothing was heard
    there, or the take has no term result."""
    return terms[0].heard if terms else None


__all__ = [
    "MAX_VARIANTS",
    "SEGMENT_PREFIX",
    "AuditionAsk",
    "AuditionRequestError",
    "Variant",
    "heard_term",
    "segment_id",
    "variant_hint",
]
