"""What the front-end checks and keeps for the DESIGN step's tools and ``audition_pronunciation`` (design
sections 3.1, 3.5, 3.6, 7.6, 17.3, 17.4), before it queues their jobs.

Their job handlers are WP34's (``design``, ``profile``) and WP35's (``pronunciation``). Each job keeps its
request by value, as a ``generate`` job does, and its identity (section 7.3) leaves out what does not change
the work: the opaque ``name`` of a design, and the ``design_id`` the front-end mints at submit.

- ``design_voice``: the description is linted (section 3.5, policy ``warn``: the design runs as asked, and
  ``lint`` lists the findings), held to ``[limits] max_description_chars``, and refused if it holds the
  model's chat markup or a control character (``narration.design.description``); the design text (the
  caller's, else ``[voice_design] design_text``) passes the text pipeline's refusals, since it is spoken.
- ``audition_pronunciation``: each variant's label is unique; the carrier, when given, contains the term
  (the respelling is applied as a hint, so the pipeline's own matching decides); every text passes the
  pipeline's refusals. The voice is checked as a clone's is (section 17.4) but need not be measured.
- ``profile_voice``: only the audio's path, sha256 and format are checked (``clips``); any WAV the owner can
  read may be profiled (section 17.3).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Final

from narration.config import Config
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import TextPlanner
from narration.contracts.models import Hint, SegmentIn
from narration.design.description import check_description

from .clips import refield

DESIGN_TAKES: Final = 3
"""``design_voice``'s default number of candidates (section 7.6)."""
DESIGN_UNKEYED: Final = ("name", "design_id")
"""What a design request's identity leaves out: an opaque label, and the id minted for the job."""


def _spoken(text: TextPlanner, spoken: str, field: str, hints: tuple[Hint, ...] = ()) -> Any:
    """The text pipeline's plan of one text the service will speak; its refusals name ``field``."""
    try:
        (planned,) = text.plan_request([SegmentIn(segment_id="s", text=spoken)], list(hints), strict_text=False)
    except NarrationError as exc:
        raise refield(exc, field) from exc
    return planned


def design_request(args: Mapping[str, Any], config: Config, text: TextPlanner) -> dict[str, Any]:
    """The request a ``design`` job keeps (without its ``design_id``): the arguments as sent, with ``takes``
    and ``design_text`` given the service's defaults. ``LIMIT_EXCEEDED`` for a description over
    ``[limits] max_description_chars``; ``TEXT_REFUSED`` for a description with the model's chat markup or a
    control character (``narration.design.description``), and the pipeline's refusals for the design text."""
    description = str(args["description"])
    limit = config.limits.max_description_chars
    if len(description) > limit:
        raise NarrationError(
            codes.LIMIT_EXCEEDED,
            f"the description is {len(description)} characters; this service takes at most {limit}",
            field="description",
            hint="Shorten the description: name the few qualities that matter most.",
            details={"chars": len(description), "max_description_chars": limit},
        )
    check_description(description)
    design_text = str(args.get("design_text") or config.voice_design.design_text)
    _spoken(text, design_text, "design_text")
    stored = copy.deepcopy(dict(args))
    stored["takes"] = int(args.get("takes", DESIGN_TAKES))
    stored["design_text"] = design_text
    return stored


def design_identity(stored: Mapping[str, Any]) -> dict[str, Any]:
    """A design request without what its identity leaves out (``DESIGN_UNKEYED``)."""
    return {k: v for k, v in stored.items() if k not in DESIGN_UNKEYED}


def audition_request(args: Mapping[str, Any], text: TextPlanner) -> dict[str, Any]:
    """The request a ``pronunciation`` job keeps: the arguments as sent, after the checks in the module
    docstring (``INVALID_ARGUMENT`` for a repeated label or a carrier without the term)."""
    term = str(args["term"])
    carrier = args.get("carrier")
    seen: dict[str, int] = {}
    for i, variant in enumerate(args["variants"]):
        label = str(variant["label"])
        if label in seen:
            raise NarrationError(
                codes.INVALID_ARGUMENT,
                f"variants[{i}] has the label {label!r} of variants[{seen[label]}]; each variant needs its own",
                field=f"variants[{i}].label",
            )
        seen[label] = i
        hint = Hint(term=term, respell=str(variant["respell"]))
        spoken = str(carrier) if carrier else term
        planned = _spoken(text, spoken, "carrier" if carrier else "term", (hint,))
        if carrier and not any(c.hints_applied for c in planned.cues):
            raise NarrationError(
                codes.INVALID_ARGUMENT,
                f"the carrier does not contain the term {term!r} as a whole word",
                field="carrier",
                hint="Write the term into the carrier sentence exactly as the script spells it, or leave the "
                "carrier out to hear the term alone.",
            )
    return copy.deepcopy(dict(args))


__all__ = ["DESIGN_TAKES", "DESIGN_UNKEYED", "audition_request", "design_identity", "design_request"]
