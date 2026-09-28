"""What a design's description may hold (design sections 3.5 and 17 item 5).

The description goes to Qwen VoiceDesign verbatim, inside the model's chat template (``<|im_start|>user`` …
``<|im_end|>``), whose tokenizer reads those markers as special tokens. So the template's markup, ``<|`` and
``|>``, is refused, and so are control characters other than tab, line feed and carriage return, and lone
surrogates (which are not text: they cannot be hashed or sent as UTF-8). The rules are the text pipeline's
(``narration.text.sanitise``), with the markup narrowed to the chat template's own. Brackets are kept, since a
description is not spoken and its lint (section 3.5) only warns.

The front-end checks a description at submit (``narration.backend.steps.design_request``), and the design job
again before it sends one to the model, as it does the design text.
"""

from __future__ import annotations

from typing import Any, Final

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.text.sanitise import find_offenders

DESCRIPTION_REFUSE: Final[tuple[str, ...]] = ("<|", "|>")
"""The markup refused in a description: the chat template's own, which the model would read as instructions."""
FIELD: Final = "description"
HINT: Final = (
    "Remove every offender listed in details.offenders (the markup <| or |>, or a control character other than tab, "
    "line feed and carriage return), then send the description again. Describe the voice in plain words."
)


def check_description(description: str) -> None:
    """Refuse a description that holds the chat template's markup, a control character other than tab, line feed
    and carriage return, or a lone surrogate: ``TEXT_REFUSED`` on ``description``, with every offender listed
    (its offset in code points, its reason and its code points)."""
    found = find_offenders(description, DESCRIPTION_REFUSE)
    if not found:
        return
    offenders: list[dict[str, Any]] = [
        {
            "field": FIELD,
            "text": o.text if o.reason != "surrogate" else "",
            "offset": o.offset,
            "reason": o.reason,
            "codepoints": o.codepoints,
        }
        for o in found
    ]
    raise NarrationError(
        codes.TEXT_REFUSED,
        f"the description holds {len(offenders)} refused character(s) or string(s): the markup "
        f"{' '.join(DESCRIPTION_REFUSE)}, which the model could take as instructions, or control characters. Nothing "
        "was designed.",
        field=FIELD,
        hint=HINT,
        details={"offenders": offenders},
    )


__all__ = ["DESCRIPTION_REFUSE", "check_description"]
