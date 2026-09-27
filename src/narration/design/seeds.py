"""The seeds of a design's candidates (design sections 3.1 and 10.3): derived from the request alone.

``design_voice`` designs ``takes`` candidates "with n derived seeds" (section 3.1). As a take's seed follows from
its request (section 10.3), a candidate's seed follows from what the design asks for, so the same design always
asks for the same seeds, and a caller may never send one. Byte for byte::

    parts = ["narration-design-seed/v1", description_sha256, sha256_hex(design text as UTF-8), str(index)]
    digest = sha256(b"\\x00".join(p.encode("utf-8") for p in parts))
    seed = int.from_bytes(digest[0:4], "big") & 0x7FFFFFFF

- ``description_sha256`` is the candidate's own: 64 hex of the description as sent, UTF-8 (section 3.5).
- The design text is the text the model is given: its spoken form (NFC, each run of whitespace one space).
- ``index`` is the candidate's number, 0 .. takes-1, written in decimal.

Left out on purpose: the engine profile (as a take's seed leaves it out, so a re-pinned engine asks for the same
seeds), the design id and the opaque ``name`` (so the same description and text give the same seeds under any
name). The seed is masked to 31 bits, as a take's is (``keys.SEED_MASK``).

KNOW (plan.md section 9, 2026-09-27): on the pinned VoiceDesign profile a design is deterministic, so the same
description, text and seed give the same clip; and a seed does not hold a voice across texts, so a changed
design text gives other voices, as it gives other seeds here.
"""

from __future__ import annotations

import hashlib
import re
from typing import Final

from narration import keys

DESIGN_SEED_SCHEME: Final = "narration-design-seed/v1"
"""The seed scheme of a design's candidates (not named by the design; chosen here, like ``names.SEED_SCHEME``)."""
MAX_INDEX: Final = 999
"""The highest candidate index the store's layout takes (``designs/<design_id>/<index>/``)."""
_HEX64: Final = re.compile(r"[0-9a-f]{64}")


def design_seed_message(*, description_sha256: str, design_text: str, index: int) -> bytes:
    """The bytes a candidate's seed hashes, exactly as the module docstring pins them."""
    if not _HEX64.fullmatch(description_sha256):
        raise ValueError("description_sha256 must be 64 lowercase hex")
    if not design_text:
        raise ValueError("the design text must not be empty")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= MAX_INDEX:
        raise ValueError(f"a candidate index is an integer from 0 to {MAX_INDEX}")
    parts = [
        DESIGN_SEED_SCHEME,
        description_sha256,
        hashlib.sha256(design_text.encode("utf-8")).hexdigest(),
        str(index),
    ]
    return b"\x00".join(p.encode("utf-8") for p in parts)


def design_seed(*, description_sha256: str, design_text: str, index: int) -> int:
    """The seed of candidate ``index`` of a design (the module docstring)."""
    digest = hashlib.sha256(
        design_seed_message(description_sha256=description_sha256, design_text=design_text, index=index)
    ).digest()
    return int.from_bytes(digest[0:4], "big") & keys.SEED_MASK


def description_sha256(description: str) -> str:
    """``description_sha256`` (section 3.5): 64 hex of the description as sent, UTF-8."""
    return hashlib.sha256(description.encode("utf-8")).hexdigest()


__all__ = ["DESIGN_SEED_SCHEME", "MAX_INDEX", "description_sha256", "design_seed", "design_seed_message"]
