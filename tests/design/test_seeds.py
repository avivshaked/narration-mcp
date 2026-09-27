"""The candidates' seeds (design sections 3.1 and 10.3): derived from the request alone, byte for byte."""

from __future__ import annotations

import hashlib

import pytest

from narration.design.seeds import DESIGN_SEED_SCHEME, description_sha256, design_seed, design_seed_message

from .support import MARSH, WARM


def test_the_seed_is_the_pinned_derivation_s10_3() -> None:
    digest = description_sha256(WARM)
    assert digest == hashlib.sha256(WARM.encode("utf-8")).hexdigest()
    parts = [DESIGN_SEED_SCHEME, digest, hashlib.sha256(MARSH.encode("utf-8")).hexdigest(), "2"]
    message = b"\x00".join(p.encode("utf-8") for p in parts)
    assert design_seed_message(description_sha256=digest, design_text=MARSH, index=2) == message
    expected = int.from_bytes(hashlib.sha256(message).digest()[0:4], "big") & 0x7FFFFFFF
    assert design_seed(description_sha256=digest, design_text=MARSH, index=2) == expected


def test_the_scheme_is_pinned_s10_3() -> None:
    """A change to the scheme would give every design new voices: this value may only change on purpose."""
    assert DESIGN_SEED_SCHEME == "narration-design-seed/v1"
    digest = description_sha256("A calm voice.")
    assert design_seed(description_sha256=digest, design_text="One two three.", index=0) == 750949783


def test_each_candidate_description_and_text_has_its_own_seed_s3_1() -> None:
    digest = description_sha256(WARM)
    seeds = {design_seed(description_sha256=digest, design_text=MARSH, index=i) for i in range(4)}
    assert len(seeds) == 4
    other_text = design_seed(description_sha256=digest, design_text=MARSH + " Then it rains.", index=0)
    other_description = design_seed(description_sha256=description_sha256(WARM + " "), design_text=MARSH, index=0)
    first = design_seed(description_sha256=digest, design_text=MARSH, index=0)
    assert first not in (other_text, other_description)
    assert all(0 <= s <= 0x7FFFFFFF for s in seeds)


@pytest.mark.parametrize(
    ("digest", "text", "index"),
    [("A" * 64, MARSH, 0), ("a" * 63, MARSH, 0), ("a" * 64, "", 0), ("a" * 64, MARSH, -1), ("a" * 64, MARSH, 1000)],
)
def test_malformed_seed_inputs_are_refused_s10_3(digest: str, text: str, index: int) -> None:
    with pytest.raises(ValueError):
        design_seed_message(description_sha256=digest, design_text=text, index=index)
