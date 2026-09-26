"""Small helpers shared by the store's tests."""

from __future__ import annotations

from narration.store.store import utc_iso


def utc(seconds: float) -> str:
    """The store's own ISO form of a Unix time."""
    return utc_iso(seconds)
