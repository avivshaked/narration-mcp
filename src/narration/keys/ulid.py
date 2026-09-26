"""ULIDs for job and design ids (design section 5: ``job_id`` and ``design_id`` are ULIDs).

A ULID is 128 bits: a 48-bit Unix time in milliseconds, then 80 random bits, written as 26 characters of
Crockford's base32 (``0-9 A-Z`` without ``I L O U``), upper case. Ids from one ``UlidGenerator`` are
strictly increasing: within one millisecond, or if the clock steps back, the previous id's random part
is incremented instead of drawing a new one (the "monotonic" variant of the ULID specification).

Standard library only; the generator's state lives in its instance, never in the module.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable
from typing import Final

from narration.contracts import names

CROCKFORD: Final = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ULID_CHARS: Final = 26
ULID_PATTERN: Final = re.compile(names.ID_PATTERNS["design_id"])
"""A canonical ULID: upper case, and the first character at most 7 (the value fits in 128 bits). It is
``names.ID_PATTERNS["design_id"]``, unanchored: use it with ``fullmatch``."""

_TIME_BITS: Final = 48
_RANDOM_BITS: Final = 80
_MAX_TIME: Final = (1 << _TIME_BITS) - 1
_MAX_RANDOM: Final = (1 << _RANDOM_BITS) - 1
_DECODE: Final = {c: i for i, c in enumerate(CROCKFORD)}


def encode(value: int) -> str:
    """The 26-character Crockford base32 form of a 128-bit value."""
    if not 0 <= value < 1 << 128:
        raise ValueError("a ULID is a 128-bit unsigned value")
    chars = []
    for _ in range(ULID_CHARS):
        chars.append(CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def decode(ulid: str) -> int:
    """The 128-bit value of a canonical ULID; raises ValueError for anything else."""
    if not is_ulid(ulid):
        raise ValueError(f"not a canonical ULID: {ulid!r}")
    value = 0
    for char in ulid:
        value = (value << 5) | _DECODE[char]
    return value


def is_ulid(text: str) -> bool:
    """Whether ``text`` is a canonical (upper-case) ULID."""
    return ULID_PATTERN.fullmatch(text) is not None


def timestamp_ms(ulid: str) -> int:
    """The Unix time in milliseconds that a ULID carries."""
    return decode(ulid) >> _RANDOM_BITS


class UlidGenerator:
    """Makes ULIDs that increase strictly within this generator, across threads.

    ``clock_ms`` and ``random_bytes`` can be replaced for tests; by default they are the system clock and
    ``os.urandom``.
    """

    def __init__(
        self,
        *,
        clock_ms: Callable[[], int] | None = None,
        random_bytes: Callable[[int], bytes] | None = None,
    ) -> None:
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._random_bytes = random_bytes or os.urandom
        self._lock = threading.Lock()
        self._last_ms = -1
        self._last_random = 0

    def new(self) -> str:
        """A new ULID, greater than every ULID this generator made before."""
        with self._lock:
            ms = self._clock_ms()
            if not 0 <= ms <= _MAX_TIME:
                raise ValueError(f"clock value {ms} ms does not fit a ULID's 48-bit time")
            if ms <= self._last_ms:
                ms = self._last_ms
                random = self._last_random + 1
                if random > _MAX_RANDOM:
                    # 2**80 ids in one millisecond cannot happen; if it does, borrow the next millisecond.
                    ms += 1
                    random = self._draw()
            else:
                random = self._draw()
            self._last_ms, self._last_random = ms, random
            return encode((ms << _RANDOM_BITS) | random)

    def _draw(self) -> int:
        raw = self._random_bytes(_RANDOM_BITS // 8)
        if len(raw) != _RANDOM_BITS // 8:
            raise ValueError("random_bytes returned the wrong number of bytes")
        return int.from_bytes(raw, "big")
