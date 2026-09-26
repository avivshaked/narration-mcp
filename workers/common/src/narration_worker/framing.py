"""JSON-lines framing of the worker protocol (design Appendix A).

One message is one JSON object, encoded as UTF-8 on a single line that ends in ``\\n``. ``NaN`` and the
infinities are not JSON, so neither side writes or accepts them. A line longer than
``protocol.MAX_LINE_BYTES`` is refused. Both the worker's request loop and the daemon's client use these
functions, so the two sides frame messages the same way.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import IO, Any

from .protocol import MAX_LINE_BYTES


class FramingError(ValueError):
    """A line that is not one protocol message: not UTF-8, not JSON, or not a JSON object."""


class LineTooLong(FramingError):
    """A line longer than ``MAX_LINE_BYTES``. The rest of the line has been read and discarded."""


def encode_message(message: Mapping[str, Any]) -> bytes:
    """Encode one message as a UTF-8 JSON line ending in ``\\n``.

    Raises ``ValueError`` for a value JSON cannot hold (``NaN``, an infinity, a type ``json`` does not
    know), and ``LineTooLong`` when the line would exceed ``MAX_LINE_BYTES``.
    """
    text = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    data = text.encode("utf-8") + b"\n"
    if len(data) > MAX_LINE_BYTES:
        raise LineTooLong(f"a message of {len(data)} bytes exceeds the protocol's {MAX_LINE_BYTES}-byte line limit")
    return data


def _refuse_constant(name: str) -> Any:
    raise FramingError(f"{name} is not valid JSON")


def decode_message(line: bytes) -> dict[str, Any]:
    """Decode one line (with or without its newline) into a message object; raises ``FramingError``."""
    if len(line) > MAX_LINE_BYTES:
        raise LineTooLong(f"a line of {len(line)} bytes exceeds the protocol's {MAX_LINE_BYTES}-byte line limit")
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FramingError(f"the line is not UTF-8 ({exc.reason} at byte {exc.start})") from exc
    try:
        value = json.loads(text, parse_constant=_refuse_constant)
    except json.JSONDecodeError as exc:
        raise FramingError(f"the line is not JSON ({exc.msg} at character {exc.pos})") from exc
    except RecursionError as exc:
        raise FramingError("the line nests too deeply") from exc
    if not isinstance(value, dict):
        raise FramingError(f"a message must be a JSON object, not {type(value).__name__}")
    return value


def read_line(stream: IO[bytes], limit: int = MAX_LINE_BYTES) -> bytes | None:
    """Read one line from a binary stream, without its ``\\n``; ``None`` at the end of input.

    A line longer than ``limit`` bytes is read to its end, discarded, and reported as ``LineTooLong``, so the
    next call starts at the next line. A last line without a newline is returned as it is.
    """
    line = stream.readline(limit + 1)
    if not line:
        return None
    if line.endswith(b"\n"):
        return line[:-1]
    if len(line) <= limit:
        return line
    while True:
        rest = stream.readline(1 << 16)
        if not rest or rest.endswith(b"\n"):
            break
    raise LineTooLong(f"a line exceeds the protocol's {limit}-byte line limit")


def is_request_id(value: object) -> bool:
    """True for a valid request id: a JSON integer (``bool`` is not one)."""
    return isinstance(value, int) and not isinstance(value, bool)
