"""JSON-lines framing (design Appendix A): one UTF-8 JSON object per line, strict JSON, bounded lines."""

from __future__ import annotations

import io

import pytest
from narration_worker.framing import (
    FramingError,
    LineTooLong,
    decode_message,
    encode_message,
    is_request_id,
    read_line,
)


def test_a_message_is_one_utf8_json_line_appA() -> None:
    data = encode_message({"id": 1, "ok": True, "text": "naïve\nline — “quoted”"})
    assert data.endswith(b"\n")
    assert data.count(b"\n") == 1
    assert "naïve".encode() in data
    assert decode_message(data) == {"id": 1, "ok": True, "text": "naïve\nline — “quoted”"}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nan_and_infinity_are_not_json_appA(value: float) -> None:
    with pytest.raises(ValueError):
        encode_message({"id": 1, "x": value})


@pytest.mark.parametrize(
    "line",
    [b"not json", b"[1, 2]", b'"text"', b"\xff\xfe{}", b'{"id": 1, "x": NaN}', b'{"id": 1, "x": Infinity}', b"{"],
)
def test_a_line_that_is_not_one_json_object_is_refused_appA(line: bytes) -> None:
    with pytest.raises(FramingError):
        decode_message(line)


def test_an_over_long_line_is_discarded_and_the_next_line_is_read_appA() -> None:
    stream = io.BytesIO(b"x" * 50 + b"\n" + b'{"id":2}\n')
    with pytest.raises(LineTooLong):
        read_line(stream, limit=20)
    assert read_line(stream, limit=20) == b'{"id":2}'
    assert read_line(stream, limit=20) is None


def test_read_line_returns_lines_without_newline_and_none_at_the_end_appA() -> None:
    stream = io.BytesIO(b'{"id":1}\n\n{"id":2}')
    assert read_line(stream) == b'{"id":1}'
    assert read_line(stream) == b""
    assert read_line(stream) == b'{"id":2}'
    assert read_line(stream) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, True), (-3, True), (2**40, True), (True, False), ("1", False), (1.0, False), (None, False)],
)
def test_a_request_id_is_a_json_integer_appA(value: object, expected: bool) -> None:
    assert is_request_id(value) is expected
