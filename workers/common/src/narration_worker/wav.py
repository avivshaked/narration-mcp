"""Byte-reproducible float32 mono WAV files: the one writer of every worker's audio (design Appendix A,
section 10.1; ADR 0002, decision 5).

The same samples always give the same file, byte for byte, so a render's sha256 is a fact about its audio.
The ``bit_exact`` tier's stable take ids and the canary gate's hash comparison depend on it.

- **Layout:** ``RIFF``/``WAVE`` with a ``fmt`` chunk (``WAVE_FORMAT_IEEE_FLOAT``, one channel, 32 bits,
  ``cbSize`` 0), a ``fact`` chunk (the sample count) and the ``data`` chunk, all little-endian, and nothing
  else: no ``PEAK`` chunk and no timestamp.
- **Why not libsndfile:** ``soundfile.write`` adds a ``PEAK`` chunk to a float WAV, holding the time of
  writing, so two writes of identical samples differ (KNOW; ADR 0002).
- **Samples as given:** NaN and infinity are written unchanged. A raw take is never cleaned, because
  ``SIGNAL_INVALID`` is the server's verdict on it (section 11.1, DC-5).
- **Whole files only:** a file is written to a temporary name beside its path and then renamed into place.

Standard library only, so every worker venv carries it. ``samples`` is any one-dimensional buffer of 32-bit
floats, in either byte order: an ``array('f')``, or a numpy ``float32`` array (a worker converts other dtypes
itself, so the conversion is its own, visible choice). ``sample_rate`` is any integer, numpy's included.
"""

from __future__ import annotations

import operator
import os
import struct
import sys
from array import array
from collections.abc import Buffer
from pathlib import Path
from typing import Final, SupportsIndex

FORMAT_FLOAT: Final = 3
"""``WAVE_FORMAT_IEEE_FLOAT``."""
_NATIVE_FLOAT32: Final = ("f", "@f", "=f")
"""``memoryview`` formats of 32-bit floats in the machine's byte order (``<f`` and ``>f`` name theirs)."""
_MAX_SAMPLE_RATE: Final = 0xFFFFFFFF // 4
"""The largest rate whose byte rate (four bytes a sample) fits the ``fmt`` chunk's 32-bit field."""


def float32_mono_wav_bytes(samples: Buffer, sample_rate: SupportsIndex) -> bytes:
    """The complete WAV file for a mono float32 signal, as bytes. It holds nothing but the audio.

    Raises ``ValueError`` for samples that are not a one-dimensional buffer of 32-bit floats (a list is not
    a buffer), or a sample rate that is not a positive integer the header can hold (a float or a ``bool`` is
    not one).
    """
    payload, count = _little_endian_float32(samples)
    rate = _sample_rate(sample_rate)
    fmt = struct.pack("<HHIIHHH", FORMAT_FLOAT, 1, rate, rate * 4, 4, 32, 0)
    fact = struct.pack("<I", count)
    body = b"WAVE" + _chunk(b"fmt ", fmt) + _chunk(b"fact", fact) + _chunk(b"data", payload)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write_float32_mono(path: Path, samples: Buffer, sample_rate: SupportsIndex) -> None:
    """Write ``samples`` as a float32 mono WAV (``float32_mono_wav_bytes``) to a temporary name beside
    ``path``, then rename it into place, so the file is never seen half written."""
    data = float32_mono_wav_bytes(samples, sample_rate)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _sample_rate(value: SupportsIndex) -> int:
    """``value`` as the header's sample rate: an integer (numpy's too) from 1 to what ``byte rate`` (four
    times it, in 32 bits) can hold; never a ``bool``, a float or a string."""
    if isinstance(value, bool):
        raise ValueError(f"sample_rate must be a positive integer, not {value!r}")
    try:
        rate = operator.index(value)
    except TypeError:
        raise ValueError(f"sample_rate must be a positive integer, not {value!r}") from None
    if not 0 < rate <= _MAX_SAMPLE_RATE:
        raise ValueError(f"sample_rate must be from 1 to {_MAX_SAMPLE_RATE}, got {rate}")
    return rate


def _little_endian_float32(samples: Buffer) -> tuple[bytes, int]:
    """The samples as little-endian float32 bytes, and how many there are."""
    try:
        view = memoryview(samples)
    except TypeError:
        kind = type(samples).__name__
        raise ValueError(f"samples must be a buffer of 32-bit floats (array('f'), numpy float32), not {kind}") from None
    with view:
        if view.ndim != 1:
            raise ValueError(f"expected mono audio (one dimension), got {view.ndim} dimensions")
        if view.itemsize != 4 or view.format not in (*_NATIVE_FLOAT32, "<f", ">f"):
            raise ValueError(f"samples must be 32-bit floats, not buffer format {view.format!r}")
        payload = view.tobytes()
        count = view.shape[0] if view.shape else 0
        big_endian = view.format == ">f" or (view.format in _NATIVE_FLOAT32 and sys.byteorder == "big")
    if big_endian:
        swapped = array("f")
        swapped.frombytes(payload)
        swapped.byteswap()
        payload = swapped.tobytes()
    return payload, count


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) & 1 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + pad
