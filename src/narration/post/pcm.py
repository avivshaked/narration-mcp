"""Audio file I/O for post-processing: reading raw takes, quantising, and writing WAV atomically.

Quantisation is done here, not by libsndfile, so the delivery's samples do not depend on the library's
float-to-integer conversion: ``q = clip(rint(y * 2**(bits-1)), -2**(bits-1), 2**(bits-1) - 1)``, rounding
half to even. soundfile then only packs the integers: they are handed over left-aligned in 32 bits, and
libsndfile keeps the top ``bits`` exactly (KNOW: a PCM_24 round trip of every 24-bit extreme is exact, and
the file is a canonical 44-byte-header WAV). Reading a file back as float gives ``q / 2**(bits-1)``.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import soundfile

BITS: Final[dict[str, int]] = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
"""The delivery subtypes supported, with their bit depths."""

Audio = npt.NDArray[np.float64]
Pcm = npt.NDArray[np.int64]


def bits_of(subtype: str) -> int:
    """The bit depth of a supported delivery subtype; ValueError for any other."""
    try:
        return BITS[subtype]
    except KeyError:
        supported = ", ".join(BITS)
        raise ValueError(f"delivery subtype {subtype!r} is not supported; use one of {supported}") from None


def quantise(y: Audio, bits: int) -> Pcm:
    """Round a float signal (full scale 1.0) to ``bits``-bit integers, clipping at the integer range."""
    scale = float(1 << (bits - 1))
    return np.clip(np.rint(y * scale), -scale, scale - 1.0).astype(np.int64)


def to_float(q: Pcm, bits: int) -> Audio:
    """The float signal a reader sees: ``q / 2**(bits-1)``."""
    return q.astype(np.float64) / float(1 << (bits - 1))


def lsb_dbfs(bits: int) -> float:
    """The level of one least significant bit, in dB relative to full scale."""
    return -20.0 * (bits - 1) * float(np.log10(2.0))


def encode_wav(q: Pcm, sample_rate: int, subtype: str) -> bytes:
    """The bytes of a mono WAV file holding ``q`` at ``subtype``."""
    bits = bits_of(subtype)
    data = (q << (32 - bits)).astype(np.int32)
    buffer = io.BytesIO()
    soundfile.write(buffer, data, sample_rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


def read_mono(path: Path) -> tuple[Audio, int]:
    """Read a mono audio file as float64 (full scale 1.0); ValueError if it has more than one channel."""
    data, sample_rate = soundfile.read(str(path), dtype="float64", always_2d=True)
    channels = int(data.shape[1])
    if channels != 1:
        raise ValueError(f"{path.name}: expected mono audio, found {channels} channels")
    return np.ascontiguousarray(data[:, 0], dtype=np.float64), int(sample_rate)


def decode_wav(data: bytes) -> tuple[Audio, int]:
    """Read mono WAV bytes as float64 (full scale 1.0)."""
    samples, sample_rate = soundfile.read(io.BytesIO(data), dtype="float64", always_2d=True)
    if int(samples.shape[1]) != 1:
        raise ValueError(f"expected mono audio, found {samples.shape[1]} channels")
    return np.ascontiguousarray(samples[:, 0], dtype=np.float64), int(sample_rate)


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to a temporary file beside ``path``, then rename it into place (never half written)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
