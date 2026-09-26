"""Audio file I/O for post-processing: reading raw takes, quantising, and writing WAV atomically.

**The delivery's bytes are made here**, not by a library, so no library version can change them:

- quantisation: ``q = clip(rint(y * 2**(bits-1)), -2**(bits-1), 2**(bits-1) - 1)``, rounding half to even;
- the file: the canonical 44-byte WAV header (``RIFF``/``WAVE``, a 16-byte ``fmt `` chunk with format tag 1,
  integer PCM, one channel; then ``data``), followed by the samples as little-endian two's complement.
  A data chunk of odd length is followed by one zero pad byte, as RIFF requires.

soundfile is used only to read audio. Reading a delivery back as float gives ``q / 2**(bits-1)``.
"""

from __future__ import annotations

import contextlib
import io
import os
import struct
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import soundfile

BITS: Final[dict[str, int]] = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
"""The delivery subtypes supported, with their bit depths."""
WAVE_FORMAT_PCM: Final = 1
HEADER_BYTES: Final = 44
MAX_DATA_BYTES: Final = 0xFFFFFFFF - HEADER_BYTES

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


def wav_header(samples: int, sample_rate: int, bits: int) -> bytes:
    """The canonical 44-byte header of a mono integer-PCM WAV file with ``samples`` samples."""
    width = bits // 8
    data = samples * width
    if data > MAX_DATA_BYTES:
        raise ValueError(f"{samples} samples do not fit in a WAV file")
    riff = 4 + (8 + 16) + (8 + data + (data & 1))
    return (
        b"RIFF"
        + struct.pack("<I", riff)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, WAVE_FORMAT_PCM, 1, sample_rate, sample_rate * width, width, bits)
        + b"data"
        + struct.pack("<I", data)
    )


def encode_wav(q: Pcm, sample_rate: int, subtype: str) -> bytes:
    """The bytes of a mono WAV file holding ``q`` at ``subtype`` (header, samples, pad byte if odd)."""
    bits = bits_of(subtype)
    width = bits // 8
    words = q.astype("<i4")
    if width == 3:
        body = words.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    elif width == 2:
        body = q.astype("<i2").tobytes()
    else:
        body = words.tobytes()
    pad = b"\x00" if len(body) & 1 else b""
    return wav_header(int(q.shape[0]), sample_rate, bits) + body + pad


def read_mono(path: Path) -> tuple[Audio, int]:
    """Read a mono audio file as float64 (full scale 1.0); ValueError if it has more than one channel."""
    data, sample_rate = soundfile.read(str(path), dtype="float64", always_2d=True)
    channels = int(data.shape[1])
    if channels != 1:
        raise ValueError(f"{path.name}: expected mono audio, found {channels} channels")
    return np.ascontiguousarray(data[:, 0], dtype=np.float64), int(sample_rate)


def decode_wav(data: bytes) -> tuple[Audio, int]:
    """Read mono WAV bytes as float64 (full scale 1.0), with soundfile."""
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
