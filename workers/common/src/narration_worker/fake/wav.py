"""Reading WAV files with the standard library alone, for the fake worker.

Reads PCM (16, 24 or 32 bit) and float (32 or 64 bit), plain or ``WAVE_FORMAT_EXTENSIBLE``, and returns the
first channel as floats. That covers what the fake reads: its own renders and the delivery files
post-processing makes from them (48 kHz PCM_24). The fake writes its renders with the workers' shared writer,
``narration_worker.wav``, as every worker does.
"""

from __future__ import annotations

import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from narration_worker.wav import FORMAT_FLOAT

FORMAT_PCM: Final = 1
FORMAT_EXTENSIBLE: Final = 0xFFFE


class WavError(ValueError):
    """Not a WAV file this reader understands."""


@dataclass(frozen=True, slots=True)
class Audio:
    """One channel of a WAV file. ``samples`` are floats, full scale ±1 for PCM."""

    sample_rate: int
    samples: array[float]

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def read_wav(path: Path) -> Audio:
    """Read the first channel of a WAV file; raises ``WavError`` for anything else."""
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise WavError(f"cannot read {path.name}: {exc}") from exc
    if len(blob) < 12 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise WavError(f"{path.name} is not a RIFF/WAVE file")
    fmt: tuple[int, int, int, int] | None = None
    data: bytes | None = None
    pos = 12
    while pos + 8 <= len(blob):
        chunk_id = blob[pos : pos + 4]
        (size,) = struct.unpack_from("<I", blob, pos + 4)
        start = pos + 8
        end = min(start + size, len(blob))
        if chunk_id == b"fmt ":
            fmt = _parse_fmt(blob[start:end])
        elif chunk_id == b"data":
            data = blob[start:end]
            break
        pos = start + size + (size & 1)
    if fmt is None or data is None:
        raise WavError(f"{path.name} has no fmt or data chunk")
    audio_format, channels, sample_rate, bits = fmt
    samples = _decode(data, audio_format, bits, channels)
    return Audio(sample_rate=sample_rate, samples=samples)


def _parse_fmt(raw: bytes) -> tuple[int, int, int, int]:
    if len(raw) < 16:
        raise WavError("the fmt chunk is too short")
    audio_format, channels, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", raw)
    if audio_format == FORMAT_EXTENSIBLE:
        if len(raw) < 26:
            raise WavError("the extensible fmt chunk is too short")
        (audio_format,) = struct.unpack_from("<H", raw, 24)
    if channels < 1 or sample_rate < 1:
        raise WavError("the fmt chunk has no channels or no sample rate")
    return audio_format, channels, sample_rate, bits


def _decode(data: bytes, audio_format: int, bits: int, channels: int) -> array[float]:
    width = bits // 8
    usable = len(data) - len(data) % (width * channels)
    raw = data[:usable]
    if audio_format == FORMAT_FLOAT and bits in (32, 64):
        values = array("f" if bits == 32 else "d")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return array("d", values[::channels])
    if audio_format != FORMAT_PCM or bits not in (16, 24, 32):
        raise WavError(f"unsupported WAV encoding: format {audio_format}, {bits} bits")
    if bits == 24:
        widened = bytearray(len(raw) // 3 * 4)
        widened[1::4] = raw[0::3]
        widened[2::4] = raw[1::3]
        widened[3::4] = raw[2::3]
        ints = array("i")
        ints.frombytes(bytes(widened))
        scale = 1.0 / 2**31
    else:
        ints = array("h" if bits == 16 else "i")
        ints.frombytes(raw)
        scale = 1.0 / 2 ** (bits - 1)
    if sys.byteorder != "little":
        ints.byteswap()
    mono = ints[::channels]
    return array("d", (v * scale for v in mono))
