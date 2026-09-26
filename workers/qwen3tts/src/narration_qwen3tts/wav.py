"""Byte-reproducible float32 mono WAV files for the worker's audio replies (design Appendix A, section 10.1).

The same audio must give the same file, byte for byte. The render's sha256 is taken over the file, so the
``bit_exact`` tier and the canary gate's hash comparison depend on it.

libsndfile (``soundfile.write``) cannot be used for this. For a float WAV it adds a ``PEAK`` chunk that
holds the time of writing, so two writes of identical samples differ in their header (KNOW, 2026-09-26: the
GPU test's two renders of the canary gate had identical samples and different bytes, at the ``PEAK``
chunk's timestamp).

This writer produces the layout of ``narration_worker.fake.wav.write_float32_mono``:
``WAVE_FORMAT_IEEE_FLOAT`` with ``cbSize`` 0, a ``fact`` chunk and the ``data`` chunk, little-endian, and
nothing else.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, Final

import numpy as np

FORMAT_FLOAT: Final = 3


def float32_mono_wav_bytes(audio: Any, sample_rate: int) -> bytes:
    """The complete WAV file for a mono float32 signal, as bytes. It holds nothing but the audio."""
    samples = np.ascontiguousarray(audio, dtype="<f4")
    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (one dimension), got shape {samples.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    payload = samples.tobytes()
    fmt = struct.pack("<HHIIHHH", FORMAT_FLOAT, 1, sample_rate, sample_rate * 4, 4, 32, 0)
    fact = struct.pack("<I", samples.size)
    body = b"WAVE" + _chunk(b"fmt ", fmt) + _chunk(b"fact", fact) + _chunk(b"data", payload)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write_float32_mono(path: Path, audio: Any, sample_rate: int) -> None:
    """Write ``audio`` as a float32 mono WAV to a temporary name beside ``path``, then rename it into place."""
    data = float32_mono_wav_bytes(audio, sample_rate)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) & 1 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + pad
