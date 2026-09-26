"""The workers' shared WAV writer: byte-reproducible float32 mono files (design Appendix A, section 10.1;
ADR 0002, decision 5). Moved here from the qwen3 worker's tests when the two writers became one."""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from array import array
from pathlib import Path
from typing import Any

import pytest
from narration_worker.wav import float32_mono_wav_bytes, write_float32_mono

GOLDEN = array("f", [0.0, 0.25, -0.5, 1.0, -1.0, 0.125, -0.0, 0.75])
GOLDEN_SHA256 = "fc81e9625d6dbd73391c521132c935901bbfe0a54ad8c50613901c7bdac81dd9"
"""``GOLDEN`` at 24 kHz, as both earlier writers (the fake's and the qwen3 worker's) wrote it, byte for byte."""


def _signal(n: int = 4801) -> array[float]:
    return array("f", (0.5 * math.sin(i * 0.01) for i in range(n)))


def _chunk_ids(data: bytes) -> list[bytes]:
    ids, at = [], 12
    while at < len(data):
        size = struct.unpack_from("<I", data, at + 4)[0]
        ids.append(data[at : at + 4])
        at += 8 + size + (size & 1)
    return ids


def test_the_layout_is_pinned_to_the_byte_s10_1() -> None:
    """Every take id and canary hash rests on this layout: change it only with a new engine profile."""
    assert hashlib.sha256(float32_mono_wav_bytes(GOLDEN, 24000)).hexdigest() == GOLDEN_SHA256


def test_the_same_samples_give_the_same_bytes_s10_1(tmp_path: Path) -> None:
    """A render's sha256 is over the file, so a repeat of the same audio must repeat every byte (bit_exact)."""
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    write_float32_mono(first, _signal(), 24000)
    write_float32_mono(second, _signal(), 24000)
    assert first.read_bytes() == second.read_bytes() == float32_mono_wav_bytes(_signal(), 24000)


def test_the_file_holds_only_fmt_fact_and_data_s10_1() -> None:
    """No PEAK chunk: libsndfile's holds the time of writing, which made identical renders differ."""
    data = float32_mono_wav_bytes(_signal(), 24000)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    assert _chunk_ids(data) == [b"fmt ", b"fact", b"data"]
    audio_format, channels, rate, byte_rate, align, bits, extra = struct.unpack_from("<HHIIHHH", data, 20)
    assert (audio_format, channels, rate, byte_rate, align, bits, extra) == (3, 1, 24000, 96000, 4, 32, 0)
    assert struct.unpack_from("<I", data, 46)[0] == 4801  # the fact chunk's sample count


def test_nan_and_infinity_are_written_as_they_are_s11_1() -> None:
    """The raw file is never cleaned: SIGNAL_INVALID (DC-5) is the server's verdict on the raw take."""
    # NaN, a NaN with a payload, +inf, -inf, -0.0 and the smallest denormal, as little-endian bit patterns
    specials = struct.pack("<6I", 0x7FC00000, 0x7FC00001, 0x7F800000, 0xFF800000, 0x80000000, 0x00000001)
    signal = array("f")
    signal.frombytes(specials)
    if sys.byteorder == "big":
        signal.byteswap()
    data = float32_mono_wav_bytes(signal, 24000)
    assert data[-len(specials) :] == specials


def test_soundfile_reads_the_samples_back_exactly(tmp_path: Path) -> None:
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "x.wav"
    write_float32_mono(path, _signal(), 24000)
    audio, rate = sf.read(str(path), dtype="float32")
    assert rate == 24000
    assert audio.tobytes() == _signal().tobytes()


def test_numpy_float32_in_either_byte_order_gives_the_same_bytes_appA() -> None:
    """A worker may hand over a numpy array (the qwen3 worker does): the file is the same as for array('f')."""
    np = pytest.importorskip("numpy")
    expected = float32_mono_wav_bytes(_signal(), 24000)
    little = np.array(_signal(), dtype="<f4")
    assert float32_mono_wav_bytes(little, 24000) == expected
    assert float32_mono_wav_bytes(little.astype(">f4"), 24000) == expected
    assert float32_mono_wav_bytes(np.repeat(little, 2)[::2], 24000) == expected  # not contiguous


def test_the_write_leaves_no_temp_file(tmp_path: Path) -> None:
    write_float32_mono(tmp_path / "x.wav", _signal(), 24000)
    assert [p.name for p in tmp_path.iterdir()] == ["x.wav"]


@pytest.mark.parametrize(
    ("samples", "rate"),
    [
        (array("d", [0.0] * 10), 24000),  # float64: a worker converts it itself
        (memoryview(bytes(40)).cast("f", shape=[2, 5]), 24000),  # two channels
        ([0.0] * 10, 24000),  # a list is not a buffer
        (array("f", [0.0] * 10), 0),
        (array("f", [0.0] * 10), -24000),
        (array("f", [0.0] * 10), 24000.0),
        (array("f", [0.0] * 10), True),
        (array("f", [0.0] * 10), "24000"),
        (array("f", [0.0] * 10), 2**30),  # four times it does not fit the header's byte rate
    ],
)
def test_bad_input_is_refused(samples: Any, rate: Any) -> None:
    """Every bad input is a ``ValueError``, as the writer's docstring says, never a ``TypeError`` or a
    ``struct.error``."""
    with pytest.raises(ValueError):
        float32_mono_wav_bytes(samples, rate)
