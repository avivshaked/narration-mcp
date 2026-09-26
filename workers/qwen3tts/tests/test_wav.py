"""The worker's WAV writer: byte-reproducible float32 mono files (design Appendix A, section 10.1)."""

from __future__ import annotations

import struct
from array import array
from pathlib import Path

import numpy as np
import pytest
from narration_qwen3tts.wav import float32_mono_wav_bytes, write_float32_mono


def _signal(n: int = 4801) -> np.ndarray:
    return (0.5 * np.sin(np.arange(n, dtype=np.float64) * 0.01)).astype(np.float32)


def _chunk_ids(data: bytes) -> list[bytes]:
    ids, at = [], 12
    while at < len(data):
        size = struct.unpack_from("<I", data, at + 4)[0]
        ids.append(data[at : at + 4])
        at += 8 + size + (size & 1)
    return ids


def test_the_same_samples_give_the_same_bytes_s10_1(tmp_path: Path) -> None:
    """A render's sha256 is over the file, so a repeat of the same audio must repeat every byte (bit_exact)."""
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    write_float32_mono(first, _signal(), 24000)
    write_float32_mono(second, _signal(), 24000)
    assert first.read_bytes() == second.read_bytes()


def test_the_file_holds_only_fmt_fact_and_data_s10_1() -> None:
    """No PEAK chunk: libsndfile's holds the time of writing, which made identical renders differ."""
    data = float32_mono_wav_bytes(_signal(), 24000)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    assert _chunk_ids(data) == [b"fmt ", b"fact", b"data"]


def test_the_layout_matches_the_fake_worker_appA(tmp_path: Path) -> None:
    """Every worker's audio replies are the same kind of file (the protocol's float32 mono WAV)."""
    fake_wav = pytest.importorskip("narration_worker.fake.wav")
    signal = _signal()
    mine, theirs = tmp_path / "mine.wav", tmp_path / "theirs.wav"
    write_float32_mono(mine, signal, 24000)
    fake_wav.write_float32_mono(theirs, array("f", signal.tolist()), 24000)
    assert mine.read_bytes() == theirs.read_bytes()


def test_soundfile_reads_the_samples_back_exactly(tmp_path: Path) -> None:
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "x.wav"
    signal = _signal()
    write_float32_mono(path, signal, 24000)
    audio, rate = sf.read(str(path), dtype="float32")
    assert rate == 24000
    assert np.array_equal(audio, signal)


def test_nan_and_infinity_are_written_as_they_are_s11_1() -> None:
    """The raw file is never cleaned: SIGNAL_INVALID (DC-5) is the server's verdict on the raw take."""
    signal = np.array([np.nan, np.inf, -np.inf, 0.5, -0.0], dtype=np.float32)
    data = float32_mono_wav_bytes(signal, 24000)
    stored = np.frombuffer(data[-signal.size * 4 :], dtype="<f4")
    assert stored.tobytes() == signal.astype("<f4").tobytes()


def test_float64_input_is_stored_as_float32() -> None:
    signal = _signal()
    assert float32_mono_wav_bytes(signal.astype(np.float64), 24000) == float32_mono_wav_bytes(signal, 24000)


def test_the_write_leaves_no_temp_file(tmp_path: Path) -> None:
    write_float32_mono(tmp_path / "x.wav", _signal(), 24000)
    assert [p.name for p in tmp_path.iterdir()] == ["x.wav"]


@pytest.mark.parametrize(
    ("audio", "rate"),
    [(np.zeros((2, 10), dtype=np.float32), 24000), (np.zeros(10, dtype=np.float32), 0)],
)
def test_bad_input_is_refused(audio: np.ndarray, rate: int) -> None:
    with pytest.raises(ValueError):
        float32_mono_wav_bytes(audio, rate)
