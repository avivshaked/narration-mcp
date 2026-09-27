"""The worker writes its raw renders with the workers' shared writer (design Appendix A, section 10.1).

The writer itself, byte-reproducible float32 mono with no ``PEAK`` chunk (ADR 0002, decision 5), is
``narration_worker.wav`` and is tested there (``workers/common/tests/test_wav.py``). What is this worker's own
is the conversion: whatever the engine returns is written as float32 samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import narration_qwen3tts.worker as worker_module
import narration_worker.wav as shared
import numpy as np
import pytest
from narration_qwen3tts.engine import Rendered


def _rendered(audio: np.ndarray) -> Rendered:
    return Rendered(
        audio=audio, sample_rate=24000, new_tokens=3, hit_token_cap=False, talker_steps=4, gen_s=0.1, max_new_tokens=8
    )


def _signal(n: int = 4801) -> np.ndarray:
    return (0.5 * np.sin(np.arange(n, dtype=np.float64) * 0.01)).astype(np.float32)


def test_the_worker_writes_with_the_shared_writer_s10_1(tmp_path: Path) -> None:
    """One writer for every worker, so every worker's identical audio is an identical file."""
    assert vars(worker_module)["write_float32_mono"] is shared.write_float32_mono  # the name the worker imported
    out = tmp_path / "raw.wav"
    reply = worker_module._write(out, _rendered(_signal()))
    assert out.read_bytes() == shared.float32_mono_wav_bytes(_signal(), 24000)
    assert (reply["samples"], reply["sample_rate"]) == (4801, 24000)


def test_float64_audio_is_written_as_float32(tmp_path: Path) -> None:
    out = tmp_path / "raw.wav"
    worker_module._write(out, _rendered(_signal().astype(np.float64)))
    assert out.read_bytes() == shared.float32_mono_wav_bytes(_signal(), 24000)


def test_a_numpy_integer_rate_is_a_rate_and_a_numpy_float_is_not() -> None:
    """The library returns its rate as it likes; the writer takes any integer, never a float."""
    assert shared.float32_mono_wav_bytes(_signal(), np.int64(24000)) == shared.float32_mono_wav_bytes(_signal(), 24000)
    with pytest.raises(ValueError):
        shared.float32_mono_wav_bytes(_signal(), cast(Any, np.float64(24000.0)))


def test_audio_that_is_not_mono_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        worker_module._write(tmp_path / "raw.wav", _rendered(np.zeros((2, 10), dtype=np.float32)))
    assert not list(tmp_path.iterdir())
