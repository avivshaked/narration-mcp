"""The QA worker returns the CUDA allocator's cached blocks after each GPU op (``narration_worker_qa.gpu``; design
section 4, the GPU scheduler). Pure: a stand-in records the calls, so no torch and no GPU are needed."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from narration_worker.handler import WorkerContext, WorkerHandler
from narration_worker_qa.asr import WhisperAsr
from narration_worker_qa.gpu import release_cached_memory


def _torch() -> tuple[SimpleNamespace, list[str]]:
    calls: list[str] = []
    return SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))), calls


def _failing_torch() -> SimpleNamespace:
    def empty_cache() -> None:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    return SimpleNamespace(cuda=SimpleNamespace(empty_cache=empty_cache))


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_a_gpu_op_returns_the_cached_blocks_s4(device: str) -> None:
    torch, calls = _torch()
    release_cached_memory(torch, device)
    assert calls == ["empty_cache"]


def test_a_cpu_op_leaves_cuda_alone_s4() -> None:
    torch, calls = _torch()
    release_cached_memory(torch, "cpu")
    assert calls == []


def test_a_release_that_fails_is_logged_never_raised_s4(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="narration_worker_qa.gpu"):
        release_cached_memory(_failing_torch(), "cuda:0")
    assert "could not return the CUDA allocator's cached blocks" in caplog.text


class _Handler(WorkerHandler):
    role = "qa"
    uses_torch = False


def test_an_ops_gpu_oom_survives_a_release_that_fails_s4(tmp_path: Path) -> None:
    # Regression (WP22 review): the release in the op's ``finally`` raised in turn, so an op that ran out of
    # GPU memory was reported INTERNAL, and the daemon did not unload, wait and retry.
    asr = WhisperAsr(_failing_torch())

    def out_of_memory(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    asr._pipeline = out_of_memory  # pyright: ignore[reportPrivateUsage]
    asr._generation = {}  # pyright: ignore[reportPrivateUsage]
    asr.device = "cuda:0"
    with pytest.raises(RuntimeError, match="CUDA out of memory") as caught:
        asr.transcribe(np.zeros(16_000, dtype=np.float32), "English", word_timestamps=False, long_form=True)
    error = _Handler(WorkerContext(role="qa", store_root=tmp_path, cpu_threads=1)).classify(caught.value)
    assert error is not None and error.code == "GPU_OOM"
