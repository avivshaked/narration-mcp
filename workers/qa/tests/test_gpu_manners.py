"""The QA worker returns the CUDA allocator's cached blocks after each GPU op (``narration_worker_qa.gpu``; design
section 4, the GPU scheduler). Pure: a stand-in records the calls, so no torch and no GPU are needed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from narration_worker_qa.gpu import release_cached_memory


def _torch() -> tuple[SimpleNamespace, list[str]]:
    calls: list[str] = []
    return SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))), calls


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_a_gpu_op_returns_the_cached_blocks_s4(device: str) -> None:
    torch, calls = _torch()
    release_cached_memory(torch, device)
    assert calls == ["empty_cache"]


def test_a_cpu_op_leaves_cuda_alone_s4() -> None:
    torch, calls = _torch()
    release_cached_memory(torch, "cpu")
    assert calls == []
