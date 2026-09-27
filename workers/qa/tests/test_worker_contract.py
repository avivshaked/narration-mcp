"""WP16's shared worker contract, run against the real ``qa`` worker (``python -m narration_worker --role qa``).

The ``load_request`` loads tiny, random snapshots of the three models on the CPU (``conftest.tiny_snapshot``),
so the contract's load round trip runs without the real models and without a GPU. ``test_gpu.py`` runs it again
with the real models on the GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

contract = pytest.importorskip("narration_worker.testing.contract", reason="needs WP16's narration_worker.testing")

REVISIONS = {"asr": "a" * 40, "sv": "b" * 40, "aligner": "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"}


class TestQaWorkerContract(contract.WorkerContract):
    role = "qa"

    @pytest.fixture
    def load_request(self, tiny_snapshot: Callable[..., Path]) -> dict[str, Any]:
        pytest.importorskip("torch", reason="the tiny snapshots need torch and transformers")
        pytest.importorskip("transformers", reason="the tiny snapshots need torch and transformers")
        models = {
            use: {"repo": f"example/{use}", "revision": REVISIONS[use], "snapshot_dir": str(tiny_snapshot(use))}
            for use in ("asr", "sv", "aligner")
        }
        return {"device": "cpu", "models": models}
