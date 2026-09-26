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

    def test_load_checks_snapshot_references_as_the_other_workers_do_s4(self, worker: Any, store_root: Path) -> None:
        """The order, codes and ``details.field`` of the fake and the ``qwen3`` worker (WP16's review): a relative
        folder, then a missing one before anything else, then a revision that is not 40-hex or does not name its
        folder."""
        folder = store_root / "snapshots" / ("0" * 40)
        folder.mkdir(parents=True)
        ref = {"repo": "example/asr", "revision": "0" * 40, "snapshot_dir": str(folder)}
        cases = [
            ({"asr": ref | {"snapshot_dir": "relative"}}, "INVALID_REQUEST", "models.asr.snapshot_dir"),
            (
                {"asr": ref | {"revision": "x", "snapshot_dir": str(store_root / "none")}},
                "BACKEND_NOT_INSTALLED",
                "models.asr",
            ),
            ({"asr": ref | {"revision": "1" * 40}}, "INVALID_REQUEST", "models.asr.snapshot_dir"),
            ({"asr": ref | {"revision": "main"}}, "INVALID_REQUEST", "models.asr.revision"),
        ]
        for models, code, field in cases:
            reply = worker.request("load", timeout_s=self.timeout_s, device="cpu", models=models)
            assert reply["ok"] is False and reply["error"]["code"] == code, (models, reply)
            assert reply["error"].get("details", {}).get("field") == field, (models, reply)
