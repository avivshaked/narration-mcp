"""The shared worker contract (design Appendix A; plan.md WP16), run against the ``qwen3`` role.

It starts ``python -m narration_worker --role qwen3`` in this venv, which finds ``Qwen3Handler`` through the
entry point this project registers. Nothing here loads a model: ``tests/test_gpu.py`` runs the same contract
with VoiceDesign loaded (markers ``gpu``, ``model`` and ``slow``). Skipped until WP16's contract suite
(``narration_worker.testing``) is in this checkout.
"""

from __future__ import annotations

import pytest

contract = pytest.importorskip(
    "narration_worker.testing.contract", reason="needs WP16's narration_worker.testing (the contract suite)"
)


class TestQwen3WorkerContract(contract.WorkerContract):
    role = "qwen3"
