"""The fake worker keeps the same contract as the real workers (plan.md WP16 acceptance)."""

from __future__ import annotations

from typing import Any

import pytest
from narration_worker.testing.contract import WorkerContract


class TestFakeWorkerContract(WorkerContract):
    role = "fake"
    timeout_s = 30.0

    @pytest.fixture
    def load_request(self) -> dict[str, Any] | None:
        return {"device": "cpu"}
