"""The fake worker keeps the same contract as the real workers (plan.md WP16 acceptance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from narration_worker.testing.contract import WorkerContract

TEXT = "Some of them thrive, and some of them simply disappear."
VOICE = "sha256:" + "5" * 64


class TestFakeWorkerContract(WorkerContract):
    role = "fake"
    timeout_s = 30.0

    @pytest.fixture
    def load_request(self) -> dict[str, Any] | None:
        return {"device": "cpu"}

    @pytest.fixture(params=["synthesize", "design"])
    def render_requests(self, request: pytest.FixtureRequest, store_root: Path) -> list[tuple[str, dict[str, Any]]]:
        """The capped call is a ``synthesize`` (after a design and a prepare_voice) or a ``design``."""
        scratch = store_root / "scratch" / "contract"
        design = {"description": "A calm voice.", "design_text": TEXT, "language": "English", "seed": 1}
        if request.param == "design":
            return [("design", {**design, "out_path": str(scratch / "designed.wav")})]
        clip = str(scratch / "clip.wav")
        synthesize = {"voice_hash": VOICE, "engine_text": TEXT, "language": "English", "seed": 2}
        return [
            ("design", {**design, "max_new_tokens": 8192, "out_path": clip}),
            ("prepare_voice", {"voice_hash": VOICE, "ref_wav": clip, "ref_text": TEXT, "x_vector_only_mode": False}),
            ("synthesize", {**synthesize, "out_path": str(scratch / "take.wav")}),
        ]
