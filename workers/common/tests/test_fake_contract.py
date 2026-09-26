"""The fake worker keeps the same contract as the real workers (plan.md WP16 acceptance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from narration_worker.fake.handler import FAKE_REVISION
from narration_worker.testing.client import WorkerProcess
from narration_worker.testing.contract import DETERMINISM, WorkerContract

TEXT = "When the loaves come out golden and crisp."
VOICE = "sha256:" + "5" * 64


class TestFakeWorkerContract(WorkerContract):
    role = "fake"
    timeout_s = 30.0

    @pytest.fixture
    def load_request(self, store_root: Path) -> dict[str, Any] | None:
        """A complete Qwen load, as the daemon sends one: the fake checks it as the qwen3 worker does."""
        snapshot = store_root.parent / "models" / FAKE_REVISION
        snapshot.mkdir(parents=True, exist_ok=True)
        return {
            "device": "cpu",
            "model": {"repo": "narration-worker/fake", "revision": FAKE_REVISION, "snapshot_dir": str(snapshot)},
            "engine_profile_id": "fake-contract",
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "determinism": dict(DETERMINISM),
            "settings": {"non_streaming_mode": False, "generation": {"max_new_tokens": 8192}},
        }

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


class _RendersWithoutRenderRequests(WorkerContract):
    """A real Qwen worker's contract class that forgot ``render_requests`` (not collected: no ``Test``)."""

    role = "qwen3"


class _QaContract(WorkerContract):
    role = "qa"


def test_a_rendering_role_without_render_requests_fails_the_cap_tests_rather_than_skipping_s10_1() -> None:
    no_worker = cast(WorkerProcess, None)  # never reached: the checks come first
    with pytest.raises(pytest.fail.Exception, match="override render_requests"):
        _RendersWithoutRenderRequests().render_at_ceiling(no_worker, {"device": "cuda:0"}, None)
    with pytest.raises(pytest.skip.Exception, match="override the load_request fixture"):
        _RendersWithoutRenderRequests().render_at_ceiling(no_worker, None, None)
    with pytest.raises(pytest.skip.Exception, match="has no synthesize or design"):
        _QaContract().render_at_ceiling(no_worker, {"device": "cpu"}, None)
