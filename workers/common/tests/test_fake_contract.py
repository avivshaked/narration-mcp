"""The fake worker keeps the same contract as the real workers (plan.md WP16 acceptance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from narration_worker.fake.handler import FAKE_REVISION
from narration_worker.testing.client import WorkerProcess
from narration_worker.testing.contract import DETERMINISM, GENERATION, WorkerContract

TEXT = "When the loaves come out golden and crisp."
VOICE = "sha256:" + "5" * 64


def qwen_load_request(store_root: Path) -> dict[str, Any]:
    """A complete Qwen load, as the daemon sends one, of a snapshot folder it creates beside the store."""
    snapshot = store_root.parent / "models" / FAKE_REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    return {
        "device": "cpu",
        "model": {"repo": "narration-worker/fake", "revision": FAKE_REVISION, "snapshot_dir": str(snapshot)},
        "engine_profile_id": "fake-contract",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "determinism": dict(DETERMINISM),
        "settings": {"non_streaming_mode": False, "generation": dict(GENERATION)},
    }


class TestFakeWorkerContract(WorkerContract):
    role = "fake"
    timeout_s = 30.0

    @pytest.fixture
    def load_request(self, store_root: Path) -> dict[str, Any] | None:
        """A complete Qwen load: the fake checks it as the qwen3 worker does."""
        return qwen_load_request(store_root)

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


@pytest.mark.parametrize("missing", ["model", "determinism", "settings", "generation", "top_k", "max_new_tokens"])
def test_the_ceiling_test_fails_clearly_for_an_incomplete_qwen_load_s10_1(tmp_path: Path, missing: str) -> None:
    """A rendering role's ``load_request`` must be a complete Qwen load, or the ceiling test fails with a
    message that says so, before it sends anything (a load without ``determinism`` would otherwise be refused
    for that, and the test would blame the ceiling)."""
    request = qwen_load_request(tmp_path / "store")
    settings, generation = request["settings"], request["settings"]["generation"]
    for member in (request, settings, generation):
        member.pop(missing, None)
    no_worker = cast(WorkerProcess, None)  # never reached: the check comes first
    with pytest.raises(pytest.fail.Exception, match="must be a complete Qwen load"):
        _RendersWithoutRenderRequests().test_a_load_without_a_valid_ceiling_is_invalid_request_s10_1(no_worker, request)
