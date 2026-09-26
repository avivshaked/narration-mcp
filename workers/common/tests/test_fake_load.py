"""The fake worker's ``load`` checks what the real workers check (design section 4, section 10.1, App. A).

In process, with ``FakeHandler`` itself: a daemon that sends a load the ``qwen3`` worker would refuse must be
refused by the fake too, with the same code and ``details.field``. The shared contract suite checks the
ceiling for both workers (``narration_worker.testing.contract``); these tests check the rest.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from narration_worker.errors import OpError
from narration_worker.fake.faults import SPEC_ENV
from narration_worker.fake.handler import CEILING_FIELD, DEFAULT_MAX_NEW_TOKENS, FakeHandler
from narration_worker.handler import WorkerContext
from narration_worker.testing.contract import DETERMINISM

REVISION = "c0ffee" + "0" * 34
TEXT = "Brisk winds carried the kites past the old mill."


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    (root / "scratch").mkdir(parents=True)
    return root


@pytest.fixture
def handler(store: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHandler:
    monkeypatch.delenv(SPEC_ENV, raising=False)
    return FakeHandler(WorkerContext(role="fake", store_root=store, cpu_threads=1))


def snapshot(store: Path, repo: str = "example/model", name: str = REVISION) -> Path:
    folder = store.parent / "models" / repo.replace("/", "--") / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def ref(folder: Path | str, revision: str = REVISION, repo: str = "example/model") -> dict[str, str]:
    return {"repo": repo, "revision": revision, "snapshot_dir": str(folder)}


def qwen_load(store: Path, **overrides: Any) -> dict[str, Any]:
    """A complete Qwen load, as the daemon sends one; an override of None leaves that member out."""
    request: dict[str, Any] = {
        "device": "cpu",
        "model": ref(snapshot(store)),
        "engine_profile_id": "fake-test",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "determinism": dict(DETERMINISM),
        "settings": {"non_streaming_mode": False, "generation": {"max_new_tokens": 8192}},
    }
    request.update(overrides)
    return {k: v for k, v in request.items() if v is not None}


def refusal(call: Callable[[], object]) -> tuple[str, object]:
    with pytest.raises(OpError) as caught:
        call()
    return caught.value.code, (caught.value.details or {}).get("field")


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda s: {"model": ref(Path("models") / REVISION)}, "model.snapshot_dir"),
        (lambda s: {"model": ref(snapshot(s, name="main"), revision="main")}, "model.revision"),
        (lambda s: {"model": ref(snapshot(s, name="main"))}, "model.snapshot_dir"),
        (lambda s: {"model": ref(snapshot(s), revision="A" * 40)}, "model.revision"),
        (lambda s: {"model": {"repo": "example/model"}}, "model"),
        (lambda s: {"device": "tpu"}, "device"),
        (lambda s: {"dtype": None}, "dtype"),
        (lambda s: {"attn_implementation": None}, "attn_implementation"),
        (lambda s: {"determinism": None}, "determinism"),
        (lambda s: {"determinism": {"tf32": False}}, "determinism"),
    ],
)
def test_a_qwen_load_is_refused_as_the_qwen3_worker_refuses_it_s4(
    handler: FakeHandler, store: Path, change: Callable[[Path], dict[str, Any]], field: str
) -> None:
    assert refusal(lambda: handler.op_load(qwen_load(store, **change(store)))) == ("INVALID_REQUEST", field)


@pytest.mark.parametrize(
    ("asr", "field"),
    [
        (lambda s: ref(Path("models") / REVISION), "models.asr.snapshot_dir"),
        (lambda s: ref(snapshot(s, name="main"), revision="main"), "models.asr.revision"),
        (lambda s: ref(snapshot(s, name="main")), "models.asr.snapshot_dir"),
    ],
)
def test_a_qa_loads_snapshots_are_checked_the_same_way_s4(
    handler: FakeHandler, store: Path, asr: Callable[[Path], dict[str, str]], field: str
) -> None:
    request = {"device": "cuda:0", "models": {"asr": asr(store), "sv": ref(snapshot(store, "example/sv"))}}
    assert refusal(lambda: handler.op_load(request)) == ("INVALID_REQUEST", field)


def test_every_snapshot_is_found_before_anything_else_is_checked_s14(handler: FakeHandler, store: Path) -> None:
    """A misnamed ``model`` and a missing ``models.asr``: the missing snapshot is what the daemon hears."""
    request = qwen_load(
        store,
        model=ref(snapshot(store, name="main")),
        models={"asr": ref(store.parent / "models" / "absent" / REVISION)},
        device="tpu",
        settings=None,
    )
    code, field = refusal(lambda: handler.op_load(request))
    assert (code, field) == ("BACKEND_NOT_INSTALLED", "models.asr")


@pytest.mark.parametrize(
    ("settings", "field"),
    [
        ({"non_streaming_mode": False}, "settings.generation"),
        ({"non_streaming_mode": False, "generation": {}}, CEILING_FIELD),
        ({"non_streaming_mode": False, "generation": {"max_new_tokens": 1}}, CEILING_FIELD),
        ("8192", "settings"),
    ],
)
def test_settings_without_a_valid_ceiling_are_refused_even_without_model_s10_1(
    handler: FakeHandler, settings: object, field: str
) -> None:
    """A load that carries ``settings`` gives its ceiling; it is never defaulted (DC-4)."""
    assert refusal(lambda: handler.op_load({"device": "cpu", "settings": settings})) == ("INVALID_REQUEST", field)


def test_a_load_that_is_not_a_qwen_load_keeps_the_default_ceiling_s10_1(handler: FakeHandler, store: Path) -> None:
    """The qwen3 worker never sees these loads: a QA load, and the fake's own bare load. vram_mb is null, as
    qwen3's is on the CPU: the fake holds nothing on a GPU."""
    qa = {"device": "cuda:0", "models": {"asr": ref(snapshot(store, "example/asr"), repo="example/asr")}}
    for request in ({"device": "cpu"}, qa):
        assert handler.op_load(request) == {"load_s": 0.0, "vram_mb": None}
        assert handler.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert handler.op_load(qwen_load(store)) == {"load_s": 0.0, "vram_mb": None}


def test_a_refused_load_changes_nothing_s10_1(handler: FakeHandler, store: Path) -> None:
    """The loaded ceiling and the prepared voices survive a load that is refused, as in the qwen3 worker."""
    low = {"non_streaming_mode": False, "generation": {"max_new_tokens": 300}}
    handler.op_load(qwen_load(store, settings=low))
    clip = store / "scratch" / "clip.wav"
    design = {"description": "A calm voice.", "design_text": TEXT, "language": "English", "seed": 3}
    handler.op_design({**design, "max_new_tokens": 300, "out_path": str(clip)})
    voice = "sha256:" + "7" * 64
    handler.op_prepare_voice({"voice_hash": voice, "ref_wav": str(clip), "ref_text": TEXT, "x_vector_only_mode": False})
    bad = {"non_streaming_mode": False, "generation": {"max_new_tokens": 1}}
    assert refusal(lambda: handler.op_load(qwen_load(store, settings=bad))) == ("INVALID_REQUEST", CEILING_FIELD)
    take = str(store / "scratch" / "take.wav")
    call = {"voice_hash": voice, "engine_text": TEXT, "language": "English", "seed": 4, "out_path": take}
    over = refusal(lambda: handler.op_synthesize({**call, "max_new_tokens": 301}))
    assert over == ("INVALID_REQUEST", "max_new_tokens")
    assert handler.op_synthesize({**call, "max_new_tokens": 300})["max_new_tokens"] == 300
