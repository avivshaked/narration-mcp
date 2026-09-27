"""The fake worker's ``load`` checks what the real workers check (design section 4, section 10.1, App. A).

In process, with ``FakeHandler`` itself: a Qwen load's ``model``, ``device``, ``dtype``,
``attn_implementation``, ``determinism`` and ``settings`` that the ``qwen3`` worker would refuse are refused
by the fake too, with the same code and ``details.field``. The fake checks ``settings`` with ``qwen3``'s own
parser (``narration_worker.qwen_settings``, tested in ``test_qwen_settings.py``). The shared contract suite
checks the ceiling for both workers (``narration_worker.testing.contract``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from narration_worker.errors import OpError
from narration_worker.fake.faults import SPEC_ENV
from narration_worker.fake.handler import DEFAULT_MAX_NEW_TOKENS, FakeHandler
from narration_worker.handler import WorkerContext
from narration_worker.qwen_settings import CEILING_FIELD, SettingsError, parse_settings
from narration_worker.testing.contract import DETERMINISM, GENERATION

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


def settings(**generation: Any) -> dict[str, Any]:
    """Complete ``settings``: every sampling value (``GENERATION``), with ``generation``'s changes."""
    return {"non_streaming_mode": False, "generation": {**GENERATION, **generation}}


def qwen_load(store: Path, **overrides: Any) -> dict[str, Any]:
    """A complete Qwen load, as the daemon sends one; an override of None leaves that member out."""
    request: dict[str, Any] = {
        "device": "cpu",
        "model": ref(snapshot(store)),
        "engine_profile_id": "fake-test",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "determinism": dict(DETERMINISM),
        "settings": settings(),
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
        (lambda s: {"dtype": "float64"}, "dtype"),
        (lambda s: {"attn_implementation": None}, "attn_implementation"),
        (lambda s: {"attn_implementation": "flash"}, "attn_implementation"),
        (lambda s: {"determinism": None}, "determinism"),
        (lambda s: {"determinism": {"tf32": False}}, "determinism"),
        (lambda s: {"settings": None}, "settings"),
    ],
)
def test_a_qwen_load_is_refused_as_the_qwen3_worker_refuses_it_s4(
    handler: FakeHandler, store: Path, change: Callable[[Path], dict[str, Any]], field: str
) -> None:
    assert refusal(lambda: handler.op_load(qwen_load(store, **change(store)))) == ("INVALID_REQUEST", field)


QWEN_SETTINGS_REFUSALS: list[tuple[object, str]] = [
    ({}, "settings.non_streaming_mode"),
    ({"generation": GENERATION}, "settings.non_streaming_mode"),
    ({"non_streaming_mode": "false", "generation": GENERATION}, "settings.non_streaming_mode"),
    ({"non_streaming_mode": False}, "settings.generation"),
    ({"non_streaming_mode": False, "generation": {}}, "settings.generation.do_sample"),
    ({"non_streaming_mode": False, "generation": {"max_new_tokens": 8192}}, "settings.generation.do_sample"),
    ({**settings(), "instruct": "calm"}, "settings.instruct"),
    (settings(typical_p=0.9), "settings.generation.typical_p"),
    (settings(top_p=1.5), "settings.generation.top_p"),
    (settings(top_k=0), "settings.generation.top_k"),
    (settings(temperature=True), "settings.generation.temperature"),
    (settings(repetition_penalty=float("nan")), "settings.generation.repetition_penalty"),
    (settings(do_sample=1), "settings.generation.do_sample"),
    (settings(max_new_tokens=1), CEILING_FIELD),
    ("8192", "settings"),
]
"""``settings`` the ``qwen3`` worker refuses, and the member it names (``qwen_settings.parse_settings``)."""


@pytest.mark.parametrize(("value", "field"), QWEN_SETTINGS_REFUSALS)
def test_a_qwen_loads_settings_are_refused_with_qwen3s_parser_s10_1(
    handler: FakeHandler, store: Path, value: object, field: str
) -> None:
    """Every audio-changing setting is checked as ``qwen3`` checks it, not only the ceiling: a ``generation``
    that gives the ceiling alone names ``do_sample``, the first value missing, as ``qwen3`` names it."""
    with pytest.raises(SettingsError) as caught:
        parse_settings(value)
    assert caught.value.field == field
    assert refusal(lambda: handler.op_load(qwen_load(store, settings=value))) == ("INVALID_REQUEST", field)


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


@pytest.mark.parametrize(("value", "field"), QWEN_SETTINGS_REFUSALS)
def test_settings_are_checked_as_qwen3_checks_them_even_without_model_s10_1(
    handler: FakeHandler, value: object, field: str
) -> None:
    """A load that carries ``settings`` gives all of them, its ceiling among them; nothing is defaulted
    (section 10.1, DC-4)."""
    assert refusal(lambda: handler.op_load({"device": "cpu", "settings": value})) == ("INVALID_REQUEST", field)
    handler.op_load({"device": "cpu", "settings": settings(max_new_tokens=30)})
    assert handler.max_new_tokens == 30


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
    handler.op_load(qwen_load(store, settings=settings(max_new_tokens=300)))
    clip = store / "scratch" / "clip.wav"
    design = {"description": "A calm voice.", "design_text": TEXT, "language": "English", "seed": 3}
    handler.op_design({**design, "max_new_tokens": 300, "out_path": str(clip)})
    voice = "sha256:" + "7" * 64
    handler.op_prepare_voice({"voice_hash": voice, "ref_wav": str(clip), "ref_text": TEXT, "x_vector_only_mode": False})
    bad = settings(max_new_tokens=1)
    assert refusal(lambda: handler.op_load(qwen_load(store, settings=bad))) == ("INVALID_REQUEST", CEILING_FIELD)
    assert refusal(lambda: handler.op_load(qwen_load(store, dtype="float64"))) == ("INVALID_REQUEST", "dtype")
    take = str(store / "scratch" / "take.wav")
    call = {"voice_hash": voice, "engine_text": TEXT, "language": "English", "seed": 4, "out_path": take}
    over = refusal(lambda: handler.op_synthesize({**call, "max_new_tokens": 301}))
    assert over == ("INVALID_REQUEST", "max_new_tokens")
    assert handler.op_synthesize({**call, "max_new_tokens": 300})["max_new_tokens"] == 300
