"""Engine profiles: what they pin, their hash, and the profile in use (design sections 6, 10.1 and 10.2;
plan.md WP32)."""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest

from narration.config import Config, EnginesConfig, QwenBaseConfig
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.models import EngineProfile
from narration.engine import current_profile, current_ref, require_expected
from narration.engine.models import PINNED, QWEN_BASE, QWEN_DESIGN, pinned
from narration.engine.profile import (
    QWEN_PACKAGES,
    UNHASHED,
    FileHashes,
    build_profile,
    determinism,
    hashed_object,
    next_profile_id,
    pin_differences,
    profile_hash,
    with_hash,
)
from narration.jobs.pins import call_cap, qwen_load_payload
from narration.store import NarrationStore, StoreIntegrityError
from tests.store.standin import StandInPlatform

from .support import GENERATION, VERSIONS, Install, make_install, make_snapshot, write_lock

REPO = Path(__file__).resolve().parents[2]
BASE_ID = "qwen3-base-1.7b.p1"
DESIGN_ID = "qwen3-design-1.7b.p1"


@pytest.fixture
def install(tmp_path: Path) -> Install:
    return make_install(tmp_path)


def _store(install: Install) -> NarrationStore:
    return NarrationStore(install.store_root, StandInPlatform())


# ------------------------------------------------------------------ what a profile pins


def test_a_built_profile_pins_the_model_the_snapshot_the_lock_and_the_settings_s6(install: Install) -> None:
    profile = build_profile(install.config, "base", engine_profile_id=BASE_ID)

    assert profile.engine_profile_id == BASE_ID
    assert profile.model_repo == names.MODEL_QWEN_BASE and profile.model_revision == QWEN_BASE.revision
    assert Path(profile.snapshot_dir) == install.snapshot(QWEN_BASE)
    assert set(profile.weights) == {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "speech_tokenizer/model.safetensors",
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in profile.weights.values())
    assert profile.packages == {p: VERSIONS[p] for p in QWEN_PACKAGES}
    assert re.fullmatch(r"[0-9a-f]{64}", profile.uv_lock_sha256)
    assert profile.worker_project == "workers/qwen3tts"  # never a local path: it is hashed
    assert profile.dtype == "bfloat16" and profile.licence == "Apache-2.0"
    assert profile.determinism.attn_implementation == "sdpa" and not profile.determinism.tf32
    assert profile.determinism.cudnn_deterministic and not profile.determinism.cudnn_benchmark
    assert profile.determinism.deterministic_algorithms == "warn_only"
    assert profile.determinism.cublas_workspace_config == ":4096:8"
    assert profile.settings == {
        "non_streaming_mode": False,
        "generation": GENERATION,
        "max_new_tokens_per_char": 2.5,
        "max_new_tokens_floor": 128,
    }
    assert profile.capabilities["controls"] == {"pace": False, "context": False, "instruct": False}
    assert profile.vram_need_mb == 6000
    assert profile.tier is None and profile.canary is None and profile.observed == {}
    assert profile.hash == profile_hash(profile) and profile.hash.startswith("sha256:")


def test_the_design_profile_pins_voicedesign_streaming_off_s10_1(install: Install) -> None:
    profile = build_profile(install.config, "design", engine_profile_id=DESIGN_ID)
    assert profile.model_repo == names.MODEL_QWEN_DESIGN and profile.model_revision == QWEN_DESIGN.revision
    assert profile.settings["non_streaming_mode"] is True
    assert profile.capabilities["ops"] == ["design"]


def test_the_job_engine_can_load_and_cap_a_built_profile_dc4(install: Install) -> None:
    """The job engine reads the settings as the profile records them (``narration.jobs.pins``)."""
    profile = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    payload = qwen_load_payload(profile, "cuda:0")
    assert payload["settings"] == {"non_streaming_mode": False, "generation": GENERATION}
    assert payload["model"]["revision"] == QWEN_BASE.revision
    assert call_cap(profile, "x" * 100) == 250
    assert call_cap(profile, "x") == 128


# ------------------------------------------------------------------ the hash


def test_the_hash_covers_every_field_but_the_five_the_contract_leaves_out_s6() -> None:
    fields = {f.name for f in dataclasses.fields(EngineProfile)}
    assert set(hashed_object(_any_profile())) == fields - UNHASHED
    assert frozenset({"hash", "snapshot_dir", "observed", "tier", "canary"}) == UNHASHED


def test_the_unhashed_fields_change_no_hash_s6(install: Install) -> None:
    profile = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    moved = dataclasses.replace(
        profile, snapshot_dir="elsewhere", observed={"gpu": "another"}, tier="similar", hash="sha256:" + "0" * 64
    )
    assert profile_hash(moved) == profile.hash


@pytest.mark.parametrize(
    "change",
    [
        "weights",
        "lock",
        "generation",
        "per_char",
        "non_streaming",
        "cublas",
    ],
)
def test_anything_that_changes_audio_changes_the_hash_s10_1(tmp_path: Path, change: str) -> None:
    install = make_install(tmp_path)
    before = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    config = install.config
    snapshot = install.snapshot(QWEN_BASE)
    if change == "weights":
        (snapshot / "model.safetensors").write_bytes(b"other weights")
    elif change == "lock":
        write_lock(install.project, {**VERSIONS, "transformers": "9.9.9"})
    elif change == "generation":
        make_snapshot(install.models_root, QWEN_BASE, generation={**GENERATION, "temperature": 0.7})
    elif change == "per_char":
        config = dataclasses.replace(
            config, engines=EnginesConfig(qwen3_base=QwenBaseConfig(max_new_tokens_per_char=3.0))
        )
    elif change == "non_streaming":
        config = dataclasses.replace(config, engines=EnginesConfig(qwen3_base=QwenBaseConfig(non_streaming_mode=True)))
    else:
        config = dataclasses.replace(
            config, workers=dataclasses.replace(config.workers, env={"CUBLAS_WORKSPACE_CONFIG": ":16:8"})
        )
    after = build_profile(config, "base", engine_profile_id=BASE_ID)
    assert after.hash != before.hash
    assert pin_differences(before, after)


def test_the_same_installation_gives_the_same_hash_in_any_folder_s10_2(tmp_path: Path) -> None:
    """The hash names no local path, so two installs of the same files agree."""
    a = build_profile(make_install(tmp_path / "a").config, "base", engine_profile_id=BASE_ID)
    b = build_profile(make_install(tmp_path / "b").config, "base", engine_profile_id=BASE_ID)
    assert a.snapshot_dir != b.snapshot_dir
    assert a.hash == b.hash
    assert pin_differences(a, b) == ()


def test_a_new_pin_is_a_new_id_and_a_new_hash_s10_1(install: Install) -> None:
    p1 = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    p2 = with_hash(dataclasses.replace(p1, engine_profile_id="qwen3-base-1.7b.p2"))
    assert p2.hash != p1.hash
    assert pin_differences(p1, p2) == ()  # the id aside, nothing changed


# ------------------------------------------------------------------ what a profile is built from


def test_a_missing_snapshot_is_backend_not_installed_s14(install: Install) -> None:
    for path in sorted(install.snapshot(QWEN_BASE).rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    install.snapshot(QWEN_BASE).rmdir()
    with pytest.raises(NarrationError) as caught:
        build_profile(install.config, "base", engine_profile_id=BASE_ID)
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED and "narration-admin install" in caught.value.hint


def test_a_sampling_value_left_to_a_library_default_is_refused_s10_1(install: Install) -> None:
    make_snapshot(install.models_root, QWEN_BASE, generation={k: v for k, v in GENERATION.items() if k != "top_k"})
    with pytest.raises(NarrationError) as caught:
        build_profile(install.config, "base", engine_profile_id=BASE_ID)
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED
    assert caught.value.details is not None and caught.value.details["missing"] == ["top_k"]


@pytest.mark.parametrize(("key", "value"), [("top_p", 1.5), ("do_sample", 1), ("max_new_tokens", 1), ("top_k", 2.5)])
def test_a_malformed_sampling_value_is_refused_s10_1(install: Install, key: str, value: object) -> None:
    make_snapshot(install.models_root, QWEN_BASE, generation={**GENERATION, key: value})
    with pytest.raises(NarrationError) as caught:
        build_profile(install.config, "base", engine_profile_id=BASE_ID)
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED and key in caught.value.message


def test_a_lock_without_a_pinned_package_is_refused_s6(install: Install) -> None:
    write_lock(install.project, {k: v for k, v in VERSIONS.items() if k != "torch"})
    with pytest.raises(NarrationError) as caught:
        build_profile(install.config, "base", engine_profile_id=BASE_ID)
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED and "torch" in caught.value.message


def test_a_missing_lock_is_backend_not_installed_s14(install: Install) -> None:
    (install.project / "uv.lock").unlink()
    with pytest.raises(NarrationError) as caught:
        build_profile(install.config, "base", engine_profile_id=BASE_ID)
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED


def test_no_qwen_worker_project_is_backend_not_installed_s14(tmp_path: Path) -> None:
    config = Config.for_tests(tmp_path / "store", tmp_path / "models")
    make_snapshot(tmp_path / "models", QWEN_BASE)
    with pytest.raises(NarrationError) as caught:
        build_profile(config, "base", engine_profile_id=BASE_ID)
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED and "[workers.qwen3]" in caught.value.hint


def test_hidden_files_in_a_snapshot_are_not_pinned_s6(install: Install) -> None:
    (install.snapshot(QWEN_BASE) / ".cache").mkdir()
    (install.snapshot(QWEN_BASE) / ".cache" / "lock").write_text("x", encoding="utf-8")
    profile = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    assert not any(name.startswith(".") for name in profile.weights)


def test_file_hashes_are_read_again_only_when_a_file_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"a" * 1000)
    hashes = FileHashes()
    opened: list[Path] = []
    real_open = Path.open

    def counting_open(self: Path, *args: object, **kwargs: object) -> object:
        if args[:1] == ("rb",):  # reads only: writing the file opens it too
            opened.append(self)
        return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", counting_open)
    first = hashes.sha256(path)
    assert hashes.sha256(path) == first and len(opened) == 1
    path.write_bytes(b"b" * 1001)
    assert hashes.sha256(path) != first and len(opened) == 2


# ------------------------------------------------------------------ ids and the profile in use


def test_profile_ids_count_up_per_family_s6() -> None:
    assert next_profile_id([], "base") == "qwen3-base-1.7b.p1"
    existing = ["qwen3-base-1.7b.p1", "qwen3-base-1.7b.p2", "qwen3-design-1.7b.p1", "qwen3-base-1.7b.test"]
    assert next_profile_id(existing, "base") == "qwen3-base-1.7b.p3"
    assert next_profile_id(existing, "design") == "qwen3-design-1.7b.p2"
    assert next_profile_id(["qwen3-base-1.7b.p9", "qwen3-base-1.7b.p10"], "base") == "qwen3-base-1.7b.p11"


def test_the_first_ids_are_the_designs_names_s6() -> None:
    assert next_profile_id([], "base") == names.ENGINE_PROFILE_BASE
    assert next_profile_id([], "design") == names.ENGINE_PROFILE_DESIGN


def test_before_a_pin_there_is_no_current_profile_s14(install: Install) -> None:
    with _store(install) as store, pytest.raises(NarrationError) as caught:
        current_profile(store, "base")
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED and "engine pin" in caught.value.hint


def test_the_current_profile_is_the_one_the_store_pins_s10_1(install: Install) -> None:
    profile = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    with _store(install) as store:
        store.put_engine_profile(profile)
        store.set_current_engine_profile("base", BASE_ID)
        assert current_profile(store, "base").hash == profile.hash
        ref = current_ref(store)
        assert (ref.id, ref.hash) == (BASE_ID, profile.hash)
        with pytest.raises(NarrationError):
            current_profile(store, "design")


def test_a_different_expected_profile_is_engine_changed_s7_3(install: Install) -> None:
    profile = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    with _store(install) as store:
        store.put_engine_profile(profile)
        store.set_current_engine_profile("base", BASE_ID)
        assert require_expected(store, None).hash == profile.hash
        assert require_expected(store, profile.hash).hash == profile.hash
        with pytest.raises(NarrationError) as caught:
            require_expected(store, "sha256:" + "0" * 64)
    error = caught.value
    assert error.code == codes.ENGINE_CHANGED and error.field == "expect_engine_profile" and not error.retryable
    assert error.details is not None and error.details["current"] == profile.hash


def test_the_store_refuses_a_changed_profile_under_a_pinned_id_s10_1(install: Install) -> None:
    """A re-pin is a new id: the store keeps a pinned profile's hashed fields as they were."""
    profile = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    (install.snapshot(QWEN_BASE) / "model.safetensors").write_bytes(b"other")
    changed = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    with _store(install) as store:
        store.put_engine_profile(profile)
        store.put_engine_profile(dataclasses.replace(profile, tier="bit_exact", observed={"gpu": "x"}))
        with pytest.raises(StoreIntegrityError):
            store.put_engine_profile(changed)


# ------------------------------------------------------------------ the pins agree with the rest of the repo


def test_the_pinned_revisions_are_the_ones_the_dev_models_tool_installs() -> None:
    source = (REPO / "tools" / "dev_models.py").read_text(encoding="utf-8")
    for model in PINNED.values():
        assert f'"{model.repo}", "{model.revision}"' in source, model.repo


def test_every_package_a_profile_pins_is_in_the_qwen_workers_fingerprint() -> None:
    """The worker reports the versions the drift check compares (``Qwen3Handler.fingerprint_packages``)."""
    source = (REPO / "workers" / "qwen3tts" / "src" / "narration_qwen3tts" / "worker.py").read_text(encoding="utf-8")
    reported: tuple[str, ...] = ()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "fingerprint_packages" for t in node.targets
        ):
            reported = tuple(ast.literal_eval(node.value))
    assert reported and set(QWEN_PACKAGES) <= set(reported)


def test_every_package_a_profile_pins_is_in_the_qwen_workers_lock() -> None:
    import tomllib

    lock = tomllib.loads((REPO / "workers" / "qwen3tts" / "uv.lock").read_text(encoding="utf-8"))
    locked = {p["name"] for p in lock["package"]}
    assert set(QWEN_PACKAGES) <= locked


def test_pinned_looks_up_a_model_and_names_the_others_when_it_cannot() -> None:
    assert pinned(names.MODEL_ASR).revision == PINNED[names.MODEL_ASR].revision
    with pytest.raises(KeyError, match="whisper"):
        pinned("someone/else")


def test_a_pin_needs_a_40_hex_revision() -> None:
    from narration.engine.models import PinnedModel

    with pytest.raises(ValueError):
        PinnedModel(repo="a/b", revision="main", licence="x")


def _any_profile() -> EngineProfile:
    root = Path("unused")
    return EngineProfile(
        engine_profile_id=BASE_ID,
        hash="",
        model_repo="a/b",
        model_revision="0" * 40,
        snapshot_dir=str(root),
        weights={},
        worker_project="workers/qwen3tts",
        uv_lock_sha256="0" * 64,
        packages={},
        dtype="bfloat16",
        determinism=determinism(Config.for_tests(root)),
        settings={},
        capabilities={},
        licence="x",
        vram_need_mb=1,
    )


def test_hashed_objects_are_plain_json_s10_2(install: Install) -> None:
    profile = build_profile(install.config, "base", engine_profile_id=BASE_ID)
    json.dumps(hashed_object(profile))
