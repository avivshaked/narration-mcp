"""The worker fingerprint check (design sections 10.1 and 14; plan.md WP32's acceptance: a changed ``uv.lock``
or weight file is caught as drift)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import EngineProfile
from narration.contracts.worker import HelloReply
from narration.engine.drift import MAX_LISTED, DriftCheck
from narration.engine.models import QWEN_BASE
from narration.engine.profile import build_profile

from .support import VERSIONS, Install, hello, make_install, write_lock

BASE_ID = "qwen3-base-1.7b.p1"


@pytest.fixture
def install(tmp_path: Path) -> Install:
    return make_install(tmp_path)


@pytest.fixture
def profile(install: Install) -> EngineProfile:
    return build_profile(install.config, "base", engine_profile_id=BASE_ID)


def _hello(profile: EngineProfile, **kwargs: Any) -> HelloReply:
    return cast(HelloReply, hello(profile.packages, **kwargs))


def test_an_engine_as_pinned_has_no_drift_s10_1(install: Install, profile: EngineProfile) -> None:
    check = DriftCheck()
    assert check.find(profile, _hello(profile), project=install.project) == []
    check.require_none(profile, _hello(profile), project=install.project)


def test_a_changed_uv_lock_is_drift_wp32(install: Install, profile: EngineProfile) -> None:
    write_lock(install.project, {**VERSIONS, "torch": "9.0.0"})
    with pytest.raises(NarrationError) as caught:
        DriftCheck().require_none(profile, _hello(profile), project=install.project)
    error = caught.value
    assert error.code == codes.ENGINE_DRIFT and not error.retryable and error.hint
    assert error.details is not None
    assert [d["what"] for d in error.details["drift"]] == ["uv_lock"]
    assert error.details["drift"][0]["pinned"] == profile.uv_lock_sha256


def test_a_missing_uv_lock_is_drift_wp32(install: Install, profile: EngineProfile) -> None:
    (install.project / "uv.lock").unlink()
    found = DriftCheck().find(profile, _hello(profile), project=install.project)
    assert [(d.what, d.found) for d in found] == [("uv_lock", None)]


def test_a_changed_weight_file_is_drift_wp32(install: Install, profile: EngineProfile) -> None:
    check = DriftCheck()
    assert check.find(profile, _hello(profile), project=install.project) == []  # hashes remembered
    weights = install.snapshot(QWEN_BASE) / "model.safetensors"
    weights.write_bytes(b"not the pinned weights")
    found = check.find(profile, _hello(profile), project=install.project)
    assert [(d.what, d.name) for d in found] == [("weights", "model.safetensors")]
    assert found[0].pinned == profile.weights["model.safetensors"] and found[0].found != found[0].pinned


def test_a_weight_file_rewritten_with_its_old_size_and_time_is_still_read_on_a_new_daemon_wp32(
    install: Install, profile: EngineProfile
) -> None:
    """A daemon's first check reads every file; only its later checks trust an unchanged size and time."""
    weights = install.snapshot(QWEN_BASE) / "model.safetensors"
    stat = weights.stat()
    data = bytearray(weights.read_bytes())
    data[0] ^= 0xFF
    weights.write_bytes(bytes(data))
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    found = DriftCheck().find(profile, _hello(profile), project=install.project)
    assert [(d.what, d.name) for d in found] == [("weights", "model.safetensors")]


def test_a_missing_or_added_snapshot_file_is_drift_s10_1(install: Install, profile: EngineProfile) -> None:
    snapshot = install.snapshot(QWEN_BASE)
    (snapshot / "config.json").unlink()
    (snapshot / "extra.safetensors").write_bytes(b"more")
    found = DriftCheck().find(profile, _hello(profile), project=install.project)
    assert {(d.what, d.name, d.found is None, d.pinned is None) for d in found} == {
        ("weights", "config.json", True, False),
        ("weights", "extra.safetensors", False, True),
    }


def test_a_missing_snapshot_is_drift_of_every_file_s10_1(install: Install, profile: EngineProfile) -> None:
    moved = Path(str(install.snapshot(QWEN_BASE)) + "-moved")
    install.snapshot(QWEN_BASE).rename(moved)
    found = DriftCheck().find(profile, _hello(profile), project=install.project)
    assert {d.name for d in found} == set(profile.weights) and all(d.found is None for d in found)


def test_a_package_the_worker_reports_at_another_version_is_drift_s10_1(
    install: Install, profile: EngineProfile
) -> None:
    packages = {**profile.packages, "transformers": "0.0.1"}
    packages.pop("soxr")
    found = DriftCheck().find(profile, cast(HelloReply, hello(packages)), project=install.project)
    assert {(d.what, d.name, d.found) for d in found} == {
        ("package", "transformers", "0.0.1"),
        ("package", "soxr", None),
    }


def test_extra_packages_the_worker_reports_are_not_drift_s10_1(install: Install, profile: EngineProfile) -> None:
    packages = {**profile.packages, "narration-worker": "0.1.0", "pip": "25.0"}
    assert DriftCheck().find(profile, cast(HelloReply, hello(packages)), project=install.project) == []


@pytest.mark.parametrize("cublas", [":16:8", None])
def test_a_worker_without_the_pinned_cublas_workspace_is_drift_s10_1(
    install: Install, profile: EngineProfile, cublas: str | None
) -> None:
    found = DriftCheck().find(profile, _hello(profile, cublas=cublas), project=install.project)
    assert [(d.what, d.name, d.found) for d in found] == [("env", "CUBLAS_WORKSPACE_CONFIG", cublas)]


def test_a_worker_that_gave_no_fingerprint_is_drift_s10_1(install: Install, profile: EngineProfile) -> None:
    found = DriftCheck().find(profile, None, project=install.project)
    assert [d.what for d in found] == ["fingerprint"]


def test_drift_lists_a_bounded_number_of_differences_with_their_count_s14(
    install: Install, profile: EngineProfile
) -> None:
    snapshot = install.snapshot(QWEN_BASE)
    for n in range(MAX_LISTED + 5):
        (snapshot / f"extra-{n:02d}.bin").write_bytes(b"x")
    with pytest.raises(NarrationError) as caught:
        DriftCheck().require_none(profile, _hello(profile), project=install.project)
    details = caught.value.details
    assert details is not None and details["count"] == MAX_LISTED + 5 and len(details["drift"]) == MAX_LISTED
    assert details["engine_profile_id"] == BASE_ID


def test_an_unfinished_download_is_drift_that_says_to_install_again_wp37(
    install: Install, profile: EngineProfile
) -> None:
    """An install interrupted after the pin leaves ``.<name>.partial``: never hashed as a file of the model,
    and named as what it is."""
    snapshot = install.snapshot(QWEN_BASE)
    (snapshot / ".model.safetensors.partial").write_bytes(b"half a file")
    (snapshot / "speech_tokenizer" / "model.safetensors.partial").write_bytes(b"half another")
    found = DriftCheck().find(profile, _hello(profile), project=install.project)
    assert [(d.what, d.name) for d in found] == [
        ("download", ".model.safetensors.partial"),
        ("download", "speech_tokenizer/model.safetensors.partial"),
    ]
    with pytest.raises(NarrationError) as caught:
        DriftCheck().require_none(profile, _hello(profile), project=install.project)
    error = caught.value
    assert error.code == codes.ENGINE_DRIFT and "unfinished download" in error.message
    assert "narration-admin install again" in error.hint
