"""``doctor``'s answers the WP37 review asked for: what NVML actually said, a next step for a store that
cannot be read, and a broken engine package told apart from an absent one (design section 7.1; WP44)."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from narration.admin import doctor
from narration.admin.doctor import GpuUnavailable, check_gpu, diagnose
from narration.config import load_config
from narration.platform.testing import StandInPlatform

from .test_doctor import make_admin, probes


def nvml_error(name: str) -> Exception:
    """An exception of NVML's class ``name`` (pynvml makes one class per error code)."""
    return type(name, (Exception,), {})(name.removeprefix("NVMLError_"))


@pytest.mark.parametrize("name", ["NVMLError_LibraryNotFound", "NVMLError_DriverNotLoaded"])
def test_no_nvidia_driver_is_the_no_gpu_message_wp44(name: str) -> None:
    failure = doctor._nvml_start_failure(nvml_error(name))  # pyright: ignore[reportPrivateUsage]
    assert failure.no_driver is True
    finding = check_gpu(None, lambda index: (_ for _ in ()).throw(failure))
    assert "no NVIDIA GPU can be used here" in finding.summary


def test_a_driver_library_mismatch_says_to_restart_s7_1() -> None:
    failure = doctor._nvml_start_failure(nvml_error("NVMLError_LibRmVersionMismatch"))  # pyright: ignore[reportPrivateUsage]
    assert failure.no_driver is False
    finding = check_gpu(None, lambda index: (_ for _ in ()).throw(failure))
    assert "no NVIDIA GPU" not in finding.summary and "does not match its library" in finding.summary
    assert finding.next_step is not None and "Restart the machine" in finding.next_step


def test_any_other_nvml_failure_is_reported_in_nvmls_words_s7_1() -> None:
    failure = doctor._nvml_start_failure(nvml_error("NVMLError_NoPermission"))  # pyright: ignore[reportPrivateUsage]
    finding = check_gpu(None, lambda index: (_ for _ in ()).throw(failure))
    assert "NVMLError_NoPermission" in finding.summary and "no NVIDIA driver" not in finding.summary
    assert finding.next_step


def test_a_store_that_cannot_be_read_has_a_next_step_s7_1(
    config_path: Path, platform: StandInPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(config_path)
    config.server.store_root.mkdir(parents=True)
    (config.server.store_root / "narration.sqlite").write_bytes(b"not a database")
    admin, _ = make_admin(config_path, platform)
    (engine,) = [f for f in diagnose(admin, probes=probes()) if f.area == "engine"]
    assert engine.level == "fail" and engine.next_step is not None and "verify" in engine.next_step


def test_engine_pins_that_cannot_load_are_a_finding_not_a_traceback_s7_1(
    config_path: Path, platform: StandInPlatform
) -> None:
    def broken() -> dict[str, str] | None:
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    admin, _ = make_admin(config_path, platform)
    findings = diagnose(admin, probes=dataclasses.replace(probes(), pins=broken))
    (models,) = [f for f in findings if f.area == "models"]
    assert models.level == "fail" and "cannot be loaded" in models.summary and models.next_step


@pytest.mark.parametrize(
    ("missing", "present"),
    [("narration.engine", False), ("narration.engine.admin", False), ("torch", True)],
)
def test_the_engine_module_is_absent_only_when_it_is_the_one_missing_s7_1(
    monkeypatch: pytest.MonkeyPatch, missing: str, present: bool
) -> None:
    def find_spec(name: str) -> object:
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert doctor.engine_module_present() is present


def test_gpu_unavailable_keeps_its_message() -> None:
    assert str(GpuUnavailable("NVML could not start")) == "NVML could not start"
