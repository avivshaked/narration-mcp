"""``narration-admin doctor``: every check says what it found and, for a problem, what to do next (design
sections 7.1, 16 and 17.8; plan.md WP37 and WP44).

The machine is replaced by probes: the GPU, the venv check and the pins are the test's. The models root and
the worker projects are folders the tests build under tmp_path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from narration.admin import doctor
from narration.admin.cli import EXIT_FAILED, EXIT_OK, Admin
from narration.admin.doctor import Finding, GpuFacts, GpuUnavailable, Probes, SyncCheck, diagnose, run_doctor
from narration.admin.models import InstalledModel, read_manifest, snapshot_dir, write_manifest
from narration.config import load_config
from narration.contracts import names
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore
from tests.store.factories import engine_profile

from .conftest import write_config

REV_A = "a" * 40
REV_B = "b" * 40
SYNCED = SyncCheck(True, "its venv matches its uv.lock")
GOOD_GPU = GpuFacts(index=0, name="A GPU", total_mb=24000, free_mb=20000, driver="1.0", count=1)


def gpu_ok(index: int) -> GpuFacts:
    return GOOD_GPU


def no_gpu(index: int) -> GpuFacts:
    raise GpuUnavailable("NVML could not start (Driver Not Loaded): no NVIDIA driver is loaded", no_driver=True)


def probes(
    *,
    gpu: Callable[[int], GpuFacts] = gpu_ok,
    synced: SyncCheck = SYNCED,
    pins: dict[str, str] | None = None,
    supported: bool = True,
    engine_module: bool = False,
) -> Probes:
    return Probes(
        gpu=gpu,
        venv_synced=lambda project: synced,
        pins=lambda: pins,
        engine_module=lambda: engine_module,
        supported=lambda: supported,
    )


def install_model(models_root: Path, repo: str, revision: str, files: dict[str, bytes]) -> InstalledModel:
    """Put a model's files in its snapshot folder and record them, as ``install`` would."""
    folder = snapshot_dir(models_root, repo, revision)
    for rel, data in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(data)
    model = InstalledModel(
        repo=repo,
        revision=revision,
        snapshot_dir=folder,
        files_sha256={rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()},
    )
    records = read_manifest(models_root)
    records[model.key] = model
    write_manifest(models_root, records)
    return model


def worker_projects(service: Path, *, venv: bool = True) -> None:
    for folder in ("qwen3tts", "qa"):
        project = service / "workers" / folder
        project.mkdir(parents=True, exist_ok=True)
        (project / "pyproject.toml").write_text("[project]\nname = 'w'\n", encoding="utf-8")
        if venv:
            python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"")


@pytest.fixture
def ready(config_path: Path) -> Path:
    """A service folder where everything the tests control is ready: models installed and recorded, worker
    projects with venvs. Returns the service folder."""
    service = config_path.parent
    install_model(service / "models", names.MODEL_ALIGNER, REV_A, {"config.json": b"{}", "model.bin": b"weights"})
    worker_projects(service)
    return service


def run(admin: Admin, *, quick: bool = False, as_json: bool = False, probes_: Probes | None = None) -> int:
    return run_doctor(admin, argparse.Namespace(quick=quick, json=as_json), probes=probes_)


def make_admin(config_path: Path | None, platform: StandInPlatform) -> tuple[Admin, io.StringIO]:
    out = io.StringIO()
    return Admin(config_path=config_path, out=out, err=io.StringIO(), platform=lambda: platform, environ={}), out


def levels(findings: list[Finding], area: str) -> list[str]:
    return [f.level for f in findings if f.area == area]


def test_every_problem_says_what_to_do_next_s7_1(tmp_path: Path, platform: StandInPlatform) -> None:
    admin, _ = make_admin(write_config(tmp_path / "bare"), platform)
    findings = diagnose(admin, probes=probes(gpu=no_gpu, synced=SyncCheck(False, "differs"), supported=False))
    problems = [f for f in findings if f.level in ("warn", "fail")]
    assert problems, "the bare configuration has problems"
    assert all(f.next_step for f in problems), [f for f in problems if not f.next_step]


def test_a_ready_machine_passes_s7_1(ready: Path, config_path: Path, platform: StandInPlatform) -> None:
    admin, out = make_admin(config_path, platform)
    pins = {names.MODEL_ALIGNER: REV_A}
    code = run(admin, probes_=probes(pins=pins))
    text = out.getvalue()
    assert "FAIL" not in text, text
    assert code == EXIT_OK
    assert "their hashes match the install record" in text and "A GPU, 24000 MB" in text


def test_an_unsupported_os_fails_the_platform_check_s7_1(
    ready: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, _ = make_admin(config_path, platform)
    findings = diagnose(admin, probes=probes(supported=False))
    (finding,) = [f for f in findings if f.area == "platform"]
    assert finding.level == "fail" and "Windows only" in finding.summary


def test_no_configuration_is_a_failure_that_says_where_to_put_one_s16(platform: StandInPlatform) -> None:
    admin, out = make_admin(None, platform)
    assert run(admin, probes_=probes()) == EXIT_FAILED
    assert "FAIL  config" in out.getvalue()
    findings = diagnose(admin, probes=probes())
    (config,) = [f for f in findings if f.area == "config"]
    assert config.level == "fail" and "--config" in config.summary
    assert levels(findings, "gpu") == ["ok"], "the GPU is checked even without a configuration"


def test_a_machine_without_an_nvidia_gpu_is_told_plainly_it_cannot_run_jobs_wp44(
    ready: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, _ = make_admin(config_path, platform)
    (gpu,) = [f for f in diagnose(admin, probes=probes(gpu=no_gpu)) if f.area == "gpu"]
    assert gpu.level == "fail"
    assert "no NVIDIA GPU can be used here" in gpu.summary and "cannot run jobs" in gpu.summary
    assert gpu.next_step is not None and "driver" in gpu.next_step


def test_a_gpu_device_that_is_not_cuda_n_is_refused_s16(tmp_path: Path, platform: StandInPlatform) -> None:
    admin, _ = make_admin(write_config(tmp_path / "s", '[gpu]\ndevice = "gpu0"\n'), platform)
    (gpu,) = [f for f in diagnose(admin, probes=probes()) if f.area == "gpu"]
    assert gpu.level == "fail" and "cuda:<n>" in (gpu.next_step or "")


def test_the_store_root_must_be_writable_and_have_free_disk_s15(
    ready: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, _ = make_admin(config_path, platform)
    findings = diagnose(admin, probes=probes())
    assert levels(findings, "store") == ["ok"] and "will be created" in next(
        f.summary for f in findings if f.area == "store"
    )
    platform.free_bytes = 1024**3
    (disk,) = [f for f in diagnose(admin, probes=probes()) if f.area == "disk"]
    assert disk.level == "fail" and "min_free_disk_gb = 5" in disk.summary


def test_a_store_root_that_is_a_file_fails_s15(tmp_path: Path, platform: StandInPlatform) -> None:
    config_path = write_config(tmp_path / "s")
    (config_path.parent / "store").write_text("not a folder", encoding="utf-8")
    admin, _ = make_admin(config_path, platform)
    (store,) = [f for f in diagnose(admin, probes=probes()) if f.area == "store"]
    assert store.level == "fail" and "not a folder" in store.summary


def test_a_pinned_model_that_is_not_installed_fails_s17_8(
    ready: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, _ = make_admin(config_path, platform)
    pins = {names.MODEL_ALIGNER: REV_A, names.MODEL_ASR: REV_B}
    findings = [f for f in diagnose(admin, probes=probes(pins=pins)) if f.area == "models"]
    missing = [f for f in findings if f.level == "fail"]
    assert len(missing) == 1 and names.MODEL_ASR in missing[0].summary and "install" in (missing[0].next_step or "")


def test_a_model_file_changed_since_the_install_fails_and_quick_skips_the_hash_s15(
    ready: Path, config_path: Path, platform: StandInPlatform
) -> None:
    (snapshot_dir(ready / "models", names.MODEL_ALIGNER, REV_A) / "model.bin").write_bytes(b"tampered")
    admin, _ = make_admin(config_path, platform)
    pins = {names.MODEL_ALIGNER: REV_A}
    (models,) = [f for f in diagnose(admin, probes=probes(pins=pins)) if f.area == "models"]
    assert models.level == "fail" and "model.bin does not match its recorded sha256" in models.summary
    (quick,) = [f for f in diagnose(admin, probes=probes(pins=pins), hash_models=False) if f.area == "models"]
    assert quick.level == "ok" and "not hashed" in quick.summary
    (snapshot_dir(ready / "models", names.MODEL_ALIGNER, REV_A) / "model.bin").unlink()
    (gone,) = [f for f in diagnose(admin, probes=probes(pins=pins), hash_models=False) if f.area == "models"]
    assert gone.level == "fail" and "model.bin is missing" in gone.summary


def test_a_model_without_an_install_record_is_a_warning_s17_8(
    ready: Path, config_path: Path, platform: StandInPlatform
) -> None:
    snapshot_dir(ready / "models", names.MODEL_ASR, REV_B).mkdir(parents=True)
    admin, _ = make_admin(config_path, platform)
    pins = {names.MODEL_ALIGNER: REV_A, names.MODEL_ASR: REV_B}
    findings = [f for f in diagnose(admin, probes=probes(pins=pins)) if f.area == "models"]
    assert [f.level for f in findings] == ["ok", "warn"]
    assert "not in the install record" in findings[1].summary


def test_without_the_pins_the_install_record_is_checked_s17_8(
    ready: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, _ = make_admin(config_path, platform)
    findings = [f for f in diagnose(admin, probes=probes(pins=None)) if f.area == "models"]
    assert [f.level for f in findings] == ["info", "ok"]
    assert "unknown" in findings[0].summary


def test_a_models_root_that_is_missing_or_whose_record_is_broken_fails_s17_8(
    tmp_path: Path, platform: StandInPlatform
) -> None:
    config_path = write_config(tmp_path / "s")
    admin, _ = make_admin(config_path, platform)
    (missing,) = [f for f in diagnose(admin, probes=probes()) if f.area == "models"]
    assert missing.level == "fail" and "does not exist" in missing.summary
    (config_path.parent / "models").mkdir()
    (config_path.parent / "models" / "manifest.json").write_text("[1, 2]", encoding="utf-8")
    (broken,) = [f for f in diagnose(admin, probes=probes()) if f.area == "models"]
    assert broken.level == "fail" and "not an install record" in broken.summary


def test_an_aligner_this_build_does_not_pin_is_a_warning_s11_2(tmp_path: Path, platform: StandInPlatform) -> None:
    config_path = write_config(tmp_path / "s", f'[alignment]\nmodel = "{names.MODEL_ALIGNER_ALTERNATIVE}"\n')
    (config_path.parent / "models").mkdir()
    admin, _ = make_admin(config_path, platform)
    findings = [f for f in diagnose(admin, probes=probes(pins={names.MODEL_ALIGNER: REV_A})) if f.area == "models"]
    assert any(f.level == "warn" and "does not pin" in f.summary for f in findings)


@pytest.mark.parametrize(
    "alignment",
    [f'model = "{names.MODEL_QWEN_BASE}"', 'device = "cuda:0"'],
    ids=["pinned_but_not_ctc", "not_on_the_cpu"],
)
def test_an_aligner_the_ctc_method_cannot_run_fails_s11_2(
    tmp_path: Path, platform: StandInPlatform, alignment: str
) -> None:
    """Doctor finds it, not the first job: a pinned model that is not a CTC model, or the aligner on a GPU."""
    config_path = write_config(tmp_path / "s", "[alignment]\n" + alignment + "\n")
    (config_path.parent / "models").mkdir()
    admin, _ = make_admin(config_path, platform)
    pins = {names.MODEL_ALIGNER: REV_A, names.MODEL_QWEN_BASE: REV_B}
    findings = [f for f in diagnose(admin, probes=probes(pins=pins)) if f.area == "models"]
    (cannot,) = [f for f in findings if "the aligner cannot run" in f.summary]
    assert cannot.level == "fail"
    assert cannot.next_step is not None and "[alignment]" in cannot.next_step


def test_the_default_aligner_is_not_reported_s11_2(tmp_path: Path, platform: StandInPlatform) -> None:
    config_path = write_config(tmp_path / "s", "")
    (config_path.parent / "models").mkdir()
    admin, _ = make_admin(config_path, platform)
    findings = [f for f in diagnose(admin, probes=probes(pins={names.MODEL_ALIGNER: REV_A})) if f.area == "models"]
    assert findings  # the models root exists, so the pinned models were checked
    assert not [f for f in findings if "the aligner cannot run" in f.summary]


def test_the_qa_profile_this_build_scores_with_is_reported_s16(tmp_path: Path, platform: StandInPlatform) -> None:
    config_path = write_config(tmp_path / "s", f'[qa]\nprofile = "{names.QA_PROFILE}"\n')
    admin, _ = make_admin(config_path, platform)
    (qa,) = [f for f in diagnose(admin, probes=probes()) if f.area == "qa"]
    assert qa.level == "ok" and names.QA_PROFILE in qa.summary and qa.next_step is None


def test_a_qa_profile_other_than_this_builds_is_a_warning_never_a_failure_s16(
    ready: Path, config_path: Path, platform: StandInPlatform
) -> None:
    """``[qa] profile`` changes nothing (the scorer uses the build's profile): doctor says which profile runs and
    to update the line, and the machine still passes."""
    with config_path.open("a", encoding="utf-8") as fh:
        fh.write('[qa]\nprofile = "an-older-profile.v1"\n')
    admin, out = make_admin(config_path, platform)
    (qa,) = [f for f in diagnose(admin, probes=probes(pins={names.MODEL_ALIGNER: REV_A})) if f.area == "qa"]
    assert qa.level == "warn"
    assert "an-older-profile.v1" in qa.summary and names.QA_PROFILE in qa.summary
    assert qa.next_step is not None and f'profile = "{names.QA_PROFILE}"' in qa.next_step
    assert run(admin, probes_=probes(pins={names.MODEL_ALIGNER: REV_A})) == EXIT_OK
    assert "FAIL" not in out.getvalue()


@pytest.mark.parametrize(
    ("synced", "level"),
    [
        (SyncCheck(True, "matches"), "ok"),
        (SyncCheck(False, "does not match"), "fail"),
        (SyncCheck(None, "unknown"), "warn"),
    ],
)
def test_each_worker_venv_must_match_its_lock_s4(
    ready: Path, config_path: Path, platform: StandInPlatform, synced: SyncCheck, level: str
) -> None:
    admin, _ = make_admin(config_path, platform)
    findings = diagnose(admin, probes=probes(synced=synced))
    assert levels(findings, "worker qwen3") == [level] and levels(findings, "worker qa") == [level]


def test_a_worker_without_its_project_or_venv_fails_s4(tmp_path: Path, platform: StandInPlatform) -> None:
    config_path = write_config(tmp_path / "s")
    admin, _ = make_admin(config_path, platform)
    (qwen,) = [f for f in diagnose(admin, probes=probes()) if f.area == "worker qwen3"]
    assert qwen.level == "fail" and "no pyproject.toml" in qwen.summary
    worker_projects(config_path.parent, venv=False)
    (qwen,) = [f for f in diagnose(admin, probes=probes()) if f.area == "worker qwen3"]
    assert (
        qwen.level == "fail" and "venv is not set up" in qwen.summary and "uv sync --locked" in (qwen.next_step or "")
    )


def test_the_engine_pin_is_read_from_the_store_s10_1(ready: Path, config_path: Path, platform: StandInPlatform) -> None:
    admin, _ = make_admin(config_path, platform)
    (unused,) = [f for f in diagnose(admin, probes=probes()) if f.area == "engine" and f.level != "info"]
    assert unused.level == "warn" and "not pinned" in unused.summary and "not in this build" in (unused.next_step or "")
    config = load_config(config_path)
    with NarrationStore(config.server.store_root, platform) as store:
        base = store.put_engine_profile(engine_profile())
        store.set_current_engine_profile("base", base.engine_profile_id)
    engine = [f for f in diagnose(admin, probes=probes(engine_module=True)) if f.area == "engine"]
    assert [f.level for f in engine] == ["warn", "warn"]
    assert "tier not measured, no canary" in engine[0].summary and "no design engine profile" in engine[1].summary


def test_json_output_lists_the_findings_s7_1(ready: Path, config_path: Path, platform: StandInPlatform) -> None:
    admin, out = make_admin(config_path, platform)
    code = run(admin, as_json=True, probes_=probes(gpu=no_gpu))
    data = json.loads(out.getvalue())
    assert code == EXIT_FAILED and data["ok"] is False
    assert {"area": "gpu", "level": "fail"}.items() <= next(f for f in data["findings"] if f["area"] == "gpu").items()


def test_doctor_is_a_registered_command_s7_1(admin: Callable[..., object], config_path: Path) -> None:
    ran = admin("--config", str(config_path), "doctor", "--quick")
    assert "platform" in ran.out and "config" in ran.out  # pyright: ignore[reportAttributeAccessIssue]


def test_the_venv_check_without_uv_is_unknown_s4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    assert doctor.uv_sync_check(tmp_path).synced is None


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
def test_the_venv_check_asks_uv_about_this_repositorys_own_venv_s4() -> None:
    """The server project this test runs from was synced by uv, so uv says it matches its lock."""
    root = Path(__file__).resolve().parents[2]
    check = doctor.uv_sync_check(root)
    assert check.synced is True, check.detail
