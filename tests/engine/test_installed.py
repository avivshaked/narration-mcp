"""The job engine the daemon builds at its first job (``narration.jobs.runner.installed_engine``; plan.md WP31,
WP32): the aligner, the QA pins and the engine guard, assembled from the config and the pins."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from narration.align import CtcAligner
from narration.config import AlignmentConfig, Config, GpuConfig
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.engine.canary import CanaryGuard
from narration.engine.models import CTC_ALIGNER, WAVLM_SV, WHISPER
from narration.engine.qa import QA_VRAM_NEED_MB, aligner_method_id, qa_pins
from narration.jobs.engine import JobEngine
from narration.jobs.gpu import NoProbe, NvmlProbe
from narration.jobs.handlers import Registry
from narration.jobs.host import RunnerHost
from narration.jobs.runner import installed_engine
from narration.measure import KIND as MEASURE_KIND
from narration.measure import MeasureHandler

from .support import Install, make_install


@dataclass
class _Host:
    """What ``installed_engine`` reads of the daemon's host: its config."""

    config: Config


def _engine(registry: Registry) -> JobEngine:
    handler = registry.handler("generate")
    assert isinstance(handler, JobEngine)
    return handler


@pytest.fixture
def install(tmp_path: Path) -> Install:
    return make_install(tmp_path)


def test_the_installed_engine_runs_generate_and_analyse_with_the_pinned_parts_s4(install: Install) -> None:
    registry = installed_engine(cast(RunnerHost, _Host(config=install.config)))

    assert isinstance(registry, Registry)
    engine = _engine(registry)
    assert registry.handler("analyse") is engine
    assert registry.residency is engine.residency
    parts = engine.parts
    assert isinstance(parts.aligner, CtcAligner)
    assert (parts.aligner.model, parts.aligner.revision) == (names.MODEL_ALIGNER, CTC_ALIGNER.revision)
    assert parts.qa_pins.asr.name == f"{names.MODEL_ASR}@{WHISPER.revision}"
    assert parts.qa_pins.sv.name == f"{names.MODEL_SV}@{WAVLM_SV.revision}"
    assert Path(parts.qa_pins.sv.snapshot_dir) == WAVLM_SV.snapshot_dir(install.models_root)
    assert parts.qa_pins.vram_need_mb == QA_VRAM_NEED_MB
    assert engine.residency.need_mb["qa"] == QA_VRAM_NEED_MB
    assert isinstance(parts.guard, CanaryGuard) and parts.guard.sv == parts.qa_pins.sv
    assert isinstance(parts.probe, NvmlProbe)  # [gpu] device is cuda:0 by default
    assert parts.check_path is None  # a caller's clip is checked by the daemon's platform (host.platform)


def test_the_analysis_licences_are_the_pinned_models_s18(install: Install) -> None:
    licence = qa_pins(install.config).licence
    assert (licence.aligner, licence.asr, licence.sv) == ("apache-2.0", "per model card", "per model card")


def test_the_aligner_takes_its_thresholds_from_the_config_s11_2(install: Install) -> None:
    config = dataclasses.replace(
        install.config, alignment=AlignmentConfig(unplaced_below=0.4, low_confidence_below=0.6)
    )
    registry = installed_engine(cast(RunnerHost, _Host(config=config)))
    aligner = _engine(registry).parts.aligner
    assert isinstance(aligner, CtcAligner)
    assert aligner.method_id == aligner_method_id(config)
    assert aligner.method_id != aligner_method_id(install.config)  # the thresholds are in the method id


def test_no_nvml_check_on_a_cpu_device_s4(install: Install) -> None:
    config = dataclasses.replace(install.config, gpu=GpuConfig(device="cpu"))
    registry = installed_engine(cast(RunnerHost, _Host(config=config)))
    assert isinstance(_engine(registry).parts.probe, NoProbe)


@pytest.mark.parametrize("model", [WHISPER, WAVLM_SV, CTC_ALIGNER])
def test_a_qa_model_not_installed_is_backend_not_installed_s14(install: Install, model: Any) -> None:
    folder = model.snapshot_dir(install.models_root)
    for path in sorted(folder.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    folder.rmdir()
    with pytest.raises(NarrationError) as caught:
        installed_engine(cast(RunnerHost, _Host(config=install.config)))
    error = caught.value
    assert error.code == codes.BACKEND_NOT_INSTALLED and "narration-admin install" in error.hint
    assert error.details is not None and error.details["repo"] == model.repo


def test_an_aligner_the_service_does_not_pin_is_backend_not_installed_s11_2(install: Install) -> None:
    config = dataclasses.replace(install.config, alignment=AlignmentConfig(model="someone/aligner"))
    with pytest.raises(NarrationError) as caught:
        installed_engine(cast(RunnerHost, _Host(config=config)))
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED and "[alignment] model" in caught.value.hint


def test_a_pinned_aligner_the_ctc_method_cannot_run_is_backend_not_installed_s11_2(install: Install) -> None:
    config = dataclasses.replace(install.config, alignment=AlignmentConfig(model=names.MODEL_QWEN_BASE))
    with pytest.raises(NarrationError) as caught:
        installed_engine(cast(RunnerHost, _Host(config=config)))
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED


def test_measure_joins_the_registry_on_the_job_engine_s4(install: Install) -> None:
    """``measure_voice``'s handler (WP33) is built on the daemon's job engine: it shares the engine's model
    residency, throughput and canary, so one group is resident across every kind."""
    registry = installed_engine(cast(RunnerHost, _Host(config=install.config)))
    engine = _engine(registry)
    measure = registry.handler(MEASURE_KIND)
    assert isinstance(measure, MeasureHandler)
    assert measure.engine is engine and measure.core is engine.core
    assert registry.residency is engine.residency and registry.throughput is engine.throughput
    assert {"generate", "analyse", MEASURE_KIND} <= set(registry.handlers)
