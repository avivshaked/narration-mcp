"""The analysis pins ``backend_for`` plans with (design sections 10.2 and 11.2; plan.md WP36 with WP32).

A plan finds a cached analysis only by its key, which names the QA models and the aligner's method id. So
``narration-mcp``'s backend takes them from the installation, as the daemon's job engine does; with none, a plan
counts every analysis as needed. That a plan with them finds what the installed engine cached is pinned by the
join test (``tests/engine/test_daemon_backend.py``). No model or GPU is needed: a snapshot is an empty folder.
"""

from __future__ import annotations

from pathlib import Path

from narration.backend import InstalledPins, backend_for
from narration.config import Config
from narration.engine.models import CTC_ALIGNER, WAVLM_SV, WHISPER
from narration.engine.qa import aligner_method_id
from narration.platform.testing import StandInPlatform
from narration.qa import Scorer
from narration.store import NarrationStore

from .support import FakeLauncher


def config_in(root: Path) -> Config:
    return Config.for_tests(root / "store", root / "models")


def install_qa(config: Config) -> None:
    """The QA group's snapshot folders at the service's pinned revisions."""
    for model in (WHISPER, WAVLM_SV, CTC_ALIGNER):
        model.snapshot_dir(config.server.models_root).mkdir(parents=True, exist_ok=True)


def test_the_pins_name_the_installed_qa_models_and_aligner_s10_2(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    install_qa(config)
    pins = InstalledPins(config)()
    assert pins is not None
    assert pins.asr_model == f"{WHISPER.repo}@{WHISPER.revision}"
    assert pins.sv_model == f"{WAVLM_SV.repo}@{WAVLM_SV.revision}"
    assert pins.aligner_method_id == aligner_method_id(config)
    scorer = Scorer(config.measurement)
    assert (pins.qa_profile, pins.number_reader) == (scorer.profile_version, scorer.number_reader)


def test_without_the_qa_models_there_are_no_pins_until_they_are_installed_s10_2(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    pins = InstalledPins(config)
    assert pins() is None, "a plan then counts every analysis as needed"
    install_qa(config)
    found = pins()
    assert found is not None and pins() is found


def test_backend_for_reports_the_configured_aligner_through_its_pins_r1(tmp_path: Path) -> None:
    config = config_in(tmp_path)
    install_qa(config)
    config.server.store_root.mkdir(parents=True)
    store = NarrationStore(config.server.store_root, StandInPlatform())
    try:
        backend = backend_for(config, store, StandInPlatform(), launcher=FakeLauncher())
        alignment = backend.get_server_status_sync()["alignment"]
        assert alignment["method_id"] == aligner_method_id(config)
    finally:
        store.close()
