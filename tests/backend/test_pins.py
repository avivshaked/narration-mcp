"""The analysis pins ``backend_for`` plans with (design sections 10.2 and 11.2; plan.md WP36 with WP32).

A plan finds a cached analysis only by its key, which names the QA models and the aligner's method id. So
``narration-mcp``'s backend takes them from the installation, as the daemon's job engine does; with none, a plan
counts every analysis as needed. That a plan with them finds what the installed engine cached is pinned by the
join test (``tests/engine/test_daemon_backend.py``), and that a plan and the engine build a key's inputs alike
by the tests below, which also pin one key's value. No model or GPU is needed: a snapshot is an empty folder.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from narration import keys
from narration.backend import AnalysisPins, InstalledPins, backend_for
from narration.backend.planning import analysis_key_inputs
from narration.config import Config
from narration.contracts.models import (
    CueText,
    ExactWords,
    Hint,
    HintApplied,
    MeasurementRecord,
    SegmentText,
    TakeRecord,
    TextChecksInfo,
)
from narration.engine.models import CTC_ALIGNER, WAVLM_SV, WHISPER
from narration.engine.qa import aligner_method_id
from narration.jobs.core import EngineCore
from narration.jobs.plan import hints_used
from narration.jobs.stages import Stages
from narration.jobs.state import JobRun, SegmentWork
from narration.platform.testing import StandInPlatform
from narration.qa import Scorer
from narration.store import NarrationStore
from tests.store.factories import measurement_record, take_record

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
    assert pins.aligner_revision == CTC_ALIGNER.revision
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
        assert alignment["benchmark"] is None, "no benchmark has run"
        assert (alignment["model"], alignment["revision"]) == (CTC_ALIGNER.repo, CTC_ALIGNER.revision)
        assert f"@{alignment['revision']}" in alignment["method_id"], "the revision the method id names"
    finally:
        store.close()


# ======================================================================== one builder of the key's inputs

FIRST = "Tobin lit the lamps."
SECOND = "The kettle sang at nine."
PINS = AnalysisPins(
    asr_model="example/asr@" + "1" * 40,
    sv_model="example/sv@" + "2" * 40,
    aligner_method_id="ctc-test/v1",
    qa_profile="qa-test/v1",
    number_reader="numbers-test/v1",
)
PINNED_KEY = "sha256:d333b07f87a9535b024b2e59072739456dd86680f440cd97e26a18a4eb509301"
"""``analysis_key`` of ``segment()``'s take under ``PINS``, as the code computed it before a plan and the engine
shared one builder (WP36, F5). A change here changes every cached analysis's key."""


def segment() -> SegmentText:
    """Two cues: a hint applied in each, and an exact span (words 3 to 5) in the second."""
    second_at = len(FIRST) + 1
    return SegmentText(
        segment_id="p01",
        cues=(
            CueText(
                index=0,
                received=FIRST,
                spoken=FIRST,
                engine=FIRST,
                spoken_span=(0, len(FIRST)),
                engine_span=(0, len(FIRST)),
                hints_applied=(HintApplied(term="Tobin", respell="TOH-bin", offset=0),),
            ),
            CueText(
                index=1,
                received=SECOND,
                spoken=SECOND,
                engine=SECOND,
                spoken_span=(second_at, second_at + len(SECOND)),
                engine_span=(second_at, second_at + len(SECOND)),
                hints_applied=(HintApplied(term="kettle", respell=None, offset=4),),
                exact=(ExactWords(start=16, end=23, words=(3, 5)),),
            ),
        ),
        spoken_text=f"{FIRST} {SECOND}",
        engine_text=f"{FIRST} {SECOND}",
        spoken_chars=second_at + len(SECOND),
        text_checks=TextChecksInfo(version="text-checks/test", rules_sha256="0" * 64),
    )


def used_hints(text: SegmentText) -> tuple[Hint, ...]:
    """The hints the segment uses; the third applies in neither cue."""
    hints = (
        Hint(term="Tobin", respell="TOH-bin", asr_aliases=("Toby", "Tobbin"), align_as="Toe bin"),
        Hint(term="kettle", asr_aliases=("kettel",)),
        Hint(term="stove"),
    )
    return hints_used(text, hints)


def take() -> TakeRecord:
    record = take_record("a" * 64, "rn_" + "0" * 16)
    return dataclasses.replace(record, delivery=dataclasses.replace(record.delivery, sha256="b" * 64))


def measured() -> MeasurementRecord:
    return dataclasses.replace(measurement_record(), measurement_key="sha256:" + "3" * 64)


def test_a_plan_and_the_engine_build_the_same_key_inputs_s10_2() -> None:
    text, hints, record = segment(), used_hints(segment()), measured()
    planned = analysis_key_inputs(take=take(), text=text, hints=hints, pins=PINS, measurement=record)
    parts = SimpleNamespace(
        scorer=SimpleNamespace(profile_version=PINS.qa_profile, number_reader=PINS.number_reader),
        text=SimpleNamespace(checks_info=TextChecksInfo(version="not used: the segment has its own", rules_sha256="")),
        qa_pins=SimpleNamespace(asr=SimpleNamespace(name=PINS.asr_model), sv=SimpleNamespace(name=PINS.sv_model)),
        aligner=SimpleNamespace(method_id=PINS.aligner_method_id),
    )
    stages = Stages(cast(EngineCore, SimpleNamespace(parts=parts)))
    run = cast(JobRun, SimpleNamespace(measurement=record, job_id="job_test"))
    engine = stages.key_inputs(run, SegmentWork(index=0, text=text, hints=hints, est_s=1.0), take())
    assert planned == engine
    assert planned.exact_spans == ((1, 3, 5),) and [h[0] for h in planned.hints_qa] == ["Tobin", "kettle"]


def test_the_analysis_key_of_a_known_take_is_unchanged_s10_2() -> None:
    inputs = analysis_key_inputs(
        take=take(), text=segment(), hints=used_hints(segment()), pins=PINS, measurement=measured()
    )
    assert keys.analysis_key(inputs) == PINNED_KEY
    with_revision = dataclasses.replace(PINS, aligner_revision="3" * 40)
    same = analysis_key_inputs(
        take=take(), text=segment(), hints=used_hints(segment()), pins=with_revision, measurement=measured()
    )
    assert keys.analysis_key(same) == PINNED_KEY, "the aligner's revision enters no key beyond its method id"
