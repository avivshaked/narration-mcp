"""Support for the DESIGN step's tests (``design_voice``, ``profile_voice``): a store with a pinned Base and a pinned
VoiceDesign engine, the fake workers, and the runner over the job engine with the ``measure``, ``design`` and
``profile`` handlers registered, as the daemon has them.

Nothing here needs a GPU or a model: every worker is ``narration_worker``'s ``fake`` role, which designs a
deterministic synthetic voice for any description and seed, hears back what it rendered, and profiles any WAV.
Every description and text is invented for these tests (design section 9.3), or is the service's own design text
(``[voice_design] design_text``, section 16).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from narration import keys
from narration.config import Config, MeasurementConfig
from narration.contracts import names
from narration.contracts.models import EngineProfile, JobRecord, Progress
from narration.contracts.names import CanaryStatus, DeterminismTier, JobKind
from narration.design import design_registry
from narration.jobs.core import EngineParts
from narration.jobs.engine import JobEngine
from narration.jobs.hooks import EngineGuard, NoGuard
from narration.jobs.host import RunnerHost
from narration.jobs.runner import EngineRunner
from narration.measure import build_measure_handler, measure_registry
from narration.post import DeliveryPipeline
from narration.qa import Scorer
from narration.store import NarrationStore
from narration.store.store import utc_iso
from narration.text import TextPipeline
from tests.jobs.support import (
    ENGINE_ID,
    METHOD_ID,
    FakePool,
    Host,
    MonotonicClock,
    TestAligner,
    drive,
    engine_profile,
    qa_pins,
    snapshot,
)
from tests.measure.support import SHORT_CALIBRATION, SHORT_LADDER, corpus_data, write_corpus, write_spec
from tests.store.standin import StandInPlatform

DESIGN_ENGINE_ID: Final = "qwen3-design-1.7b.test"
DESIGN_REVISION: Final = "2" * 40
DESIGN_HASH: Final = names.HASH_PREFIX + hashlib.sha256(b"narration design tests: VoiceDesign engine").hexdigest()

WARM: Final = "A warm, low narrator with an unhurried, even pace and a clear, smooth tone."
"""An invented positive-only description."""
NEGATED: Final = "A bright storyteller, not theatrical and never rushed, with no rasp."
"""An invented description with negations: linted, and still designed (section 3.5)."""
MARSH: Final = "Herons wait in the marsh at first light, and the reeds lean gently toward the water."
"""An invented design text."""


def design_profile(models_root: Path, *, tier: DeterminismTier | None = None) -> EngineProfile:
    """A pinned VoiceDesign engine profile, as WP32 records it: every audio-changing setting, ``non_streaming_mode``
    true (section 10.1)."""
    base = engine_profile(models_root)
    return dataclasses.replace(
        base,
        engine_profile_id=DESIGN_ENGINE_ID,
        hash=DESIGN_HASH,
        model_repo=names.MODEL_QWEN_DESIGN,
        model_revision=DESIGN_REVISION,
        snapshot_dir=snapshot(models_root, names.MODEL_QWEN_DESIGN, DESIGN_REVISION),
        settings={**base.settings, "non_streaming_mode": True},
        tier=tier,
    )


class FixedGuard:
    """An engine guard that reports a fixed canary outcome after every Qwen load, and records the profiles it
    was asked about (WP32's canary gate is tested in ``tests/engine``)."""

    def __init__(self, status: CanaryStatus) -> None:
        self.status: CanaryStatus = status
        self.profiles: list[str] = []

    def after_load(self, host: RunnerHost, profile: EngineProfile, hello: Any) -> CanaryStatus:
        self.profiles.append(profile.engine_profile_id)
        return self.status


def design_request(
    *, description: str = WARM, takes: int = 2, design_text: str | None = None, name: str = "harbour narrator"
) -> dict[str, Any]:
    """A ``design`` job's request as the front-end keeps it (``narration.backend.steps``), with a new design id."""
    config_text = Config.for_tests(Path("store")).voice_design.design_text
    return {
        "name": name,
        "description": description,
        "takes": takes,
        "design_text": design_text if design_text is not None else config_text,
        "design_id": keys.Keys().new_design_id(),
    }


def queue(
    store: NarrationStore, kind: JobKind, body: Mapping[str, Any], *, identity: Mapping[str, Any] | None = None
) -> JobRecord:
    """Queue a job as the front-end would: the request by value, its sha256, ``batch`` priority."""
    now = utc_iso(time.time())
    hashed = identity if identity is not None else body
    record = JobRecord(
        job_id=keys.Keys().new_job_id(),
        kind=kind,
        request=dict(body),
        request_sha256=hashlib.sha256(json.dumps(hashed, sort_keys=True).encode("utf-8")).hexdigest(),
        label=str(body.get("name")) if kind == "design" else None,
        priority="batch",
        status="queued",
        phase=None,
        round=0,
        progress=Progress(done_s=0.0, total_s=0.0, fraction=0.0, segments_done=0, segments_total=0),
        outcome=None,
        error=None,
        idempotency_key=None,
        created_at=now,
        updated_at=now,
    )
    job, _ = store.create_job(record)
    return job


@dataclass
class DesignWorld:
    """Everything a design or profile test drives."""

    root: Path
    config: Config
    store: NarrationStore
    host: Host
    engine: JobEngine
    runner: EngineRunner
    spec: Path
    guard: EngineGuard

    @property
    def pool(self) -> FakePool:
        assert isinstance(self.host.workers, FakePool)
        return self.host.workers

    def design(self, **args: Any) -> JobRecord:
        """Queue a ``design`` job (``design_request``'s arguments)."""
        return queue(self.store, "design", design_request(**args))

    def profile(self, path: Path, sha256: str) -> JobRecord:
        """Queue a ``profile`` job for the audio at ``path``."""
        return queue(self.store, "profile", {"audio": {"path": str(path), "sha256": sha256}})

    def measure(self, path: str, sha256: str, transcript: str) -> JobRecord:
        """Queue a ``measure`` job for a voice."""
        return queue(self.store, "measure", {"voice": {"path": path, "sha256": sha256, "transcript": transcript}})

    def job(self, job_id: str) -> JobRecord:
        job = self.store.get_job(job_id)
        assert job is not None
        return job

    def run(self, *, max_steps: int = 2000) -> int:
        return drive(self.runner, self.host, max_steps=max_steps)

    def step_until(self, predicate: Callable[[], bool], *, limit: int = 400) -> None:
        """Step the runner until ``predicate`` holds."""
        for _ in range(limit):
            if predicate():
                return
            self.runner.step(self.host)
        raise AssertionError("the condition never held")

    def faults(self, *faults: Mapping[str, Any], transcripts: Mapping[str, str] | None = None) -> None:
        write_spec(self.spec, *faults, transcripts=transcripts)

    def requests(self, op: str) -> list[dict[str, Any]]:
        """Every request of ``op`` the fake workers were sent, in order."""
        return [payload for _, o, payload in self.pool.requests if o == op]

    def restart(self) -> None:
        """A new daemon over the same store and workers: a new engine and runner, nothing in memory."""
        self.host.stop_mode = None
        self.runner.registry.close()
        self.engine = JobEngine(self.config, self.engine.parts)
        self.runner = EngineRunner(_registry(self.engine, self.root / "material"))

    def close(self) -> None:
        self.runner.registry.close()
        self.pool.close()
        self.store.close()


def _registry(engine: JobEngine, material: Path) -> Any:
    measure = build_measure_handler(engine, material_root=material)
    return design_registry(engine, measure_registry(engine, measure))


def make_world(
    root: Path,
    *,
    guard: EngineGuard | None = None,
    tier: DeterminismTier | None = None,
    pin_design: bool = True,
) -> DesignWorld:
    """A store with a pinned Base and (``pin_design``) VoiceDesign engine, the fake workers, and the runner over the
    job engine with the ``measure``, ``design`` and ``profile`` handlers. The engine is built as the daemon builds
    it: no path check of its own, so a caller's file is read through the host platform's check."""
    store_root, models_root = root / "store", root / "models"
    (store_root / "scratch").mkdir(parents=True, exist_ok=True)
    base = Config.for_tests(store_root, models_root)
    config = dataclasses.replace(
        base, measurement=MeasurementConfig(seeds=2, length_ladder_spoken_chars=tuple(SHORT_LADDER))
    )
    write_corpus(root / "material", corpus_data(calibration=SHORT_CALIBRATION))
    store = NarrationStore(store_root, StandInPlatform(), alignment_method_id=METHOD_ID)
    store.put_engine_profile(engine_profile(models_root))
    store.set_current_engine_profile("base", ENGINE_ID)
    if pin_design:
        store.put_engine_profile(design_profile(models_root, tier=tier))
        store.set_current_engine_profile("design", DESIGN_ENGINE_ID)
    spec = root / "fake-spec.json"
    write_spec(spec)
    clock = MonotonicClock()
    host = Host(store=store, config=config, workers=FakePool(config, spec), clock=clock)
    chosen = guard if guard is not None else NoGuard()
    parts = EngineParts(
        text=TextPipeline(config.text),
        delivery=DeliveryPipeline(fade_s=config.delivery.fade_s),
        scorer=Scorer(config.measurement),
        aligner=TestAligner(),
        qa_pins=qa_pins(models_root),
        guard=chosen,
        clock=clock,
        defer_s=0.05,
    )
    engine = JobEngine(config, parts)
    runner = EngineRunner(_registry(engine, root / "material"))
    return DesignWorld(
        root=root, config=config, store=store, host=host, engine=engine, runner=runner, spec=spec, guard=chosen
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
