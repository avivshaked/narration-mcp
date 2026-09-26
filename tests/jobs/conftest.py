"""Fixtures for the job engine's tests: a store with a pinned engine and a measured voice, fake workers, and
the engine and runner over them. No GPU and no model: every worker is the fake role."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from narration.config import Config
from narration.contracts.models import JobRecord, ProvenanceEntry
from narration.contracts.names import JobKind
from narration.jobs.core import EngineParts
from narration.jobs.engine import JobEngine
from narration.jobs.gpu import VramProbe
from narration.jobs.runner import EngineRunner
from narration.post import DeliveryPipeline
from narration.qa import Scorer
from narration.store import NarrationStore
from narration.store.store import utc_iso
from narration.text import TextPipeline
from narration.workers import SubprocessWorkerClient, worker_command
from tests.store.standin import StandInPlatform

from .support import (
    DESIGN_ID,
    ENGINE_ID,
    METHOD_ID,
    FakePool,
    Host,
    MonotonicClock,
    TestAligner,
    drive,
    engine_profile,
    measurement,
    qa_pins,
    request,
    submit,
    write_clip,
    write_spec,
)


@pytest.fixture(scope="session")
def anchor(tmp_path_factory: pytest.TempPathFactory) -> tuple[float, ...]:
    """The test voice's anchor: the fake's speaker embedding of the clip itself, which every take cloned
    from the clip sits near (about 0.99)."""
    root = tmp_path_factory.mktemp("anchor") / "store"
    clip = root / "scratch" / "clip.wav"
    write_clip(clip)
    config = Config.for_tests(root, root.parent / "models")
    pins = qa_pins(root.parent / "models")
    with SubprocessWorkerClient(worker_command(config, "fake")) as client:
        client.start()
        client.request("load", pins.load_payload("cpu"), timeout_s=60.0)
        reply = client.request("embed", {"wav": str(clip), "device": "cpu"}, timeout_s=60.0)
    return tuple(float(v) for v in reply["embedding"])


@dataclass
class World:
    """Everything a test drives: the store, the host (with the fake workers), the engine and the runner."""

    root: Path
    config: Config
    store: NarrationStore
    host: Host
    engine: JobEngine
    runner: EngineRunner
    clip: Path
    clip_sha256: str
    spec: Path
    anchor: tuple[float, ...]

    @property
    def pool(self) -> FakePool:
        assert isinstance(self.host.workers, FakePool)
        return self.host.workers

    def request(self, *texts: str, **options: Any) -> dict[str, Any]:
        return request(self.clip, self.clip_sha256, *texts, **options)

    def submit(self, *texts: str, kind: JobKind = "generate", **options: Any) -> JobRecord:
        return submit(self.store, self.request(*texts, **options), kind=kind)

    def submit_body(self, body: Mapping[str, Any], *, kind: JobKind = "generate") -> JobRecord:
        return submit(self.store, body, kind=kind)

    def job(self, job_id: str) -> JobRecord:
        job = self.store.get_job(job_id)
        assert job is not None
        return job

    def run(self, *, max_steps: int = 400) -> int:
        return drive(self.runner, self.host, max_steps=max_steps)

    def faults(self, *faults: Mapping[str, Any]) -> None:
        write_spec(self.spec, *faults)

    def measure(self, **options: Any) -> None:
        """Replace the voice's measurement (e.g. a shorter reliable length), under a new measurement key."""
        self.store.put_measurement(measurement(self.clip_sha256, self.anchor, corpus_hex="1" * 64, **options))

    def new_engine(self, *, probe: VramProbe | None = None, **parts: Any) -> None:
        """Rebuild the engine and runner (another probe, another guard)."""
        base = self.engine.parts
        changes: dict[str, Any] = dict(parts)
        if probe is not None:
            changes["probe"] = probe
        self.engine = JobEngine(self.config, dataclasses.replace(base, **changes))
        self.runner = EngineRunner(self.engine)


def make_world(root: Path, anchor: tuple[float, ...], *, holder: str | None = None) -> World:
    store_root = root / "store"
    models_root = root / "models"
    (store_root / "scratch").mkdir(parents=True, exist_ok=True)
    config = Config.for_tests(store_root, models_root)
    spec = root / "fake-spec.json"
    write_spec(spec)
    store = NarrationStore(store_root, StandInPlatform(), alignment_method_id=METHOD_ID)
    if store.get_engine_profile(ENGINE_ID) is None:
        store.put_engine_profile(engine_profile(models_root))
        store.set_current_engine_profile("base", ENGINE_ID)
    clip = root / "voice" / "clip.wav"
    clip_sha256 = write_clip(clip)
    store.add_provenance(ProvenanceEntry(clip_sha256=clip_sha256, design_id=DESIGN_ID, date=utc_iso(time.time())))
    store.put_measurement(measurement(clip_sha256, anchor))
    clock = MonotonicClock()
    host = Host(store=store, config=config, workers=FakePool(config, spec), clock=clock)
    if holder is not None:
        host.holder = holder
    parts = EngineParts(
        text=TextPipeline(config.text),
        delivery=DeliveryPipeline(fade_s=config.delivery.fade_s),
        scorer=Scorer(config.measurement),
        aligner=TestAligner(),
        qa_pins=qa_pins(models_root),
        clock=clock,
        defer_s=0.05,
    )
    engine = JobEngine(config, parts)
    return World(
        root=root,
        config=config,
        store=store,
        host=host,
        engine=engine,
        runner=EngineRunner(engine),
        clip=clip,
        clip_sha256=clip_sha256,
        spec=spec,
        anchor=anchor,
    )


@pytest.fixture
def world(tmp_path: Path, anchor: tuple[float, ...]) -> Iterator[World]:
    w = make_world(tmp_path, anchor)
    try:
        yield w
    finally:
        w.pool.close()
        w.store.close()
