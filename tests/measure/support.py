"""Support for the ``measure_voice`` tests: a store with a pinned engine and an unmeasured voice, fake workers,
the job engine with the ``measure`` handler registered, and a calibration corpus the tests control.

The corpus is the service's own (``material/calibration/narration-en.v1``), copied into the test's folder with
a ``design_text`` paragraph added (the service's DC-13 design text, which gate H1's freeze adds to the real
set), and a manifest that names its bytes. No text here comes from a caller's script. Nothing needs a GPU or a
model: every worker is ``narration_worker``'s ``fake`` role, which hears back what it rendered and, through
its spec's ``transcripts``, the clip the tests clone from.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from narration import keys
from narration.config import Config, MeasurementConfig, VoiceDesignConfig
from narration.contracts.models import JobRecord, Progress, ProvenanceEntry
from narration.contracts.names import CORPUS, Priority
from narration.jobs.core import EngineParts
from narration.jobs.engine import JobEngine
from narration.jobs.runner import EngineRunner
from narration.jobs.voice import PathCheck
from narration.measure import build_measure_handler, measure_registry
from narration.measure.corpus import CALIBRATION_KIND, DESIGN_TEXT, MANIFEST, PARAGRAPHS, default_material_root
from narration.measure.handler import MeasureHandler
from narration.post import DeliveryPipeline
from narration.qa import Scorer
from narration.store import NarrationStore
from narration.store.store import utc_iso
from narration.text import TextPipeline
from tests.jobs.support import (
    DESIGN_ID,
    ENGINE_ID,
    METHOD_ID,
    VOICE_TRANSCRIPT,
    FakePool,
    Host,
    MonotonicClock,
    TestAligner,
    drive,
    engine_profile,
    qa_pins,
    submit,
    write_clip,
)
from tests.store.standin import StandInPlatform

SHORT_LADDER: Final = (80, 150, 350)
"""Two rungs in the trend band and one above it: every rule of the ladder, and quick on the fake."""
SHORT_CALIBRATION: Final = ("cal-02",)
"""The corpus paragraphs the tests' calibration set keeps, beside the design text."""
DESIGN_SEGMENT: Final = "cal-design"


# ======================================================================== the corpus


def corpus_data(*, design: bool = True, calibration: Sequence[str] | None = None) -> dict[str, Any]:
    """The service's calibration corpus as data, with the design text paragraph when ``design``, and only the
    calibration paragraphs named in ``calibration`` (all of them when None)."""
    source = default_material_root() / CALIBRATION_KIND / CORPUS / PARAGRAPHS
    data: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    if calibration is not None:
        data["calibration"] = [p for p in data["calibration"] if p["segment_id"] in calibration]
    if design:
        text = VoiceDesignConfig().design_text
        data[DESIGN_TEXT] = {"segment_id": DESIGN_SEGMENT, "spoken_chars": len(text), "cues": [{"text": text}]}
    return data


def write_corpus(root: Path, data: Mapping[str, Any], *, status: str = "draft", set_id: str = CORPUS) -> Path:
    """Write a calibration corpus set under ``root`` (a material root), with its manifest; returns ``root``."""
    folder = root / CALIBRATION_KIND / set_id
    folder.mkdir(parents=True, exist_ok=True)
    body = json.dumps(dict(data), ensure_ascii=False, indent=1).encode("utf-8")
    (folder / PARAGRAPHS).write_bytes(body)
    manifest = {
        "set": set_id,
        "kind": CALIBRATION_KIND,
        "version": 1,
        "status": status,
        "files": [{"path": PARAGRAPHS, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}],
    }
    (folder / MANIFEST).write_bytes(json.dumps(manifest, indent=1).encode("utf-8"))
    return root


def paragraph_text(data: Mapping[str, Any], segment_id: str) -> str:
    """A corpus paragraph's cues joined, by its segment id (the design text's included)."""
    for group in ("calibration", "ladder"):
        for p in data[group]:
            if p["segment_id"] == segment_id:
                return " ".join(c["text"] for c in p["cues"])
    design = data.get(DESIGN_TEXT)
    if design is not None and design["segment_id"] == segment_id:
        return " ".join(c["text"] for c in design["cues"])
    raise KeyError(segment_id)


# ======================================================================== the fake's spec


def write_spec(path: Path, *faults: Mapping[str, Any], transcripts: Mapping[str, str] | None = None) -> None:
    """The fake worker's spec: planted faults, and the transcripts of files it did not render (by sha256)."""
    tmp = path.with_suffix(".tmp")
    body = {"faults": [dict(f) for f in faults], "transcripts": dict(transcripts or {})}
    tmp.write_text(json.dumps(body), encoding="utf-8")
    os.replace(tmp, path)
    stamp = time.time_ns() + 1_000_000  # a new mtime even on a coarse clock, so the change is seen
    os.utime(path, ns=(stamp, stamp))


# ======================================================================== the world


def measure_request(clip: Path, clip_sha256: str, transcript: str = VOICE_TRANSCRIPT) -> dict[str, Any]:
    """A ``measure_voice`` request, as its input schema has it."""
    return {"voice": {"path": str(clip), "sha256": clip_sha256, "transcript": transcript}}


def generate_request(clip: Path, clip_sha256: str, *texts: str, **options: Any) -> dict[str, Any]:
    """A ``submit_job`` request with one segment per text."""
    body: dict[str, Any] = {
        "voice": {"path": str(clip), "sha256": clip_sha256, "transcript": VOICE_TRANSCRIPT},
        "segments": [{"segment_id": f"p{i:02d}", "text": t} for i, t in enumerate(texts, start=1)],
    }
    if options:
        body["options"] = dict(options)
    return body


def submit_measure(store: NarrationStore, body: Mapping[str, Any], priority: Priority = "batch") -> JobRecord:
    """Queue a ``measure`` job as the front-end would (WP36): the request by value, the priority on the job
    (``measure_voice``'s input has no options)."""
    now = utc_iso(time.time())
    record = JobRecord(
        job_id=keys.Keys().new_job_id(),
        kind="measure",
        request=dict(body),
        request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest(),
        label=None,
        priority=priority,
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
class MeasureWorld:
    """Everything a measure test drives."""

    root: Path
    config: Config
    store: NarrationStore
    host: Host
    engine: JobEngine
    handler: MeasureHandler
    runner: EngineRunner
    clip: Path
    clip_sha256: str
    spec: Path
    material: Path
    corpus: dict[str, Any]

    @property
    def pool(self) -> FakePool:
        assert isinstance(self.host.workers, FakePool)
        return self.host.workers

    def measure(self, *, transcript: str = VOICE_TRANSCRIPT, priority: Priority = "batch") -> JobRecord:
        return submit_measure(self.store, measure_request(self.clip, self.clip_sha256, transcript), priority)

    def generate(self, *texts: str, **options: Any) -> JobRecord:
        return submit(self.store, generate_request(self.clip, self.clip_sha256, *texts, **options))

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

    def restart(self) -> None:
        """A new daemon over the same store and workers: a new engine, handler and runner, nothing in memory."""
        self.host.stop_mode = None
        self.runner.registry.close()
        self.engine = JobEngine(self.config, self.engine.parts)
        self.handler = build_measure_handler(self.engine, material_root=self.material)
        self.runner = EngineRunner(measure_registry(self.engine, self.handler))

    def faults(self, *faults: Mapping[str, Any], heard: str = VOICE_TRANSCRIPT) -> None:
        write_spec(self.spec, *faults, transcripts={self.clip_sha256: heard})

    def text(self, segment_id: str) -> str:
        return paragraph_text(self.corpus, segment_id)

    def close(self) -> None:
        self.runner.registry.close()
        self.pool.close()
        self.store.close()


def make_world(
    root: Path,
    *,
    ladder: Sequence[int] = SHORT_LADDER,
    seeds: int = 2,
    corpus: Mapping[str, Any] | None = None,
    status: str = "draft",
    material: Path | None = None,
    check_path: PathCheck | None = None,
) -> MeasureWorld:
    """A store with a pinned engine and a designed clip that is not measured yet, and the runner over the job
    engine and the ``measure`` handler. The engine is built as the daemon builds it: with no path check of its
    own (``check_path`` None), so a caller's clip is read through the host platform's check."""
    store_root, models_root = root / "store", root / "models"
    (store_root / "scratch").mkdir(parents=True, exist_ok=True)
    base = Config.for_tests(store_root, models_root)
    config = dataclasses.replace(
        base, measurement=MeasurementConfig(seeds=seeds, length_ladder_spoken_chars=tuple(ladder))
    )
    data = dict(corpus) if corpus is not None else corpus_data(calibration=SHORT_CALIBRATION)
    material_root = material if material is not None else write_corpus(root / "material", data, status=status)
    store = NarrationStore(store_root, StandInPlatform(), alignment_method_id=METHOD_ID)
    store.put_engine_profile(engine_profile(models_root))
    store.set_current_engine_profile("base", ENGINE_ID)
    clip = root / "voice" / "clip.wav"
    clip_sha256 = write_clip(clip)
    store.add_provenance(ProvenanceEntry(clip_sha256=clip_sha256, design_id=DESIGN_ID, date=utc_iso(time.time())))
    spec = root / "fake-spec.json"
    write_spec(spec, transcripts={clip_sha256: VOICE_TRANSCRIPT})
    clock = MonotonicClock()
    host = Host(store=store, config=config, workers=FakePool(config, spec), clock=clock)
    parts = EngineParts(
        text=TextPipeline(config.text),
        delivery=DeliveryPipeline(fade_s=config.delivery.fade_s),
        scorer=Scorer(config.measurement),
        aligner=TestAligner(),
        qa_pins=qa_pins(models_root),
        check_path=check_path,
        clock=clock,
        defer_s=0.05,
    )
    engine = JobEngine(config, parts)
    handler = build_measure_handler(engine, material_root=material_root)
    return MeasureWorld(
        root=root,
        config=config,
        store=store,
        host=host,
        engine=engine,
        handler=handler,
        runner=EngineRunner(measure_registry(engine, handler)),
        clip=clip,
        clip_sha256=clip_sha256,
        spec=spec,
        material=material_root,
        corpus=data,
    )
