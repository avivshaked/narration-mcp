"""Support for the backend's tests: a platform with the section 17.3 path check, a daemon launcher that
records what it was asked, and the backend over the job engine's test world (``tests.jobs``).

Nothing here needs a GPU or a model: the job engine runs the fake worker. Every text is invented for these
tests (design section 9.3).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from narration import keys
from narration.backend import AnalysisPins, NarrationBackend
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Store
from narration.contracts.models import DaemonStatus, EngineProfile, GpuStatus, MeasurementRecord
from narration.jobs.plan import VoiceSpec
from narration.qa import Scorer
from tests.jobs.support import METHOD_ID, qa_pins
from tests.store.standin import StandInPlatform

GIB = 1024**3


class TestPlatform(StandInPlatform):
    """The store's stand-in platform, plus section 17.3's path check as the tests need it on any OS (the
    real one is Windows only, tested in ``tests/platform``) and a settable free disk."""

    __test__ = False

    def __init__(self) -> None:
        super().__init__()
        self.free_bytes = 100 * GIB
        self.read_paths: list[str] = []

    def check_readable_path(self, path: str) -> Path:
        self.read_paths.append(path)
        if path.startswith(("\\\\", "//")):
            raise NarrationError(codes.PATH_NOT_ALLOWED, "a network or device path", details={"rule": "network"})
        candidate = Path(path)
        if not candidate.is_absolute():
            raise NarrationError(codes.PATH_NOT_ALLOWED, "the path is not absolute", details={"rule": "relative"})
        if not candidate.is_file():
            raise NarrationError(codes.PATH_NOT_ALLOWED, "there is no file at the path", details={"rule": "not_found"})
        return candidate.resolve()

    def free_disk_bytes(self, path: Path) -> int:
        return self.free_bytes


@dataclass
class FakeLauncher:
    """A ``DaemonLauncher`` that starts nothing: it records each ``ensure`` with whether the job was already
    in the store, and reports ``status`` as the running daemon. ``on_ensure`` runs at each ``ensure``."""

    store: Store | None = None
    status: DaemonStatus | None = None
    ensured: list[int] = field(default_factory=list)
    """The number of jobs in the store at each ``ensure``."""
    error: NarrationError | None = None
    on_ensure: Callable[[], None] | None = None

    def running(self, store: Store) -> DaemonStatus | None:
        return self.status

    def ensure(self, store: Store) -> None:
        self.ensured.append(len(store.queued_jobs()))
        if self.error is not None:
            raise self.error
        if self.on_ensure is not None:
            self.on_ensure()


def daemon_status(*, state: str = "idle", holder: Any = None, job_id: str | None = None) -> DaemonStatus:
    """A daemon's status as ``run/daemon.json`` has it."""
    from narration.contracts.models import CurrentJob

    current = (
        CurrentJob(job_id=job_id, kind="generate", label=None, phase="rendering", started_at="2026-09-27T00:00:00Z")
        if job_id
        else None
    )
    return DaemonStatus(
        state=state,  # pyright: ignore[reportArgumentType]
        pid=os.getpid(),
        started_at="2026-09-27T00:00:00Z",
        workers=(),
        current_job=current,
        gpu=GpuStatus(
            name="Test GPU", total_mb=24_000, free_mb=20_000, in_use=holder is not None, holder=holder, unload_in_s=None
        ),
        est_drain_s=12.0,
        updated_at="2026-09-27T00:00:00Z",
    )


class FakeMeasurements:
    """The ``Measurements`` seam over the store alone: any stored measurement is current. The rule that decides
    whether a stored measurement is current (its key) is WP33's and is tested there."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def voice_hash(self, voice: VoiceSpec) -> str:
        return keys.voice_hash(
            model=names.MODEL_QWEN_BASE,
            clip_sha256=voice.sha256,
            transcript=voice.transcript,
            language=names.LANGUAGE,
            x_vector_only_mode=False,
        )

    def require(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord:
        found = self.current(voice_hash, profile)
        if found is None:
            raise NarrationError(codes.VOICE_NOT_MEASURED, "not measured", field="voice")
        return found

    def current(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord | None:
        return self.store.get_measurement(voice_hash, profile.engine_profile_id)


def pins_for(models_root: Path) -> AnalysisPins:
    """The analysis pins the job engine's test world scores with (the fake worker's models)."""
    pins = qa_pins(models_root)
    scorer = Scorer()
    return AnalysisPins(
        asr_model=pins.asr.name,
        sv_model=pins.sv.name,
        aligner_method_id=METHOD_ID,
        qa_profile=scorer.profile_version,
        number_reader=scorer.number_reader,
    )


def make_backend(
    world: Any, *, platform: TestPlatform | None = None, launcher: FakeLauncher | None = None, pins: bool = True
) -> tuple[NarrationBackend, TestPlatform, FakeLauncher]:
    """The backend over a ``tests.jobs`` world's store and config."""
    platform = platform or TestPlatform()
    launcher = launcher or FakeLauncher(store=world.store)
    found = pins_for(world.config.server.models_root) if pins else None
    backend = NarrationBackend(
        world.config,
        world.store,
        platform,
        launcher=launcher,
        pins=lambda: found,
        measurements=FakeMeasurements(world.store),
        poll_s=0.02,
    )
    return backend, platform, launcher


def write_wav(path: Path, *, seconds: float = 2.0, rate: int = 24_000, freq: float = 200.0) -> str:
    """A short synthetic WAV (a tone); returns its sha256."""
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    tone = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), tone, rate, subtype="PCM_16")
    return sha256_of(path)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
