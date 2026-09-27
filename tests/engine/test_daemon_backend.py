"""The installed engine behind the daemon and the backend (plan.md WP32 with WP30, WP33 and WP36).

A caller's ``measure_voice`` and ``submit_job`` go through ``narration-mcp``'s backend (``backend_for``), which
starts the daemon; the daemon runs its default job runner (``narration.jobs.runner:default_runner``), which
builds ``installed_engine`` at its first job from a pinned installation: Base and VoiceDesign pinned by
``engine pin`` with their canaries. The daemon runs on a thread of this process, as WP30's tests run it, with
the fake role for every worker group. No GPU and no model: every text is invented for these tests or is the
service's own calibration corpus.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from narration_worker.fake.faults import SPEC_ENV

from narration.backend import NarrationBackend, backend_for
from narration.backend.launch import SERVING_STATES
from narration.config import Config, MeasurementConfig
from narration.contracts import codes
from narration.contracts.interfaces import Store
from narration.contracts.models import DaemonStatus, ProvenanceEntry
from narration.daemon import start as daemon_start
from narration.daemon.service import Daemon
from narration.daemon.settings import DaemonSettings
from narration.daemon.supervisor import FAKE_ROLES, WorkerSupervisor
from narration.engine.models import CTC_ALIGNER
from narration.engine.pinning import pin
from narration.engine.qa import aligner_method_id
from narration.jobs.runner import EngineRunner, default_runner
from narration.measure.handler import MeasureHandler
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore
from narration.store.store import utc_iso
from tests.daemon.conftest import WAIT_S, DaemonHarness
from tests.jobs.support import DESIGN_ID, KETTLE, LAMPS, VOICE_TRANSCRIPT, write_clip
from tests.measure.support import write_spec

from .support import FAKE_WORKER_PACKAGES, CountingStarter, Install, fake_install

MEASURED = MeasurementConfig(seeds=2, length_ladder_spoken_chars=(80, 150, 350))
"""Two seeds and three rungs of the service's corpus: every rule of the ladder, and quick on the fake."""


class ThreadLauncher:
    """Starts the daemon on a thread when none serves the store, as ``ensure_daemon`` does."""

    def __init__(self, start: Callable[[], None]) -> None:
        self._start = start
        self.started = 0

    def running(self, store: Store) -> DaemonStatus | None:
        return daemon_start.running_daemon(store)

    def ensure(self, store: Store) -> None:
        status = daemon_start.running_daemon(store)
        if status is not None and status.state in SERVING_STATES:
            return
        self.started += 1
        self._start()


class Service:
    """A pinned installation, its store, the backend over it, and the daemons that backend started."""

    def __init__(self, root: Path) -> None:
        install: Install = fake_install(root)
        self.config: Config = dataclasses.replace(install.config, measurement=MEASURED)
        self.store = NarrationStore(
            install.store_root, StandInPlatform(), alignment_method_id=aligner_method_id(self.config)
        )
        with CountingStarter(self.config) as starter:
            pin(self.config, self.store, mode="pin", starter=starter, packages=FAKE_WORKER_PACKAGES, device="cpu")
        self.clip = root / "voice" / "clip.wav"
        self.clip_sha256 = write_clip(self.clip)
        self.store.add_provenance(
            ProvenanceEntry(
                clip_sha256=self.clip_sha256, design_id=DESIGN_ID, date=utc_iso(os.path.getmtime(self.clip))
            )
        )
        self.spec = root / "fake-spec.json"
        write_spec(self.spec, transcripts={self.clip_sha256: VOICE_TRANSCRIPT})
        self.harnesses: list[DaemonHarness] = []
        self.runners: list[EngineRunner] = []
        self.launcher = ThreadLauncher(self.start_daemon)
        self.backend: NarrationBackend = backend_for(self.config, self.store, StandInPlatform(), launcher=self.launcher)

    def voice(self) -> dict[str, Any]:
        return {"path": str(self.clip), "sha256": self.clip_sha256, "transcript": VOICE_TRANSCRIPT}

    def start_daemon(self) -> None:
        settings = DaemonSettings(
            store_root=self.config.server.store_root,
            idle_unload_s=60.0,
            idle_exit_s=60.0,
            poll_s=0.05,
            fake_workers=True,
            worker_close_s=5.0,
            stop_now_grace_s=10.0,
            takeover_wait_s=5.0,
            command_wait_s=2.0,
        )
        env = {k: v for k, v in os.environ.items() if k != SPEC_ENV}
        env[SPEC_ENV] = str(self.spec)
        platform = StandInPlatform()
        runner = default_runner()  # the runner the daemon loads by name: installed_engine at the first job
        self.runners.append(runner)

        def supervisor(on_change: Callable[[], None]) -> WorkerSupervisor:
            return WorkerSupervisor(
                self.config, platform, roles=FAKE_ROLES, base_env=env, close_timeout_s=5.0, on_change=on_change
            )

        daemon = Daemon(
            settings=settings,
            config=self.config,
            store=self.store,
            platform=platform,
            runner=runner,
            supervisor_factory=supervisor,
        )
        self.harnesses.append(DaemonHarness(daemon, self.store).start())

    def finished(self, job_id: str) -> dict[str, Any]:
        """``get_job``, long-polled until the job has finished."""

        async def main() -> dict[str, Any]:
            with anyio.fail_after(WAIT_S * 8):
                while True:
                    job = await self.backend.get_job({"job_id": job_id, "wait_s": 10}, None)
                    if job["status"] in ("completed", "failed", "cancelled"):
                        return job

        return anyio.run(main)


@pytest.fixture
def service(tmp_path: Path) -> Iterator[Service]:
    made = Service(tmp_path)
    try:
        yield made
    finally:
        if any(h.thread.is_alive() for h in made.harnesses):
            made.store.post_command("stop_now")
        for harness in made.harnesses:
            harness.thread.join(WAIT_S)
        made.store.close()
        assert not any(h.thread.is_alive() for h in made.harnesses), "the test daemon did not stop"


def test_the_daemon_runs_measure_then_generate_on_the_installed_engine_s4(service: Service) -> None:
    """``measure_voice`` then ``submit_job`` through the backend. The daemon it starts builds the installed
    engine at its first job, measures the voice under the pinned Base profile, and narrates with that profile
    behind the canary gate (``hash_match``: the fake renders the canary's pinned bytes again).

    Every take is analysed by the pinned CTC aligner, as installed. The fake QA worker names its own aligner
    model in its reply, which the pinned aligner refuses (``model_mismatch``) as it must refuse any other
    model's alignment, so the takes fail QA here. Cue placement on the fake is WP36's M1 test, which runs the
    tests' aligner."""
    base = service.store.current_engine_profile("base")
    assert base is not None and base.canary is not None

    queued = service.backend.measure_voice_sync({"voice": service.voice()})
    assert queued["status"] == "queued", queued
    assert service.launcher.started == 1
    measured = service.finished(queued["job_id"])
    assert measured["status"] == "completed", measured
    measurement = service.store.get_measurement(queued["voice_hash"], base.engine_profile_id)
    assert measurement is not None and measurement.engine_profile.hash == base.hash
    again = service.backend.measure_voice_sync({"voice": service.voice()})
    assert again["status"] == "completed" and "job_id" not in again  # the backend finds what the engine stored

    request = {
        "voice": service.voice(),
        "segments": [{"segment_id": "p01", "cues": [{"text": LAMPS}, {"text": KETTLE}]}],
        "options": {"takes": 1},
    }
    submitted = service.backend.submit_job_sync(request)
    assert submitted["engine_profile"] == {"id": base.engine_profile_id, "hash": base.hash}
    assert submitted["plan"]["renders_needed"] == 1
    job = service.finished(submitted["job_id"])
    assert job["status"] == "completed", job
    assert service.launcher.started == 1, "the daemon that serves the store runs the second job too"

    (runner,) = service.runners
    assert isinstance(runner.registry.handler("measure"), MeasureHandler)  # installed_engine's registry

    record = service.store.get_job(submitted["job_id"])
    assert record is not None
    attempts = [a for item in record.items for a in item.attempts]
    assert attempts
    for attempt in attempts:
        render = service.store.get_render(attempt.render_key)
        assert render is not None
        assert (render.engine.engine_profile_id, render.engine.engine_profile_hash) == (
            base.engine_profile_id,
            base.hash,
        )
        assert render.canary.batch_status == "hash_match"

    results = service.backend.get_results_sync({"job_id": submitted["job_id"]})
    (segment,) = results["segments"]
    assert segment["takes"]
    for take in segment["takes"]:
        alignment = take["alignment"]
        assert (alignment["model"], alignment["revision"]) == (CTC_ALIGNER.repo, CTC_ALIGNER.revision)
        refused = [f["details"]["reason"] for f in alignment["flags"] if f["code"] == codes.ALIGNMENT_ERROR]
        assert refused == ["model_mismatch"]  # the fake's reply names its own model

    # The backend keys a render as the installed engine does (the pinned profile's hash): asked again, it
    # finds the render cached.
    again = service.backend.submit_job_sync(request)
    assert again["job_id"] != submitted["job_id"]
    assert again["plan"]["renders_needed"] == 0
    assert service.finished(again["job_id"])["status"] == "completed"

    # ``backend_for`` plans with this installation's analysis pins, which name what the installed engine keys
    # its analyses with: planned again, the segment is cached down to its analysis (section 10.2).
    planned = service.backend.submit_job_sync({**request, "options": {"takes": 1, "dry_run": True}})["plan"]
    assert (planned["segments_cached"], planned["analyses_needed"], planned["renders_needed"]) == (1, 0, 0)
