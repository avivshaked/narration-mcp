"""A submission run by the real daemon (WP30) and job engine (WP31) on fake workers, from ``submit_job`` to
``get_results`` (design sections 4, 4.1, 7.3 to 7.5).

The daemon runs on a thread of this process, as WP30's own tests run it, with the fake role for every worker
group. The launcher starts it the way ``narration.daemon.start.ensure_daemon`` does: only after the job is
committed, and only when no daemon serves the store. No GPU and no model: every text is invented for these
tests.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from narration_worker.fake.faults import SPEC_ENV

from narration.backend import NarrationBackend
from narration.backend.launch import SERVING_STATES
from narration.contracts.interfaces import Store
from narration.contracts.models import DaemonStatus
from narration.daemon import start as daemon_start
from narration.daemon.service import Daemon
from narration.daemon.settings import DaemonSettings
from narration.daemon.supervisor import FAKE_ROLES, WorkerSupervisor
from narration.jobs.runner import build_runner
from narration.platform.testing import StandInPlatform
from tests.daemon.conftest import WAIT_S, DaemonHarness
from tests.jobs.conftest import World, make_world
from tests.jobs.support import KETTLE, LAMPS, ORCHARD, FixedProbe, TestAligner, check_readable_path, qa_pins

from .support import FakeMeasurements, TestPlatform, pins_for, sha256_of


class ThreadLauncher:
    """Starts the daemon on a thread, by ``ensure_daemon``'s rule, and reads its status as the front-end
    does (``running_daemon``)."""

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
    def __init__(self, world: World) -> None:
        self.world = world
        self.harnesses: list[DaemonHarness] = []
        self.platform = StandInPlatform()
        self.launcher = ThreadLauncher(self.start_daemon)
        found = pins_for(world.config.server.models_root)
        self.backend = NarrationBackend(
            world.config,
            world.store,
            TestPlatform(),
            launcher=self.launcher,
            pins=lambda: found,
            measurements=FakeMeasurements(world.store),
            poll_s=0.05,
        )

    def start_daemon(self) -> None:
        world = self.world
        settings = DaemonSettings(
            store_root=world.config.server.store_root,
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
        env[SPEC_ENV] = str(world.spec)
        runner = build_runner(
            world.config,
            aligner=TestAligner(),
            qa_pins=qa_pins(world.config.server.models_root),
            probe=FixedProbe(),
            check_path=check_readable_path,
        )

        def supervisor(on_change: Callable[[], None]) -> WorkerSupervisor:
            return WorkerSupervisor(
                world.config, self.platform, roles=FAKE_ROLES, base_env=env, close_timeout_s=5.0, on_change=on_change
            )

        daemon = Daemon(
            settings=settings,
            config=world.config,
            store=world.store,
            platform=self.platform,
            runner=runner,
            supervisor_factory=supervisor,
        )
        self.harnesses.append(DaemonHarness(daemon, world.store).start())

    def finished(self, job_id: str) -> dict[str, Any]:
        """``get_job``, long-polled until the job has finished."""

        async def main() -> dict[str, Any]:
            with anyio.fail_after(WAIT_S * 4):
                while True:
                    job = await self.backend.get_job({"job_id": job_id, "wait_s": 10}, None)
                    if job["status"] in ("completed", "failed", "cancelled"):
                        return job

        return anyio.run(main)


@pytest.fixture
def service(tmp_path: Path, anchor: tuple[float, ...]) -> Iterator[Service]:
    made = Service(make_world(tmp_path, anchor))
    try:
        yield made
    finally:
        if any(h.thread.is_alive() for h in made.harnesses):
            made.world.store.post_command("stop_now")
        for harness in made.harnesses:
            harness.thread.join(WAIT_S)
        made.world.pool.close()
        made.world.store.close()
        assert not any(h.thread.is_alive() for h in made.harnesses), "the test daemon did not stop"


def test_a_submission_is_run_by_the_daemon_it_starts_s4(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.world.request(LAMPS, KETTLE, takes=2))
    assert service.launcher.started == 1, "the first submission starts the daemon, after the job is committed"

    job = service.finished(submitted["job_id"])
    assert job["status"] == "completed", job
    results = service.backend.get_results_sync({"job_id": submitted["job_id"]})
    for segment in results["segments"]:
        assert segment["status"] in ("passed", "warned"), segment
        assert len(segment["takes"]) == 2
        for take in segment["takes"]:
            assert sha256_of(Path(take["delivery"]["path"])) == take["delivery"]["sha256"]
            assert all(c["start_s"] is not None for c in take["cues"])
            assert take["qa"]["verdict"] in ("pass", "warn", "fail")
    assert Path(results["report_md"]).is_file()

    status = service.backend.get_server_status_sync({})
    assert status["daemon"]["state"] in SERVING_STATES
    assert status["daemon"]["pid"] == os.getpid()

    again = service.backend.submit_job_sync(service.world.request(ORCHARD))
    assert service.launcher.started == 1, "a daemon that serves the store is not started again"
    assert service.finished(again["job_id"])["status"] == "completed"
