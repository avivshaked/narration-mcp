"""Fixtures for the daemon's tests: a configuration and store under pytest's tmp_path, the platform stand-in,
fake-worker supervisors, and a harness that runs a ``Daemon`` on a thread and always stops it."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from narration_worker.fake.faults import SPEC_ENV

from narration.config import Config, parse_config
from narration.contracts.models import DaemonCommand, DaemonStatus, JobRecord, Progress
from narration.contracts.names import DaemonCommandKind, JobStatus, WorkerRole
from narration.daemon.seam import JobRunner
from narration.daemon.service import Daemon
from narration.daemon.settings import DaemonSettings
from narration.daemon.supervisor import FAKE_ROLES, WorkerSupervisor
from narration.daemon.sweep import read_status
from narration.daemon.testing import job_request
from narration.keys import Keys
from narration.store import NarrationStore
from narration.store.store import utc_iso
from narration.workers import WorkerCommand, worker_command

from .standin import StandInPlatform

WAIT_S = 30.0
"""The longest any test waits for a daemon to reach a state."""


@pytest.fixture
def platform() -> StandInPlatform:
    return StandInPlatform()


@pytest.fixture
def service_root(tmp_path: Path) -> Path:
    """``<service_root>`` with the two worker project folders (the workers run in them, section 17)."""
    root = tmp_path / "service"
    for name in ("qwen3tts", "qa"):
        (root / "workers" / name).mkdir(parents=True)
    return root


@pytest.fixture
def config(service_root: Path) -> Config:
    return parse_config(
        {"server": {"store_root": "store", "models_root": "models"}},
        base=service_root,
        path=service_root / "narration.toml",
    )


@pytest.fixture
def store(config: Config, platform: StandInPlatform) -> Iterator[NarrationStore]:
    with NarrationStore(config.server.store_root, platform) as s:
        yield s


def fake_env(folder: Path, spec: dict[str, Any] | None = None) -> dict[str, str]:
    """This process's environment for a fake worker, with a fault spec when one is given."""
    env = {k: v for k, v in os.environ.items() if k != SPEC_ENV}
    if spec is not None:
        path = folder / "fake-spec.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        env[SPEC_ENV] = str(path)
    return env


def unstartable(
    config: Config, env: dict[str, str], platform: StandInPlatform
) -> Callable[[WorkerRole, str | None], WorkerCommand]:
    """A command factory whose fake worker cannot start: its handler cannot be imported, so it exits with
    ``EXIT_START_FAILED`` before ``hello``, as a worker whose env is broken does (``BACKEND_NOT_INSTALLED``)."""
    python = platform.python_for(Path(sys.executable), console=True)

    def command(role: WorkerRole, cublas: str | None) -> WorkerCommand:
        made = worker_command(config, role, python=python, base_env=env, cublas_workspace_config=cublas)
        return dataclasses.replace(made, argv=(*made.argv, "--handler", "no_such_module_here:Handler"))

    return command


def make_job(store: NarrationStore, *texts: str, status: JobStatus = "queued", label: str | None = None) -> JobRecord:
    """A job the fake runner accepts: one segment per text."""
    request = job_request(*texts)
    now = utc_iso(time.time())
    record = JobRecord(
        job_id=Keys().new_job_id(),
        kind="generate",
        request=request,
        request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest(),
        label=label,
        priority="batch",
        status=status,
        phase=None,
        round=0,
        progress=Progress(done_s=0.0, total_s=0.0, fraction=0.0, segments_done=0, segments_total=len(texts)),
        outcome=None,
        error=None,
        idempotency_key=None,
        created_at=now,
        updated_at=now,
    )
    return store.create_job(record)[0]


def wait_until(predicate: Callable[[], bool], timeout_s: float = WAIT_S, what: str = "a condition") -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout_s} s waiting for {what}")
        time.sleep(0.02)


class DaemonHarness:
    """A ``Daemon`` running on a thread of this process."""

    def __init__(self, daemon: Daemon, store: NarrationStore) -> None:
        self.daemon = daemon
        self.store = store
        self.exit_code: int | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="test-daemon", daemon=True)

    def _run(self) -> None:
        try:
            self.exit_code = self.daemon.run()
        except BaseException as exc:
            self.error = exc

    def start(self) -> DaemonHarness:
        self.thread.start()
        return self

    def status(self) -> DaemonStatus | None:
        return read_status(self.store)

    def wait_status(self, predicate: Callable[[DaemonStatus], bool], what: str = "a status") -> DaemonStatus:
        found: list[DaemonStatus] = []

        def check() -> bool:
            status = self.status()
            if status is not None and predicate(status):
                found.append(status)
                return True
            return False

        wait_until(check, what=what)
        return found[0]

    def wait_serving(self) -> DaemonStatus:
        """This daemon's own status (its pid is this process's), once it serves."""
        return self.wait_status(lambda s: s.state in ("idle", "busy") and s.pid == os.getpid(), "the daemon to serve")

    def command(self, kind: DaemonCommandKind, timeout_s: float = WAIT_S) -> DaemonCommand:
        posted = self.store.post_command(kind)
        done = self.store.wait_for_command(posted.command_id, timeout_s=timeout_s)
        assert done is not None, f"{kind} was not completed within {timeout_s} s"
        return done

    def join(self, timeout_s: float = WAIT_S) -> int:
        self.thread.join(timeout_s)
        assert not self.thread.is_alive(), f"the daemon did not stop within {timeout_s} s"
        if self.error is not None:
            raise self.error
        assert self.exit_code is not None
        return self.exit_code


DaemonFactory = Callable[..., DaemonHarness]


@pytest.fixture
def run_daemon(
    config: Config, store: NarrationStore, platform: StandInPlatform, tmp_path: Path
) -> Iterator[DaemonFactory]:
    """Starts daemons on threads, with fake workers; every one is stopped (``stop_now``) at teardown."""
    harnesses: list[DaemonHarness] = []

    def start(
        runner: JobRunner,
        *,
        spec: dict[str, Any] | None = None,
        start: bool = True,
        broken_workers: bool = False,
        **overrides: Any,
    ) -> DaemonHarness:
        env = fake_env(tmp_path, spec)
        values: dict[str, Any] = {
            "store_root": config.server.store_root,
            "idle_unload_s": 60.0,
            "idle_exit_s": 60.0,
            "poll_s": 0.05,
            "fake_workers": True,
            "worker_close_s": 5.0,
            "stop_now_grace_s": 10.0,
            "takeover_wait_s": 5.0,
            "command_wait_s": 2.0,
        }
        values.update(overrides)
        settings = DaemonSettings(**values)

        def supervisor(on_change: Callable[[], None]) -> WorkerSupervisor:
            return WorkerSupervisor(
                config,
                platform,
                roles=FAKE_ROLES,
                base_env=env,
                close_timeout_s=5.0,
                on_change=on_change,
                command_factory=unstartable(config, env, platform) if broken_workers else None,
            )

        daemon = Daemon(
            settings=settings,
            config=config,
            store=store,
            platform=platform,
            runner=runner,
            supervisor_factory=supervisor,
        )
        harness = DaemonHarness(daemon, store)
        harnesses.append(harness)
        return harness.start() if start else harness

    try:
        yield start
    finally:
        for harness in harnesses:
            if harness.thread.is_alive():
                store.post_command("stop_now")
            harness.thread.join(WAIT_S)
        assert not any(h.thread.is_alive() for h in harnesses), "a test daemon did not stop"
