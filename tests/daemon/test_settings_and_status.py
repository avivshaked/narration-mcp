"""The daemon's settings (section 16) and ``run/daemon.json`` (sections 4.1, 7.6, 15)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pytest

from narration.config import Config, DaemonConfig, WorkersConfig
from narration.contracts.models import DaemonStatus, WorkerInfo
from narration.contracts.names import STOP_REASONS, StopReason
from narration.daemon.seam import GpuFacts
from narration.daemon.settings import DaemonSettings
from narration.daemon.status import StatusBoard
from narration.store import NarrationStore

from .conftest import make_job


# ---------------------------------------------------------------- settings (section 16)
def test_settings_come_from_daemon_and_workers_s16(tmp_path: Path) -> None:
    config = dataclasses.replace(
        Config.for_tests(tmp_path / "store"),
        daemon=DaemonConfig(idle_unload_s=7, idle_exit_min=3),
        workers=WorkersConfig(cpu_threads=5, priority="normal"),
    )
    settings = DaemonSettings.from_config(config)
    assert settings.store_root == tmp_path / "store"
    assert settings.idle_unload_s == 7.0
    assert settings.idle_exit_s == 180.0
    assert settings.cpu_threads == 5
    assert settings.below_normal is False


def test_the_design_defaults_are_120_s_and_15_min_s4() -> None:
    settings = DaemonSettings.from_config(Config.for_tests(Path("store")))
    assert (settings.idle_unload_s, settings.idle_exit_s, settings.below_normal) == (120.0, 900.0, True)


def test_overrides_win_and_none_is_no_override_s16(tmp_path: Path) -> None:
    settings = DaemonSettings.from_config(
        Config.for_tests(tmp_path), idle_unload_s=0.2, idle_exit_s=None, poll_s=0.05, fake_workers=True
    )
    assert settings.idle_unload_s == 0.2
    assert settings.idle_exit_s == 900.0
    assert settings.poll_s == 0.05
    assert settings.fake_workers


@pytest.mark.parametrize(
    ("field", "value"),
    [("idle_unload_s", -1.0), ("poll_s", 0.0), ("cpu_threads", 0), ("idle_exit_s", float("nan"))],
)
def test_settings_refuse_nonsense(tmp_path: Path, field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(DaemonSettings(store_root=tmp_path), **{field: value})


# ---------------------------------------------------------------- run/daemon.json (sections 4.1, 7.6, 15)
def test_the_status_is_written_on_start_and_only_on_change_s15(store: NarrationStore) -> None:
    board = StatusBoard(store, pid=4321, started_at="2026-09-26T10:00:00.000Z")
    board.set_state("idle")
    written = store.get_daemon_status()
    assert written is not None
    assert (written.state, written.pid, written.started_at) == ("idle", 4321, "2026-09-26T10:00:00.000Z")
    assert written.est_drain_s is None
    assert board.writes == 1
    board.set_state("idle")
    board.set_workers((), None)
    assert board.writes == 1, "nothing changed, so nothing was written"
    board.set_workers((WorkerInfo(role="fake", pid=77),), "qwen")
    assert board.writes == 2
    written = store.get_daemon_status()
    assert written is not None
    assert written.workers == (WorkerInfo(role="fake", pid=77),)
    assert (written.gpu.in_use, written.gpu.holder) == (True, "qwen")


def test_a_job_makes_the_daemon_busy_with_its_phase_s7_6(store: NarrationStore) -> None:
    job = make_job(store, "One line.", label="chapter one")
    board = StatusBoard(store, pid=1, started_at="2026-09-26T10:00:00.000Z")
    board.set_state("idle")
    board.job_started(job)
    busy = store.get_daemon_status()
    assert busy is not None and busy.state == "busy"
    assert busy.current_job is not None
    assert (busy.current_job.job_id, busy.current_job.kind, busy.current_job.label) == (
        job.job_id,
        "generate",
        "chapter one",
    )
    board.job_phase("rendering")
    rendering = store.get_daemon_status()
    assert rendering is not None and rendering.current_job is not None
    assert rendering.current_job.phase == "rendering"
    assert rendering.current_job.started_at == busy.current_job.started_at
    board.job_finished()
    idle = store.get_daemon_status()
    assert idle is not None and (idle.state, idle.current_job) == ("idle", None)


def test_stopping_outlasts_the_job_that_finishes_during_it_s4_1(store: NarrationStore) -> None:
    board = StatusBoard(store, pid=1, started_at="2026-09-26T10:00:00.000Z")
    board.job_started(make_job(store, "One line."))
    board.set_state("stopping")
    board.job_finished()
    assert board.state == "stopping"


def test_unload_in_s_counts_down_from_updated_at_only_while_idle_s7_6(store: NarrationStore) -> None:
    board = StatusBoard(store, pid=1, started_at="2026-09-26T10:00:00.000Z")
    board.set_workers((WorkerInfo(role="fake", pid=5),), "qwen")
    board.set_unload_at(time.time() + 100.0)
    status = store.get_daemon_status()
    assert status is not None and status.gpu.unload_in_s is not None
    assert 99.0 < status.gpu.unload_in_s <= 100.0
    board.job_started(make_job(store, "One line."))
    busy = store.get_daemon_status()
    assert busy is not None and busy.gpu.unload_in_s is None, "no idle countdown while a job runs"
    board.job_finished()
    board.set_workers((), None)
    board.set_unload_at(time.time() + 100.0)
    nothing = store.get_daemon_status()
    assert nothing is not None and nothing.gpu.unload_in_s is None, "no countdown without a model on the GPU"


def test_gpu_facts_and_the_drain_estimate_come_from_the_runner_dc2(store: NarrationStore) -> None:
    board = StatusBoard(store, pid=1, started_at="2026-09-26T10:00:00.000Z")
    board.set_gpu_facts(
        GpuFacts(name="a GPU", total_mb=24000, free_mb=9000, need_mb={"qwen": 7000}, waiting_since=None)
    )
    board.set_est_drain(42.5)
    status = store.get_daemon_status()
    assert status is not None
    assert (status.gpu.name, status.gpu.total_mb, status.gpu.free_mb) == ("a GPU", 24000, 9000)
    assert status.gpu.need_mb == {"qwen": 7000}
    assert status.est_drain_s == 42.5


@pytest.mark.parametrize("reason", STOP_REASONS)
def test_the_last_write_says_stopped_with_no_pid_keeps_the_start_and_says_why_s15(
    store: NarrationStore, reason: StopReason
) -> None:
    board = StatusBoard(store, pid=99, started_at="2026-09-26T10:00:00.000Z")
    board.set_workers((WorkerInfo(role="fake", pid=5),), "qwen")
    board.stopped(reason)
    status = store.get_daemon_status()
    assert status is not None
    assert (status.state, status.pid, status.workers) == ("stopped", None, ())
    assert status.started_at == "2026-09-26T10:00:00.000Z", "the start time is kept (contracts 1.6.11)"
    assert status.stop_reason == reason
    assert (status.gpu.in_use, status.gpu.holder, status.current_job) == (False, None, None)


def test_a_running_status_says_no_stop_reason_s15(store: NarrationStore) -> None:
    board = StatusBoard(store, pid=99, started_at="2026-09-26T10:00:00.000Z")
    board.set_state("stopping")
    status = store.get_daemon_status()
    assert status is not None and (status.state, status.stop_reason) == ("stopping", None)
    written = json.loads(store.layout.daemon_json_path().read_text(encoding="utf-8"))
    assert "stop_reason" not in written, "left out while the daemon runs, so an older reader still loads it"


def test_a_status_written_before_1_6_11_still_loads_s15(store: NarrationStore) -> None:
    # An older daemon's last write: stopped, with no start time and no stop_reason field at all.
    old = {
        "state": "stopped",
        "pid": None,
        "started_at": None,
        "workers": [],
        "current_job": None,
        "gpu": {"name": None, "total_mb": None, "free_mb": None, "in_use": False, "holder": None, "unload_in_s": None},
        "est_drain_s": None,
        "updated_at": "2026-09-26T10:00:00.000Z",
    }
    path = store.layout.daemon_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old), encoding="utf-8")
    status = store.get_daemon_status()
    assert status is not None
    assert (status.state, status.started_at, status.stop_reason) == ("stopped", None, None)


def test_a_status_that_cannot_be_written_never_stops_the_daemon(store: NarrationStore) -> None:
    class Broken:
        def put_daemon_status(self, status: object) -> None:
            raise OSError("the disk is full")

    board = StatusBoard(Broken(), pid=1, started_at="2026-09-26T10:00:00.000Z")  # pyright: ignore[reportArgumentType]
    board.set_state("idle")
    assert board.writes == 0


def test_a_status_write_that_failed_is_tried_again_by_flush_s15(store: NarrationStore) -> None:
    class FailsOnce:
        def __init__(self) -> None:
            self.failed = False

        def put_daemon_status(self, status: DaemonStatus) -> None:
            if not self.failed:
                self.failed = True
                raise PermissionError(13, "Permission denied")  # a reader held the file open (Windows)
            store.put_daemon_status(status)

    board = StatusBoard(FailsOnce(), pid=1, started_at="2026-09-26T10:00:00.000Z")  # pyright: ignore[reportArgumentType]
    board.set_state("idle")
    assert (board.writes, store.get_daemon_status()) == (0, None)
    board.flush()
    written = store.get_daemon_status()
    assert board.writes == 1 and written is not None and written.state == "idle"
    board.flush()
    assert board.writes == 1, "nothing changed since, so nothing more is written"
