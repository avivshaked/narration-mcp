"""Starting the daemon detached, and telling whether one runs (sections 4, 4.1; the helper WP36 and WP37 use)."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import DaemonStatus, GpuStatus
from narration.daemon.start import daemon_argv, daemon_python, ensure_daemon, running_daemon, start_detached
from narration.store import NarrationStore
from narration.store.store import utc_iso

from .standin import StandInPlatform


def live_status(state: str = "idle") -> DaemonStatus:
    """A status naming this test process as the daemon (alive, created before ``started_at``)."""
    return DaemonStatus(
        state=state,  # pyright: ignore[reportArgumentType]
        pid=os.getpid(),
        started_at=utc_iso(time.time()),
        workers=(),
        current_job=None,
        gpu=GpuStatus(name=None, total_mb=None, free_mb=None, in_use=False, holder=None, unload_in_s=None),
        est_drain_s=None,
        updated_at=utc_iso(time.time()),
    )


def test_the_command_line_carries_the_identity_marker_s4_1(tmp_path: Path) -> None:
    argv = daemon_argv(tmp_path / "store", tmp_path / "narration.toml", python=Path(sys.executable))
    assert argv[1:] == [
        "-m",
        "narration.daemon",
        "--store",
        str(tmp_path / "store"),
        "--config",
        str(tmp_path / "narration.toml"),
    ]


@pytest.mark.skipif(os.name != "nt", reason="pythonw.exe exists on Windows only")
def test_on_windows_the_daemon_runs_windowless_s4_1(tmp_path: Path) -> None:
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    python = scripts / "python.exe"
    python.write_bytes(b"")
    assert daemon_python(python) == python, "without pythonw.exe beside it, python.exe is used"
    (scripts / "pythonw.exe").write_bytes(b"")
    assert daemon_python(python) == scripts / "pythonw.exe"


@pytest.mark.skipif(os.name == "nt", reason="checks the behaviour off Windows")
def test_off_windows_the_daemon_runs_as_this_python_s4_1(tmp_path: Path) -> None:
    (tmp_path / "pythonw.exe").write_bytes(b"")
    assert daemon_python(tmp_path / "python.exe") == tmp_path / "python.exe"


def test_a_detached_start_uses_the_platform_with_the_store_as_cwd_s4_1(tmp_path: Path) -> None:
    platform = StandInPlatform()
    root = tmp_path / "new-store"
    pid = start_detached(root, tmp_path / "narration.toml", platform=platform, env={"A": "1"})
    assert pid == platform.spawn_pid
    assert root.is_dir(), "the store root exists before the daemon takes its singleton"
    ((argv, cwd, env),) = platform.spawned
    assert argv[1:5] == ["-m", "narration.daemon", "--store", str(root)]
    assert cwd == root
    assert env == {"A": "1"}


def test_breakaway_refused_is_daemon_unavailable_and_nothing_else_starts_s4_1(tmp_path: Path) -> None:
    platform = StandInPlatform()
    platform.refuse_spawn = NarrationError(codes.DAEMON_UNAVAILABLE, "breakaway refused", retry_after_s=60.0)
    with pytest.raises(NarrationError) as info:
        start_detached(tmp_path / "store", tmp_path / "narration.toml", platform=platform)
    assert info.value.code == codes.DAEMON_UNAVAILABLE
    assert platform.spawned == [], "no fallback to a daemon that is not detached"


def test_ensure_starts_nothing_while_a_daemon_serves_s4(store: NarrationStore, tmp_path: Path) -> None:
    platform = StandInPlatform()
    store.put_daemon_status(live_status("busy"))
    result = ensure_daemon(store, tmp_path / "narration.toml", platform=platform)
    assert (result.started, result.spawned_pid) == (False, None)
    assert result.status is not None and result.status.pid == os.getpid()
    assert platform.spawned == []


@pytest.mark.parametrize("state", ["stopping", "stopped"])
def test_ensure_starts_one_when_the_daemon_is_leaving_or_gone_s4(
    store: NarrationStore, tmp_path: Path, state: str
) -> None:
    platform = StandInPlatform()
    store.put_daemon_status(live_status(state))
    result = ensure_daemon(store, tmp_path / "narration.toml", platform=platform)
    assert result.started and result.spawned_pid == platform.spawn_pid
    assert len(platform.spawned) == 1


def test_ensure_can_wait_until_the_new_daemon_serves_s4(store: NarrationStore, tmp_path: Path) -> None:
    platform = StandInPlatform()
    writer = threading.Timer(0.3, lambda: store.put_daemon_status(live_status("idle")))
    writer.start()
    try:
        result = ensure_daemon(store, tmp_path / "narration.toml", platform=platform, wait_s=10.0)
    finally:
        writer.cancel()
    assert result.started
    assert result.status is not None and result.status.state == "idle"


def test_running_daemon_is_none_without_a_live_status_s4_1(store: NarrationStore) -> None:
    assert running_daemon(store) is None
    store.put_daemon_status(live_status("idle"))
    assert running_daemon(store) is not None
