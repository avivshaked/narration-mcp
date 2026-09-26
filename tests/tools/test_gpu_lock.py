"""The developers' GPU lock: one holder at a time, never broken (plan.md section 2.5)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .test_check_tracked import load_tool

gpu_lock = load_tool("gpu_lock")


def test_acquire_then_release(tmp_path: Path) -> None:
    assert gpu_lock.acquire("WP20", 10, root=tmp_path) == 0
    owner = json.loads((tmp_path / ".dev" / "gpu.lock" / "owner.json").read_text(encoding="utf-8"))
    assert owner["holder"] == "WP20"
    assert gpu_lock.release("WP20", root=tmp_path) == 0
    assert not (tmp_path / ".dev" / "gpu.lock").exists()


def test_a_second_holder_is_refused(tmp_path: Path) -> None:
    assert gpu_lock.acquire("WP20", 10, root=tmp_path) == 0
    assert gpu_lock.acquire("WP22", 10, root=tmp_path) == 1
    assert gpu_lock.release("WP22", root=tmp_path) == 1
    assert (tmp_path / ".dev" / "gpu.lock").exists()


def test_runs_over_thirty_minutes_need_the_lead(tmp_path: Path) -> None:
    assert gpu_lock.acquire("WP39", 45, root=tmp_path) == 2
    assert not (tmp_path / ".dev" / "gpu.lock").exists()


def test_an_expired_lock_with_a_dead_pid_is_stale() -> None:
    long_ago = gpu_lock.now() - dt.timedelta(hours=2)
    owner = {"holder": "x", "expected_end": long_ago.isoformat(), "pid": 2**22 + 12345}
    assert gpu_lock.is_stale(owner)
    fresh = {"holder": "x", "expected_end": (gpu_lock.now() + dt.timedelta(minutes=5)).isoformat(), "pid": 1}
    assert not gpu_lock.is_stale(fresh)
