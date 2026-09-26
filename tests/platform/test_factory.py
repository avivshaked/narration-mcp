"""The platform factory and the unsupported-OS behaviour (plan.md Q2, WP19). These run on every OS."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from narration.contracts import codes
from narration.contracts.errors import UnsupportedPlatform
from narration.contracts.interfaces import Platform
from narration.platform import SUPPORTED_PLATFORMS, UnsupportedOsPlatform, get_platform, is_supported

PLATFORM_METHODS = sorted(
    name for name, member in inspect.getmembers(Platform, inspect.isfunction) if not name.startswith("_")
)


def _call(platform: Any, name: str, tmp_path: Path) -> Callable[[], object]:
    """A call of ``name`` with plausible arguments (never reached on an unsupported OS)."""
    args: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {
        "singleton": ((tmp_path,), {}),
        "spawn_detached": ((["python", "-c", "pass"],), {"cwd": tmp_path, "env": {}}),
        "kill_on_close_group": ((), {}),
        "set_below_normal_priority": ((1234,), {}),
        "check_readable_path": ((str(tmp_path / "clip.wav"),), {}),
        "check_store_path": ((tmp_path / "x", tmp_path), {}),
        "free_disk_bytes": ((tmp_path,), {}),
    }
    positional, keyword = args[name]
    return lambda: getattr(platform, name)(*positional, **keyword)


def test_the_protocol_lists_the_seven_operations_wp19() -> None:
    assert PLATFORM_METHODS == [
        "check_readable_path",
        "check_store_path",
        "free_disk_bytes",
        "kill_on_close_group",
        "set_below_normal_priority",
        "singleton",
        "spawn_detached",
    ]


def test_get_platform_returns_a_platform_on_every_os_q2() -> None:
    platform = get_platform()
    assert isinstance(platform, Platform)
    assert is_supported() == (sys.platform in SUPPORTED_PLATFORMS)
    assert SUPPORTED_PLATFORMS == ("win32",)
    if sys.platform == "win32":
        assert type(platform).__name__ == "WindowsPlatform"
    else:
        assert isinstance(platform, UnsupportedOsPlatform)


def test_the_unsupported_platform_is_a_platform_q2() -> None:
    assert isinstance(UnsupportedOsPlatform("linux"), Platform)


@pytest.mark.parametrize("name", PLATFORM_METHODS)
def test_every_call_on_an_unsupported_os_raises_unsupported_platform_q2(name: str, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedPlatform) as info:
        _call(UnsupportedOsPlatform("linux"), name, tmp_path)()
    error = info.value
    assert error.code == codes.DAEMON_UNAVAILABLE
    assert error.retryable is False
    assert error.operation == name
    assert error.platform == "linux"
    assert error.details == {"operation": name, "platform": "linux"}
    assert "Windows" in error.hint
    assert "doctor" in error.hint


@pytest.mark.skipif(sys.platform == "win32", reason="checks the behaviour off Windows (plan.md Q2)")
@pytest.mark.parametrize("name", PLATFORM_METHODS)
def test_this_os_refuses_every_call_off_windows_q2(name: str, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedPlatform) as info:
        _call(get_platform(), name, tmp_path)()
    assert info.value.platform == sys.platform


@pytest.mark.skipif(sys.platform == "win32", reason="checks the behaviour off Windows (plan.md Q2)")
def test_the_windows_module_refuses_to_import_off_windows_q2() -> None:
    with pytest.raises(ImportError, match="Windows only"):
        importlib.import_module("narration.platform._windows")
