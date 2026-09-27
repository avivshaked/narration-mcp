"""Starting the daemon from the front-end (design sections 4.1, 16; plan.md DC-2): ``DetachedLauncher`` over a
stand-in for WP30's ``narration.daemon.start``. Nothing is started here."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from narration.backend import launch
from narration.backend.launch import DAEMON_RETRY_S, START_HINT, DetachedLauncher
from narration.contracts import codes
from narration.contracts.errors import NarrationError, UnsupportedPlatform

from .support import daemon_status

STORE: Any = object()
CONFIG = Path("narration.toml")


def with_start(monkeypatch: pytest.MonkeyPatch, **functions: Any) -> list[tuple[Any, ...]]:
    """Stand in for ``narration.daemon.start``; returns the calls made to ``ensure_daemon``."""
    calls: list[tuple[Any, ...]] = []

    def ensure_daemon(store: Any, config_path: Path, *, extra: tuple[str, ...] = ()) -> Any:
        calls.append((store, config_path, extra))
        raising = functions.get("raises")
        if raising is not None:
            raise raising
        return SimpleNamespace(started=True, spawned_pid=42)

    module = SimpleNamespace(ensure_daemon=ensure_daemon, running_daemon=functions.get("running", lambda s: None))
    monkeypatch.setattr(launch, "_start_module", lambda: module)
    return calls


def unavailable(launcher: DetachedLauncher) -> NarrationError:
    with pytest.raises(NarrationError) as caught:
        launcher.ensure(STORE)
    return caught.value


def test_ensure_starts_the_daemon_with_the_config_s4_1(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = with_start(monkeypatch)
    DetachedLauncher(CONFIG, extra=("--fake-workers",)).ensure(STORE)
    assert calls == [(STORE, CONFIG, ("--fake-workers",))]


def test_without_autostart_nothing_is_started_s16(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = with_start(monkeypatch)
    DetachedLauncher(CONFIG, autostart=False).ensure(STORE)
    assert calls == []


def test_running_reads_the_daemons_status_s4_1(monkeypatch: pytest.MonkeyPatch) -> None:
    status = daemon_status()
    with_start(monkeypatch, running=lambda store: status)
    assert DetachedLauncher(CONFIG).running(STORE) is status


@pytest.mark.parametrize(
    "failure",
    [OSError(5, "Access is denied"), NarrationError(codes.DAEMON_UNAVAILABLE, "breakaway refused")],
)
def test_a_start_that_fails_is_retryable_with_what_to_do_dc2(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    with_start(monkeypatch, raises=failure)
    error = unavailable(DetachedLauncher(CONFIG))
    assert (error.code, error.retryable, error.retry_after_s) == (codes.DAEMON_UNAVAILABLE, True, DAEMON_RETRY_S)
    assert error.hint == START_HINT


def test_a_platform_the_daemon_cannot_run_on_is_not_retryable_s4(monkeypatch: pytest.MonkeyPatch) -> None:
    with_start(monkeypatch, raises=UnsupportedPlatform("spawn_detached", "plan9"))
    error = unavailable(DetachedLauncher(CONFIG))
    assert (error.code, error.retryable) == (codes.DAEMON_UNAVAILABLE, False)
    assert error.hint == UnsupportedPlatform.HINT


def test_no_config_file_means_no_daemon_can_be_started_s16(monkeypatch: pytest.MonkeyPatch) -> None:
    with_start(monkeypatch)
    error = unavailable(DetachedLauncher(None))
    assert (error.code, error.retryable) == (codes.DAEMON_UNAVAILABLE, True)


def test_a_build_without_the_daemon_says_so_s4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launch, "_start_module", lambda: None)
    assert DetachedLauncher(CONFIG).running(STORE) is None
    assert unavailable(DetachedLauncher(CONFIG)).code == codes.DAEMON_UNAVAILABLE
