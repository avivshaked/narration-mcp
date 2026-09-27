"""How the daemon starts Python processes on each OS (``narration.platform.ProcessPlatform``; WP30).

The Windows answers are checked on Windows; the neutral answers of an unsupported OS everywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from narration.platform import ProcessPlatform, UnsupportedOsPlatform, get_platform

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="checks the Windows platform")


def interpreters(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")


def test_every_platform_says_how_it_starts_python_processes() -> None:
    platform: ProcessPlatform = get_platform()
    assert callable(platform.python_for)
    assert isinstance(platform.worker_creationflags(below_normal=True), int)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in platform.hardening_env().items())


@windows_only
def test_windows_runs_the_daemon_windowless_and_workers_as_console_programs_s4_1(tmp_path: Path) -> None:
    platform = get_platform()
    scripts = tmp_path / "Scripts"
    interpreters(scripts, "python.exe")
    assert platform.python_for(scripts / "python.exe", console=False) == scripts / "python.exe", "no pythonw.exe"
    interpreters(scripts, "pythonw.exe")
    assert platform.python_for(scripts / "python.exe", console=False) == scripts / "pythonw.exe"
    assert platform.python_for(scripts / "pythonw.exe", console=True) == scripts / "python.exe"
    assert platform.python_for(scripts / "python.exe", console=True) == scripts / "python.exe"
    assert platform.python_for(tmp_path / "other.exe", console=False) == tmp_path / "other.exe"


@windows_only
def test_windows_workers_have_no_window_and_may_run_below_normal_s4_1() -> None:
    platform = get_platform()
    no_window = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    below = subprocess.BELOW_NORMAL_PRIORITY_CLASS  # type: ignore[attr-defined]
    assert platform.worker_creationflags(below_normal=False) == no_window
    assert platform.worker_creationflags(below_normal=True) == no_window | below


@windows_only
def test_windows_turns_off_cmds_search_of_the_working_folder_s17() -> None:
    assert dict(get_platform().hardening_env()) == {"NoDefaultCurrentDirectoryInExePath": "1"}


def test_an_unsupported_os_answers_with_neutral_values(tmp_path: Path) -> None:
    platform = UnsupportedOsPlatform("plan9")
    interpreters(tmp_path, "python.exe", "pythonw.exe")
    assert platform.python_for(tmp_path / "python.exe", console=False) == tmp_path / "python.exe"
    assert platform.worker_creationflags(below_normal=True) == 0
    assert dict(platform.hardening_env()) == {}
