"""``narration.platform.find_program``: a program is found on ``PATH``, never in the working folder (security
review S2; design section 17 item 9)."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from narration.platform import find_program

NAME = "narration-probe-tool"


def _program(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (f"{NAME}.bat" if sys.platform == "win32" else NAME)
    path.write_text("@echo off\n" if sys.platform == "win32" else "#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_a_program_in_the_working_folder_is_never_found_s17_9(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    here = tmp_path / "working"
    planted = _program(here)
    on_path = _program(tmp_path / "bin")
    monkeypatch.chdir(here)
    # An empty and a relative entry both mean the working folder to a POSIX shell; Windows adds it by itself.
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", str(on_path.parent)]))
    found = find_program(NAME)
    assert found is not None
    assert Path(found).resolve() == on_path.resolve()
    assert Path(found).resolve() != planted.resolve()


def test_a_program_only_in_the_working_folder_is_not_found_s17_9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    here = tmp_path / "working"
    _program(here)
    monkeypatch.chdir(here)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", str(tmp_path / "empty")]))
    assert find_program(NAME) is None
