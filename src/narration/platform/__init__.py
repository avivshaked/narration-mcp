"""Every OS-specific mechanism of the service, behind one interface (design sections 4.1, 17.2, 17.3; WP19).

``get_platform()`` returns the ``narration.contracts.interfaces.Platform`` for this OS:

* on Windows, ``WindowsPlatform``: the daemon's singleton (a named mutex keyed on the store path), detached
  start with breakaway (the daemon runs only once Windows confirms it is in no Job Object), kill-on-close
  Job Objects for workers, below-normal priority, the path rules of sections 17.2 and 17.3, and the
  free-disk check;
* on any other OS, ``UnsupportedOsPlatform``, whose every call raises
  ``narration.contracts.errors.UnsupportedPlatform`` (v1 is Windows only, plan.md Q2).

Both also implement ``ProcessPlatform``: how the daemon starts Python processes on this OS (which
interpreter, which creation flags, which environment variables). Those three are pure descriptions, so the
unsupported platform answers them with the neutral values (no change, no flags, nothing added) rather than
refusing.

``narration.platform.testing.StandInPlatform`` is a stand-in ``Platform`` for tests on any OS; only tests
use it.

This package is the only place allowed to import ``msvcrt``, ``ctypes.windll``/``WinDLL``, ``winreg``,
``fcntl`` or ``win32*`` (AGENTS.md section 6; ruff's TID251 enforces it elsewhere).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol

from narration.contracts.interfaces import Platform

from . import winpaths
from ._unsupported import UnsupportedOsPlatform

SUPPORTED_PLATFORMS: Final[tuple[str, ...]] = ("win32",)
"""The values of ``sys.platform`` that have a real implementation."""


class ProcessPlatform(Platform, Protocol):
    """``Platform`` plus how the daemon starts Python processes on this OS (plan.md WP30).

    On Windows: the detached daemon runs as ``pythonw.exe``, which has no console to show or close, and
    workers as ``python.exe`` (they speak over their standard streams) created with ``CREATE_NO_WINDOW``,
    plus ``BELOW_NORMAL_PRIORITY_CLASS`` when asked; every process the service starts gets
    ``NoDefaultCurrentDirectoryInExePath=1``, so ``cmd.exe`` never runs a program from its working folder
    (section 17).
    """

    def python_for(self, python: Path, *, console: bool) -> Path:
        """The interpreter to run a Python program with, given ``python``: with ``console`` False, one with no
        console of its own (the detached daemon); with ``console`` True, one that has standard streams (a
        worker). Returns ``python`` itself when this OS makes no such distinction or the other one is missing."""
        ...

    def worker_creationflags(self, *, below_normal: bool) -> int:
        """The ``subprocess`` creation flags of a worker process (0 where the OS has none)."""
        ...

    def hardening_env(self) -> Mapping[str, str]:
        """Environment variables every process the service starts gets, on this OS (empty where none)."""
        ...


def is_supported() -> bool:
    """Whether this OS has a real ``Platform`` (``narration-admin doctor`` reports it)."""
    return sys.platform in SUPPORTED_PLATFORMS


def get_platform() -> ProcessPlatform:
    """The ``Platform`` for this OS: ``WindowsPlatform`` on Windows, else ``UnsupportedOsPlatform``.

    It never raises; on an unsupported OS the returned object raises ``UnsupportedPlatform`` on each call.
    Instances hold no state, so every call may build a new one.
    """
    if sys.platform == "win32":
        from ._windows import WindowsPlatform

        return WindowsPlatform()
    return UnsupportedOsPlatform(sys.platform)


def real_path(path: str | os.PathLike[str]) -> str:
    r"""``os.path.realpath(path)``, without the ``\\?\`` prefix CPython can leave on it on Windows.

    KNOW (WP30, spike g): ``ntpath.realpath`` resolves a path to its ``\\?\`` form, and drops the prefix only
    after a second look at the plain path finds the same file. If the file is replaced between the two looks
    (a new file renamed over it), that look fails and the prefix stays: ``\\?\D:\store\run\daemon.json`` for a
    file inside ``D:\store``, which a comparison with the store root takes for a path outside it. The first
    look had already resolved every link, so the plain path is the real one. ``\\?\UNC\server\share``
    becomes ``\\server\share`` (``winpaths.strip_verbatim``).

    This is string handling after the call, with no branch on the OS: a POSIX ``realpath`` never starts
    with the prefix. The result is then ``os.path.normpath``-ed, so no ``..`` survives the stripping. Every
    check that compares ``realpath``s (the store's confinement, section 17.2) goes through this, on both
    sides.
    """
    return os.path.normpath(winpaths.strip_verbatim(os.path.realpath(path)))


__all__ = [
    "SUPPORTED_PLATFORMS",
    "ProcessPlatform",
    "UnsupportedOsPlatform",
    "get_platform",
    "is_supported",
    "real_path",
]
