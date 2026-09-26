"""Every OS-specific mechanism of the service, behind one interface (design sections 4.1, 17.2, 17.3; WP19).

``get_platform()`` returns the ``narration.contracts.interfaces.Platform`` for this OS:

* on Windows, ``WindowsPlatform``: the daemon's singleton (a named mutex keyed on the store path), detached
  start with breakaway, kill-on-close Job Objects for workers, below-normal priority, the path rules of
  sections 17.2 and 17.3, and the free-disk check;
* on any other OS, ``UnsupportedOsPlatform``, whose every call raises
  ``narration.contracts.errors.UnsupportedPlatform`` (v1 is Windows only, plan.md Q2).

This package is the only place allowed to import ``msvcrt``, ``ctypes.windll``/``WinDLL``, ``winreg``,
``fcntl`` or ``win32*`` (AGENTS.md section 6; ruff's TID251 enforces it elsewhere).
"""

from __future__ import annotations

import os
import sys
from typing import Final

from narration.contracts.interfaces import Platform

from . import winpaths
from ._unsupported import UnsupportedOsPlatform

SUPPORTED_PLATFORMS: Final[tuple[str, ...]] = ("win32",)
"""The values of ``sys.platform`` that have a real implementation."""


def is_supported() -> bool:
    """Whether this OS has a real ``Platform`` (``narration-admin doctor`` reports it)."""
    return sys.platform in SUPPORTED_PLATFORMS


def get_platform() -> Platform:
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


__all__ = ["SUPPORTED_PLATFORMS", "UnsupportedOsPlatform", "get_platform", "is_supported", "real_path"]
