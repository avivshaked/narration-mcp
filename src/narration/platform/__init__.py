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

import sys
from typing import Final

from narration.contracts.interfaces import Platform

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


__all__ = ["SUPPORTED_PLATFORMS", "UnsupportedOsPlatform", "get_platform", "is_supported"]
