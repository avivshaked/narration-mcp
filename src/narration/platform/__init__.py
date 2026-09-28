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
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol

from narration.contracts.interfaces import Platform

from . import winpaths
from ._unsupported import UnsupportedOsPlatform

SUPPORTED_PLATFORMS: Final[tuple[str, ...]] = ("win32",)
"""The values of ``sys.platform`` that have a real implementation."""

DAEMON_RETRY_AFTER_S: Final = 60.0
"""``retry_after_s`` of every ``DAEMON_UNAVAILABLE`` raised because the daemon could not be detached (DC-2 asks
every retryable error for one). The fix needs a person: someone must read the hint, run ``narration-admin
daemon start`` in a terminal, and let the daemon come up. A retry sooner than that fails the same way (lead
ruling, WP19 review). The platform raises it with this value and the front-end (``narration.backend.launch``)
keeps it, so an MCP client sees the figure the operator's terminal does."""

BREAKAWAY_REFUSED: Final = "breakaway_refused"
"""``DAEMON_UNAVAILABLE``'s ``details["reason"]`` when the OS refused the breakaway (on Windows, ``CreateProcess``
with access denied): this process's innermost Job Object forbids it."""

LEFT_IN_JOB: Final = "left_in_job"
"""``DAEMON_UNAVAILABLE``'s ``details["reason"]`` when the OS accepted the breakaway but the daemon was still in
a Job Object: one around this process's innermost job forbids breakaway (``WindowsPlatform.spawn_detached``,
"Nested jobs"). The daemon was ended before it ran its first instruction."""

JOB_CHECK_FAILED: Final = "job_check_failed"
"""``DAEMON_UNAVAILABLE``'s ``details["reason"]`` when the OS could not say whether the daemon was in a job. It
was ended before it ran: a daemon that may die with its client is never let run."""

# The messages of those three refusals. Each says what Windows did and states the rule once ("a daemon runs only
# in no Job Object at all"). They name this process, never a client or a terminal: an MCP client shows them, and
# so does ``narration-admin daemon start``, whose ``refusal_text`` quotes them and adds only the terminal's
# context and the way out.
BREAKAWAY_REFUSED_MESSAGE: Final = (
    "Windows refused to start the daemon detached (access denied). This usually means the Job Object this "
    "process runs in forbids breakaway. A daemon runs only in no Job Object at all, so none was started."
)
"""``DAEMON_UNAVAILABLE``'s message with ``BREAKAWAY_REFUSED``."""

LEFT_IN_JOB_MESSAGE: Final = (
    "Windows started the daemon inside a Job Object it could not leave: this process's innermost job allows "
    "breakaway (a venv launcher's does), but one around it does not, and the daemon stayed in that one. A daemon "
    "runs only in no Job Object at all, so it was ended before it started."
)
"""``DAEMON_UNAVAILABLE``'s message with ``LEFT_IN_JOB``."""

JOB_CHECK_FAILED_MESSAGE: Final = (
    "Windows could not say whether the daemon had left this process's Job Objects ({error}). A daemon runs only "
    "in no Job Object at all, so it was ended before it started."
)
"""``DAEMON_UNAVAILABLE``'s message with ``JOB_CHECK_FAILED``; ``{error}`` is what Windows said
(``JOB_CHECK_FAILED_MESSAGE.format(error=...)``)."""


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


def find_program(name: str) -> str | None:
    """The program ``name`` in the first folder of ``PATH`` that has it, or None; never from the working folder.

    ``shutil.which(name)`` on Windows looks in the working folder before ``PATH`` (unless
    ``NoDefaultCurrentDirectoryInExePath`` is set), so an operator who runs ``narration-admin`` in a folder
    holding a ``uv.exe`` would run that one (``docs/security-review.md``, S2). Here only absolute ``PATH``
    entries are searched (``PATHEXT`` still applies on Windows): an empty or relative entry, which on POSIX
    also means the working folder, is skipped.
    """
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        folder = entry.strip().strip('"')
        if not folder or not os.path.isabs(folder):
            continue
        found = shutil.which(os.path.join(folder, name))
        if found is not None:
            return found
    return None


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
    "BREAKAWAY_REFUSED",
    "BREAKAWAY_REFUSED_MESSAGE",
    "DAEMON_RETRY_AFTER_S",
    "JOB_CHECK_FAILED",
    "JOB_CHECK_FAILED_MESSAGE",
    "LEFT_IN_JOB",
    "LEFT_IN_JOB_MESSAGE",
    "SUPPORTED_PLATFORMS",
    "ProcessPlatform",
    "UnsupportedOsPlatform",
    "find_program",
    "get_platform",
    "is_supported",
    "real_path",
]
