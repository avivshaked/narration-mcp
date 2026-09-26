"""A stand-in for ``narration.platform`` in the daemon's tests, so its logic is tested on every OS.

- ``singleton``: a lock per store folder in this process (a second holder is refused, as the real mutex
  refuses one in the same process); ``hold`` takes it from outside, for "another daemon runs" tests.
- ``kill_on_close_group``: records each pid added (with its creation time), and on close kills those
  processes and their children, checking the creation time first so a reused pid is never touched. Only
  processes the test's own daemon started are ever added.
- ``set_below_normal_priority``: records the pid.
- ``check_store_path``: confinement under the root, as the store's tests do (``PATH_NOT_ALLOWED``, rule
  ``outside_root``).
- ``check_readable_path`` (section 17.3), for WP31's tests through ``host.platform``: the real platform's
  pure text rules (``narration.platform.winpaths``) first, so network and device paths and reserved device
  names are refused on every OS, as the real platform refuses them, before any file is touched; then the
  file itself: it must resolve (links followed) to an existing regular file. It records each path asked
  (``paths_checked``). Unlike the real platform, it does not check the drive type, so a mapped network
  drive is not detected: that is the Windows platform's own test's job.
- ``free_disk_bytes``: ``free_bytes``, which a test may set (``DEFAULT_FREE_BYTES`` until then), for any
  path; it records each path asked (``disk_asked``).
- ``python_for``, ``worker_creationflags``, ``hardening_env`` (``ProcessPlatform``): this OS's real answers
  (``narration.platform.get_platform()``; pure, so safe in tests), so tests start workers as the daemon does.
"""

from __future__ import annotations

import os
import posixpath
import stat
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Final

import psutil

from narration.contracts.errors import NarrationError
from narration.platform import get_platform, winpaths

_REAL = get_platform()

DEFAULT_FREE_BYTES: Final = 500 * 1024**3
"""What ``free_disk_bytes`` answers until a test sets ``free_bytes``: plenty (500 GiB)."""

_HELD: dict[str, threading.Lock] = {}
_HELD_LOCK = threading.Lock()


def _key(root: Path) -> str:
    return os.path.normcase(os.path.realpath(root))


class StandInPlatform:
    """Implements ``narration.contracts.interfaces.Platform`` for tests; see the module docstring."""

    def __init__(self) -> None:
        self.added: list[int] = []
        self.lowered: list[int] = []
        self.groups_opened = 0
        self.groups_closed = 0
        self.spawned: list[tuple[list[str], Path, dict[str, str]]] = []
        self.refuse_spawn: NarrationError | None = None
        self.spawn_pid = 4242
        self.free_bytes: int = DEFAULT_FREE_BYTES
        self.disk_asked: list[Path] = []
        self.paths_checked: list[str] = []

    # ---- the singleton
    @contextmanager
    def singleton(self, store_root: Path) -> Iterator[bool]:
        lock = self._lock_for(store_root)
        acquired = lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()

    @contextmanager
    def hold(self, store_root: Path) -> Iterator[None]:
        """Hold the singleton from outside, as another daemon would."""
        lock = self._lock_for(store_root)
        assert lock.acquire(blocking=False), "the singleton is already held"
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def _lock_for(store_root: Path) -> threading.Lock:
        with _HELD_LOCK:
            return _HELD.setdefault(_key(store_root), threading.Lock())

    # ---- processes
    def spawn_detached(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
        if self.refuse_spawn is not None:
            raise self.refuse_spawn
        self.spawned.append((list(argv), cwd, dict(env)))
        return self.spawn_pid

    def kill_on_close_group(self) -> AbstractContextManager[Callable[[int], None]]:
        return self._group()

    @contextmanager
    def _group(self) -> Iterator[Callable[[int], None]]:
        members: list[tuple[int, float]] = []
        closed = threading.Event()

        def add(pid: int) -> None:
            if closed.is_set():
                raise ValueError("this kill-on-close group is closed")
            members.append((pid, psutil.Process(pid).create_time()))
            self.added.append(pid)

        self.groups_opened += 1
        try:
            yield add
        finally:
            closed.set()
            self.groups_closed += 1
            for pid, created in members:
                _kill_tree(pid, created)

    def set_below_normal_priority(self, pid: int) -> None:
        self.lowered.append(pid)

    # ---- paths
    def check_store_path(self, path: Path, root: Path) -> Path:
        root_abs = Path(os.path.abspath(root))
        try:
            Path(os.path.abspath(path)).relative_to(root_abs)
        except ValueError:
            raise winpaths.path_error("outside_root", f"{path} is outside {root}", str(path)) from None
        return Path(os.path.realpath(path))

    def check_readable_path(self, path: str) -> Path:
        """Section 17.3's check, for tests on any OS (see the module docstring); the path resolved."""
        self.paths_checked.append(path)
        check_as_text(path)
        resolved = Path(os.path.realpath(path))
        try:
            info = os.stat(resolved)
        except (FileNotFoundError, NotADirectoryError):
            raise winpaths.path_error("not_found", "there is no file at the path", path) from None
        except OSError as exc:
            raise winpaths.path_error("unreadable", f"the path cannot be read ({exc.strerror})", path) from exc
        if not stat.S_ISREG(info.st_mode):
            raise winpaths.path_error("not_regular_file", "the path is not a regular file (a folder?)", path)
        return resolved

    def free_disk_bytes(self, path: Path) -> int:
        """``free_bytes``, whatever the path."""
        self.disk_asked.append(path)
        return self.free_bytes

    # ---- how Python processes are started: this OS's real, pure answers
    def python_for(self, python: Path, *, console: bool) -> Path:
        return _REAL.python_for(python, console=console)

    def worker_creationflags(self, *, below_normal: bool) -> int:
        return _REAL.worker_creationflags(below_normal=below_normal)

    def hardening_env(self) -> Mapping[str, str]:
        return _REAL.hardening_env()


def check_as_text(path: str, *, windows: bool = os.name == "nt") -> None:
    """The real platform's text rules, before any file is touched (``PATH_NOT_ALLOWED`` with the rule).

    On Windows (``windows``), ``winpaths.parse_absolute`` itself. Elsewhere the same rules for a POSIX path: a path that
    starts with two slashes of either kind is judged by ``parse_absolute`` (a network or a device path); a
    relative path is refused; every name is checked by ``winpaths.name_problem`` (reserved device names, a
    colon, characters Windows refuses), so a test gets the same answer on every OS.
    """
    if windows:
        winpaths.parse_absolute(path)
        return
    if not path or "\x00" in path:
        raise winpaths.path_error("empty", "the path is empty or contains a NUL character", path)
    if path.replace("/", "\\").startswith("\\\\"):
        winpaths.parse_absolute(path)  # raises: every such path is a network or a device path
    if not posixpath.isabs(path):
        raise winpaths.path_error("not_absolute", "the path is not absolute: give a full path", path)
    for name in (n for n in posixpath.normpath(path).split("/") if n):
        problem = winpaths.name_problem(name)
        if problem is not None:
            raise winpaths.path_error(problem[0], problem[1], path)


def _kill_tree(pid: int, created: float) -> None:
    """Kill ``pid`` and its children, if ``pid`` is still the process that was added (same creation time)."""
    try:
        process = psutil.Process(pid)
        if process.create_time() != created:
            return
        tree = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return
    for member in tree:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(tree, timeout=10)
