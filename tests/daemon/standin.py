"""A stand-in for ``narration.platform`` in the daemon's tests, so its logic is tested on every OS.

- ``singleton``: a lock per store folder in this process (a second holder is refused, as the real mutex
  refuses one in the same process); ``hold`` takes it from outside, for "another daemon runs" tests.
- ``kill_on_close_group``: records each pid added (with its creation time), and on close kills those
  processes and their children, checking the creation time first so a reused pid is never touched. Only
  processes the test's own daemon started are ever added.
- ``set_below_normal_priority``: records the pid.
- ``check_store_path``: confinement under the root, as the store's tests do; the rest raise.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import psutil

from narration.contracts import codes
from narration.contracts.errors import NarrationError

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
            raise NarrationError(codes.PATH_NOT_ALLOWED, f"{path} is outside {root}") from None
        return Path(os.path.realpath(path))

    def check_readable_path(self, path: str) -> Path:
        raise NotImplementedError

    def free_disk_bytes(self, path: Path) -> int:
        raise NotImplementedError


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
