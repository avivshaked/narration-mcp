"""The platform for an OS v1 does not support (plan.md Q2: Windows first; WP19).

Every method raises ``narration.contracts.errors.UnsupportedPlatform`` (``DAEMON_UNAVAILABLE``, not
retryable), naming the operation and the OS. The POSIX implementations (an ``fcntl`` lock file,
``start_new_session``, process groups with ``PR_SET_PDEATHSIG``, the ``/proc`` and ``/dev`` path rules) are
later work; they will replace this class in ``narration.platform.get_platform``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from narration.contracts.errors import UnsupportedPlatform


class UnsupportedOsPlatform:
    """A ``Platform`` whose every call raises ``UnsupportedPlatform``; ``system`` is ``sys.platform``."""

    def __init__(self, system: str) -> None:
        self.system = system

    def _refuse(self, operation: str) -> UnsupportedPlatform:
        return UnsupportedPlatform(operation, self.system)

    def singleton(self, store_root: Path) -> AbstractContextManager[bool]:
        """Refused: the daemon's singleton lock is not implemented on this OS."""
        raise self._refuse("singleton")

    def spawn_detached(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
        """Refused: detached start is not implemented on this OS."""
        raise self._refuse("spawn_detached")

    def kill_on_close_group(self) -> AbstractContextManager[Callable[[int], None]]:
        """Refused: kill-on-close groups are not implemented on this OS."""
        raise self._refuse("kill_on_close_group")

    def set_below_normal_priority(self, pid: int) -> None:
        """Refused: process priority is not implemented on this OS."""
        raise self._refuse("set_below_normal_priority")

    def check_readable_path(self, path: str) -> Path:
        """Refused: the read rules of section 17.3 are not implemented on this OS."""
        raise self._refuse("check_readable_path")

    def check_store_path(self, path: Path, root: Path) -> Path:
        """Refused: the write rules of section 17.2 are not implemented on this OS."""
        raise self._refuse("check_store_path")

    def free_disk_bytes(self, path: Path) -> int:
        """Refused: the free-disk check is not implemented on this OS."""
        raise self._refuse("free_disk_bytes")
