"""A stand-in for ``narration.platform`` (WP19, written in parallel) in the store's tests.

Only ``check_store_path`` is real: it refuses a path outside the root, a Windows reserved name or a name
ending in a dot or a space in any component, and a symlink or junction anywhere between the root and the
path. It runs on every OS, so the confinement tests run on Linux CI too. Everything else raises.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from narration.contracts import codes
from narration.contracts.errors import NarrationError

_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])$", re.IGNORECASE)


def _refuse(message: str) -> NarrationError:
    return NarrationError(codes.PATH_NOT_ALLOWED, message)


class StandInPlatform:
    """Implements ``narration.contracts.interfaces.Platform`` for tests; see the module docstring."""

    def __init__(self) -> None:
        self.checked: list[Path] = []

    def check_store_path(self, path: Path, root: Path) -> Path:
        self.checked.append(path)
        root_abs = Path(os.path.abspath(root))
        try:
            rel = Path(os.path.abspath(path)).relative_to(root_abs)
        except ValueError:
            raise _refuse(f"{path} is outside {root}") from None
        current = root_abs
        for part in rel.parts:
            if _RESERVED.match(part.split(".", 1)[0].rstrip(" ")) or part.endswith((".", " ")):
                raise _refuse(f"{part!r} is a reserved name")
            current = current / part
            if os.path.islink(current) or os.path.isjunction(current):
                raise _refuse(f"{current} is a link or junction")
        return Path(os.path.realpath(path))

    # ---- not needed by the store
    def singleton(self, store_root: Path) -> AbstractContextManager[bool]:
        raise NotImplementedError

    def spawn_detached(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
        raise NotImplementedError

    def kill_on_close_group(self) -> AbstractContextManager[Callable[[int], None]]:
        raise NotImplementedError

    def set_below_normal_priority(self, pid: int) -> None:
        raise NotImplementedError

    def check_readable_path(self, path: str) -> Path:
        raise NotImplementedError

    def free_disk_bytes(self, path: Path) -> int:
        raise NotImplementedError


def make_link(link: Path, target: Path) -> bool:
    """Create a directory symlink, or on Windows without the privilege a junction; False if neither can be
    made on this machine (the test then skips)."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        pass
    try:
        import _winapi  # CPython's own module; the only unprivileged way to make a junction on Windows

        _winapi.CreateJunction(str(target), str(link))  # pyright: ignore[reportAttributeAccessIssue]
        return True
    except (ImportError, AttributeError, OSError):
        return False


def remove_link(link: Path) -> None:
    """Remove a link made by ``make_link`` without touching its target."""
    if not os.path.lexists(link):
        return
    try:
        os.unlink(link)
    except (IsADirectoryError, PermissionError):
        os.rmdir(link)
