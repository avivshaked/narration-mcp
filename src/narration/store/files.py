"""File mechanics of the store: hashing, atomic writes, read-only files, safe removal (design section 15).

Nothing is ever published half written: a file is written under a temporary name (``.tmp-…``) in its own
folder, flushed to disk, and renamed into place with ``os.replace``; a folder of files is assembled under
``.staging-…`` and renamed into place whole. A crash leaves only such temporary names, which no reader
looks at and ``gc`` removes.

Portable: a folder cannot be fsynced on Windows, so that step is attempted and skipped where the OS does
not allow it.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import re
import secrets
import stat
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Final

CHUNK: Final = 1 << 20


def token() -> str:
    """A short random name part for temporary files and folders."""
    return secrets.token_hex(8)


def sha256_file(path: Path) -> tuple[str, int]:
    """The hex sha256 and the size of a file."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def fsync_dir(path: Path) -> None:
    """Flush a folder's entries to disk where the OS allows it (not on Windows, where this is a no-op)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def make_readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def make_writable(path: Path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def is_readonly(path: Path) -> bool:
    return not os.stat(path).st_mode & stat.S_IWRITE


def temp_name(path: Path) -> Path:
    """A temporary sibling of ``path``: same folder (so the rename stays on one volume), never a published
    name."""
    return path.with_name(f".tmp-{token()}-{path.name}")


TRASH_PREFIX: Final = ".trash-"
_TRASH_NAME: Final = re.compile(r"\.trash-([0-9]{9,11})-[0-9a-f]{16}-.+", re.DOTALL)
"""``.trash-<UTC epoch seconds>-<16 hex token>-<name>``. The token of the older layout (``.trash-<16 hex
token>-<name>``) has 16 characters, so it never reads as a time here."""


def trash_name(path: Path, now: float) -> Path:
    """The sibling ``path`` is renamed to before it is removed: ``.trash-<UTC epoch seconds>-<token>-<name>``.

    The time is when the rename happened. A renamed file keeps its own modification time, which for a
    published file replaced today may be months old, so ``gc`` judges a trash name's age by this time
    instead: a trash name another process has just made, and may still rename back, is never collected
    as an old leftover.
    """
    return path.with_name(f"{TRASH_PREFIX}{int(now)}-{token()}-{path.name}")


def trash_time(name: str) -> float | None:
    """The moment written in a trash name (``trash_name``), or None for any other name, including the
    older layout without a time."""
    match = _TRASH_NAME.fullmatch(name)
    return float(match.group(1)) if match else None


def write_temp(
    path: Path, data: bytes | Iterable[bytes], *, readonly: bool, durable: bool = True
) -> tuple[Path, str, int]:
    """Write ``data`` to a temporary sibling of ``path``; returns (the temporary path, sha256 hex, size).

    The caller renames it into place (``publish_temp``) or discards it. ``readonly`` makes it read-only
    before it can appear under its name; ``durable`` flushes the data to disk first.
    """
    tmp = temp_name(path)
    digest = hashlib.sha256()
    size = 0
    try:
        with open(tmp, "xb") as f:
            chunks: Iterable[bytes] = (data,) if isinstance(data, bytes) else data
            for chunk in chunks:
                f.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            f.flush()
            if durable:
                os.fsync(f.fileno())
        if readonly:
            make_readonly(tmp)
    except BaseException:
        discard(tmp)
        raise
    return tmp, digest.hexdigest(), size


REPLACE_ATTEMPTS: Final = 20
"""How often a rename is tried before it fails. On Windows a rename is refused ("access denied") while
another handle has the file, or any file inside the folder, open: a reader of the store (Python opens
files without delete-sharing) or an antivirus scan. Those hold files for milliseconds, so a short retry
(about a second in all) rides it out."""


def _retry_sleep(attempt: int) -> None:
    time.sleep(0.005 * (attempt + 1))


def rename_retrying(src: Path, dst: Path, *, attempts: int = REPLACE_ATTEMPTS) -> None:
    """``os.rename`` (never replacing an existing ``dst``), retried while Windows refuses it because
    something inside ``src`` is open. Any other error is raised at once. ``attempts`` bounds the tries:
    ``gc`` passes 1, so a folder that is in use is listed rather than waited for under the write lock.

    Raises ``FileExistsError`` if ``dst`` exists. Windows refuses such a rename itself, but POSIX
    ``rename`` silently replaces a file or an empty folder, so ``dst`` is checked first on every OS. That
    check is not atomic with the rename. That is safe for the store's own tree, which is renamed only
    under the store's write lock; a caller's source path could lose a race only to another program
    writing that same path at that moment.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    for attempt in range(attempts):
        if os.path.lexists(dst):
            raise FileExistsError(errno.EEXIST, "already exists; not replaced", str(dst))
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            _retry_sleep(attempt)


def read_retrying(path: Path, *, attempts: int = REPLACE_ATTEMPTS) -> bytes:
    """``path.read_bytes()``, retried while Windows refuses to open a file that is being replaced.

    A file the store rewrites in place (``run/daemon.json``, by ``write_atomic``) gets a new file renamed
    over it. On Windows a reader that opens it during that rename gets ``PermissionError``: KNOW (WP30,
    spike g), about one read in 25 while a daemon rewrites the status in a tight loop. The rename takes
    milliseconds, so the read is retried as ``rename_retrying`` retries a rename (about a second in all).
    Any other error is raised at once, and ``PermissionError`` once ``attempts`` tries have failed.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    for attempt in range(attempts - 1):
        try:
            return path.read_bytes()
        except PermissionError:
            _retry_sleep(attempt)
    return path.read_bytes()


def publish_temp(tmp: Path, path: Path) -> None:
    """Rename a finished temporary file into place, replacing a file already there (read-only or not).

    If the rename still fails after the retries, a replaced file that was read-only is made read-only
    again, so an immutable file never stays writable because a publish failed.
    """
    was_readonly = os.path.lexists(path) and not os.path.islink(path) and is_readonly(path)
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            if os.path.lexists(path):
                make_writable(path)
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                if was_readonly:
                    with contextlib.suppress(OSError):
                        make_readonly(path)
                raise
            _retry_sleep(attempt)


def write_atomic(path: Path, data: bytes | Iterable[bytes], *, readonly: bool, durable: bool = True) -> tuple[str, int]:
    """Write ``data`` to ``path`` through a temporary file and a rename; returns (sha256 hex, size).

    ``readonly`` makes the file read-only before it appears under its name, so an immutable file is
    never writable at its published path. An existing file at ``path`` is replaced. ``durable`` flushes
    the data to disk before the rename.
    """
    tmp, digest, size = write_temp(path, data, readonly=readonly, durable=durable)
    try:
        publish_temp(tmp, path)
    except BaseException:
        discard(tmp)
        raise
    if durable:
        fsync_dir(path.parent)
    return digest, size


def move_into(src: Path, dst: Path, *, readonly: bool) -> None:
    """Move a finished file (same volume) to ``dst``, flushing its data to disk first, so that a crash can
    never leave ``dst`` with data that is not on disk.

    The file's modification time becomes the time of the move. A rename keeps the old one, and a caller's
    file that waited days in ``scratch/`` would then look, under a ``.tmp-`` name, like an old leftover
    that ``gc`` may remove before the publish renames it into place.
    """
    with open(src, "r+b") as f:
        os.fsync(f.fileno())
    os.utime(src)
    os.replace(src, dst)
    if readonly:
        make_readonly(dst)


def discard(path: Path) -> None:
    """Remove a file if it exists, read-only or not. A link is removed as a link: ``chmod`` would follow
    it, so it is never called on one."""
    with contextlib.suppress(FileNotFoundError):
        if os.path.islink(path):
            os.unlink(path)
        elif not os.path.isdir(path):
            with contextlib.suppress(OSError):
                make_writable(path)
            os.unlink(path)


def remove_tree(path: Path) -> None:
    """Remove a folder and everything in it, read-only files included.

    A symlink or junction inside it is removed as a link; what it points to is never touched, so a link
    planted in the store cannot make a removal reach outside it. An entry that disappears meanwhile (two
    ``gc`` runs removing the same leftover) is not an error.
    """
    if not os.path.lexists(path):
        return
    if _is_link(path):
        with contextlib.suppress(FileNotFoundError):
            _remove_link(path)
        return
    if not os.path.isdir(path):
        discard(path)
        return
    try:
        with os.scandir(path) as entries:
            children = list(entries)
    except FileNotFoundError:
        return
    for entry in children:
        child = Path(entry.path)
        if entry.is_symlink() or entry.is_junction():
            with contextlib.suppress(FileNotFoundError):
                _remove_link(child)
        elif entry.is_dir(follow_symlinks=False):
            remove_tree(child)
        else:
            discard(child)
    with contextlib.suppress(FileNotFoundError):
        os.rmdir(path)


def _is_link(path: Path) -> bool:
    return os.path.islink(path) or os.path.isjunction(path)


def _remove_link(path: Path) -> None:
    try:
        os.unlink(path)
    except (IsADirectoryError, PermissionError):
        os.rmdir(path)  # a junction, or a directory symlink on Windows


def tree_size(path: Path) -> int:
    """Total size of the regular files under ``path`` (links are not followed)."""
    if not os.path.lexists(path) or _is_link(path):
        return 0
    if not os.path.isdir(path):
        return os.lstat(path).st_size
    total = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_symlink() or entry.is_junction():
                continue
            if entry.is_dir(follow_symlinks=False):
                total += tree_size(Path(entry.path))
            else:
                total += entry.stat(follow_symlinks=False).st_size
    return total
