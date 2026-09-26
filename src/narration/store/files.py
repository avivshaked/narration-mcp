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
import hashlib
import os
import secrets
import stat
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


def publish_temp(tmp: Path, path: Path) -> None:
    """Rename a finished temporary file into place, replacing a file already there (read-only or not)."""
    if os.path.lexists(path):
        make_writable(path)
    os.replace(tmp, path)


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
    never leave ``dst`` with data that is not on disk."""
    with open(src, "r+b") as f:
        os.fsync(f.fileno())
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
    planted in the store cannot make a removal reach outside it.
    """
    if not os.path.lexists(path):
        return
    if _is_link(path):
        _remove_link(path)
        return
    if not os.path.isdir(path):
        discard(path)
        return
    with os.scandir(path) as entries:
        for entry in list(entries):
            child = Path(entry.path)
            if entry.is_symlink() or entry.is_junction():
                _remove_link(child)
            elif entry.is_dir(follow_symlinks=False):
                remove_tree(child)
            else:
                discard(child)
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
