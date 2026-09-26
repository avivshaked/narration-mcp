"""File mechanics: atomic writes, read-only files, removal that never follows a link (design section 15)."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from narration.store import files

from .standin import make_link, remove_link


def test_write_atomic_publishes_whole_read_only_files_s15(tmp_path: Path) -> None:
    path = tmp_path / "take.json"
    digest, size = files.write_atomic(path, b'{"a":1}\n', readonly=True)
    assert path.read_bytes() == b'{"a":1}\n'
    assert (digest, size) == (hashlib.sha256(b'{"a":1}\n').hexdigest(), 8)
    assert files.is_readonly(path)
    assert [p.name for p in tmp_path.iterdir()] == ["take.json"]


def test_write_atomic_replaces_a_read_only_file(tmp_path: Path) -> None:
    path = tmp_path / "engine.json"
    files.write_atomic(path, b"old", readonly=True)
    files.write_atomic(path, b"new", readonly=True)
    assert path.read_bytes() == b"new" and files.is_readonly(path)


def test_a_failed_write_leaves_neither_the_file_nor_a_temp_s15(tmp_path: Path) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"first half"
        raise RuntimeError("the producer failed")

    with pytest.raises(RuntimeError):
        files.write_atomic(tmp_path / "raw.wav", chunks(), readonly=True)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_replace_keeps_the_old_file_whole(tmp_path: Path) -> None:
    path = tmp_path / "measurement.json"
    files.write_atomic(path, b"complete old content", readonly=True)

    def chunks() -> Iterator[bytes]:
        yield b"partial"
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        files.write_atomic(path, chunks(), readonly=True)
    assert path.read_bytes() == b"complete old content"
    assert [p.name for p in tmp_path.iterdir()] == ["measurement.json"]


def test_move_into_makes_the_moved_file_read_only(tmp_path: Path) -> None:
    src = tmp_path / "scratch.wav"
    src.write_bytes(b"audio")
    dst = tmp_path / "raw.wav"
    files.move_into(src, dst, readonly=True)
    assert not src.exists() and dst.read_bytes() == b"audio" and files.is_readonly(dst)


def test_remove_tree_removes_read_only_files(tmp_path: Path) -> None:
    tree = tmp_path / "tk_1"
    (tree / "analyses").mkdir(parents=True)
    files.write_atomic(tree / "delivery.wav", b"x", readonly=True)
    files.write_atomic(tree / "analyses" / "an_1.json", b"{}", readonly=True)
    assert files.tree_size(tree) == 3
    files.remove_tree(tree)
    assert not tree.exists()


def test_remove_tree_never_follows_a_link_out_of_the_store_s17_2(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    precious = outside / "keep.txt"
    precious.write_bytes(b"not the store's")
    tree = tmp_path / "store" / "renders" / "77"
    tree.mkdir(parents=True)
    link = tree / "rn_link"
    if not make_link(link, outside):
        pytest.skip("this machine can make neither a symlink nor a junction")
    try:
        assert files.tree_size(tree) == 0
        files.remove_tree(tree)
    finally:
        remove_link(link)
    assert not tree.exists()
    assert precious.read_bytes() == b"not the store's"


def test_discard_is_quiet_about_a_missing_file(tmp_path: Path) -> None:
    files.discard(tmp_path / "missing")
    assert not os.path.lexists(tmp_path / "missing")
