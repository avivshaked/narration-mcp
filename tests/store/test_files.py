"""File mechanics: atomic writes, read-only files, removal that never follows a link (design section 15)."""

from __future__ import annotations

import hashlib
import os
import threading
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


def test_a_reader_holding_the_old_file_only_delays_a_replace(tmp_path: Path) -> None:
    # On Windows the rename is refused while the old file is open; the write waits for the reader to close.
    path = tmp_path / "daemon.json"
    files.write_atomic(path, b'{"state":"idle"}', readonly=False, durable=False)
    reader = open(path, "rb")  # noqa: SIM115 - held open on purpose, closed by the timer
    timer = threading.Timer(0.05, reader.close)
    timer.start()
    try:
        files.write_atomic(path, b'{"state":"busy"}', readonly=False, durable=False)
    finally:
        timer.join(timeout=10)
        reader.close()
    assert path.read_bytes() == b'{"state":"busy"}'


def _refuse_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(src: object, dst: object) -> None:
        raise PermissionError(13, "held open by a reader", str(dst))

    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(files, "_retry_sleep", lambda attempt: None)


def test_a_replace_that_gives_up_leaves_the_old_file_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An immutable file is never left writable because a publish failed.
    path = tmp_path / "engine.json"
    files.write_atomic(path, b"old", readonly=True)
    _refuse_replace(monkeypatch)
    with pytest.raises(PermissionError):
        files.write_atomic(path, b"new", readonly=True)
    assert path.read_bytes() == b"old" and files.is_readonly(path)
    assert [p.name for p in tmp_path.iterdir()] == ["engine.json"]


def test_a_writable_file_stays_writable_when_a_replace_gives_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "daemon.json"
    files.write_atomic(path, b"old", readonly=False)
    _refuse_replace(monkeypatch)
    with pytest.raises(PermissionError):
        files.write_atomic(path, b"new", readonly=False)
    assert path.read_bytes() == b"old" and not files.is_readonly(path)


def test_a_folder_rename_is_retried_while_it_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / ".staging-x"
    src.mkdir()
    real = os.rename
    calls = {"n": 0}

    def flaky(a: object, b: object) -> None:
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(13, "a file inside is open", str(a))
        real(a, b)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "rename", flaky)
    monkeypatch.setattr(files, "_retry_sleep", lambda attempt: None)
    files.rename_retrying(src, tmp_path / "rn_1")
    assert calls["n"] == 4 and (tmp_path / "rn_1").is_dir()


def test_a_folder_rename_gives_up_after_its_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / ".staging-x"
    src.mkdir()
    calls = {"n": 0}

    def refuse(a: object, b: object) -> None:
        calls["n"] += 1
        raise PermissionError(13, "a file inside is open", str(a))

    monkeypatch.setattr(os, "rename", refuse)
    monkeypatch.setattr(files, "_retry_sleep", lambda attempt: None)
    with pytest.raises(PermissionError):
        files.rename_retrying(src, tmp_path / "rn_1")
    assert calls["n"] == files.REPLACE_ATTEMPTS and src.is_dir()


def test_a_rename_onto_an_existing_folder_is_not_retried(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(FileExistsError):
        files.rename_retrying(tmp_path / "a", tmp_path / "b")
    assert (tmp_path / "a").is_dir() and (tmp_path / "b").is_dir()


def test_a_rename_never_replaces_an_existing_file_on_any_os_s15(tmp_path: Path) -> None:
    # POSIX rename replaces a file silently; return_sources would overwrite a caller's new file.
    (tmp_path / "a.wav").write_bytes(b"ours")
    (tmp_path / "b.wav").write_bytes(b"the caller's")
    with pytest.raises(FileExistsError):
        files.rename_retrying(tmp_path / "a.wav", tmp_path / "b.wav")
    assert (tmp_path / "b.wav").read_bytes() == b"the caller's"
    assert (tmp_path / "a.wav").read_bytes() == b"ours"


def test_remove_tree_is_quiet_about_entries_that_vanish(tmp_path: Path) -> None:
    # Two gc runs may remove the same leftover at once.
    files.remove_tree(tmp_path / "never-there")
    tree = tmp_path / ".trash-x"
    (tree / "a").mkdir(parents=True)
    files.remove_tree(tree)
    files.remove_tree(tree)
    assert not tree.exists()
