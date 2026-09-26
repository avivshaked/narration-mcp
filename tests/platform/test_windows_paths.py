"""The Windows path rules against a real file system (design sections 17.2 and 17.3; WP19), and the free-disk
check. Links to network shares are only ever created, never opened: the rules must refuse them as text.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.platform import get_platform, winpaths

from ._support import WINDOWS_ONLY

if sys.platform != "win32":
    raise pytest.skip.Exception(WINDOWS_ONLY, allow_module_level=True)

import _winapi

from narration.platform import _windows

REMOTE_TARGET = "\\\\example.invalid\\share\\clip.wav"
"""A network path under a reserved top-level domain: it never resolves, so no sign-in could ever be offered."""


def rule_of(exc: pytest.ExceptionInfo[NarrationError]) -> str:
    assert exc.value.code == codes.PATH_NOT_ALLOWED
    assert exc.value.details is not None
    return exc.value.details["rule"]


def same(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    return os.path.normcase(os.fspath(a)) == os.path.normcase(os.fspath(b))


@pytest.fixture
def links() -> Iterator[list[Path]]:
    """Junctions and symlinks a test makes; removed (the link, never its target) afterwards."""
    made: list[Path] = []
    yield made
    for link in reversed(made):
        if os.path.lexists(link):
            if os.lstat(link).st_file_attributes & stat.FILE_ATTRIBUTE_DIRECTORY:
                os.rmdir(link)  # a junction or a directory symlink: removes the link, not its target
            else:
                os.unlink(link)


def junction(target: Path, link: Path, made: list[Path]) -> Path:
    _winapi.CreateJunction(str(target), str(link))
    made.append(link)
    return link


def symlink(target: str | Path, link: Path, made: list[Path], *, directory: bool = False) -> Path:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except OSError as exc:
        if exc.winerror == 1314:  # ERROR_PRIVILEGE_NOT_HELD
            pytest.skip("creating a symbolic link needs Developer Mode or an administrator on this machine")
        raise
    made.append(link)
    return link


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    folder = tmp_path / "voices"
    folder.mkdir()
    path = folder / "clip.wav"
    path.write_bytes(b"RIFF")
    return path


# ---------------------------------------------------------------- readable paths (section 17.3)
def test_an_absolute_local_regular_file_is_accepted_s17_3(clip: Path) -> None:
    platform = get_platform()
    assert same(platform.check_readable_path(str(clip)), clip)
    assert same(platform.check_readable_path(str(clip).replace("\\", "/")), clip)
    assert same(platform.check_readable_path(str(clip.parent / ".." / "voices" / "clip.wav")), clip)


@pytest.mark.parametrize(
    ("make", "rule"),
    [
        (lambda clip: "clip.wav", "not_absolute"),
        (lambda clip: str(clip)[2:], "not_absolute"),
        (lambda clip: str(clip.parent), "not_regular_file"),
        (lambda clip: str(clip.parent / "missing.wav"), "not_found"),
        (lambda clip: str(clip.parent / "missing" / "clip.wav"), "not_found"),
        (lambda clip: str(clip / "inside.wav"), "not_found"),
        (lambda clip: str(clip.parent / "nul.wav"), "reserved_name"),
        (lambda clip: str(clip.parent / "CON"), "reserved_name"),
        (lambda clip: str(clip) + ":stream", "invalid_name"),
    ],
)
def test_paths_that_are_not_local_regular_files_are_refused_s17_3(
    clip: Path, make: Callable[[Path], str], rule: str
) -> None:
    with pytest.raises(NarrationError) as exc:
        get_platform().check_readable_path(make(clip))
    assert rule_of(exc) == rule


@pytest.mark.parametrize(
    ("path", "rule"),
    [
        (REMOTE_TARGET, "network"),
        ("//example.invalid/share/clip.wav", "network"),
        ("\\\\?\\UNC\\example.invalid\\share\\clip.wav", "network"),
        ("\\\\?\\C:\\clip.wav", "device"),
        ("\\\\.\\PhysicalDrive0", "device"),
        ("\\\\.\\pipe\\narration", "device"),
    ],
)
def test_network_and_device_paths_are_refused_before_any_file_access_s17_3(
    path: str, rule: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def untouched(*args: object) -> None:
        raise AssertionError("the file system was consulted for a path the text already refuses")

    monkeypatch.setattr(_windows, "_drive_type", untouched)
    monkeypatch.setattr(_windows, "_resolve_links", untouched)
    with pytest.raises(NarrationError) as exc:
        get_platform().check_readable_path(path)
    assert rule_of(exc) == rule


def test_a_mapped_network_drive_is_refused_before_any_file_access_s17_3(
    clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def untouched(*args: object) -> None:
        raise AssertionError("the path was resolved on a network drive")

    monkeypatch.setattr(_windows, "_drive_type", lambda drive: _windows._DRIVE_REMOTE)
    monkeypatch.setattr(_windows, "_resolve_links", untouched)
    with pytest.raises(NarrationError, match="network drive") as exc:
        get_platform().check_readable_path(str(clip))
    assert rule_of(exc) == "network"


def test_a_missing_drive_is_refused_s17_3(clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_windows, "_drive_type", lambda drive: _windows._DRIVE_NO_ROOT_DIR)
    with pytest.raises(NarrationError) as exc:
        get_platform().check_readable_path(str(clip))
    assert rule_of(exc) == "no_such_drive"


def test_the_real_drive_of_the_test_folder_is_local_s17_3(tmp_path: Path) -> None:
    assert _windows._drive_type(tmp_path.drive) not in (0, 1, _windows._DRIVE_REMOTE)


def test_a_path_through_junctions_resolves_to_its_target_s17_3(clip: Path, tmp_path: Path, links: list[Path]) -> None:
    first = junction(clip.parent, tmp_path / "first", links)
    second = junction(first, tmp_path / "second", links)
    resolved = get_platform().check_readable_path(str(second / "clip.wav"))
    assert same(resolved, clip)


def test_a_junction_to_a_network_target_is_refused_without_following_it_s17_3(
    clip: Path, tmp_path: Path, links: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A junction cannot point at a share, so this one reads back as if it did: the rule must refuse the
    # target as text, before anything opens it.
    link = junction(clip.parent, tmp_path / "share", links)
    real_readlink = os.readlink

    def readlink(path: str | os.PathLike[str]) -> str:
        return "\\\\?\\UNC\\example.invalid\\share" if same(path, link) else real_readlink(path)

    monkeypatch.setattr(os, "readlink", readlink)
    with pytest.raises(NarrationError, match="the target of the link") as exc:
        get_platform().check_readable_path(str(link / "clip.wav"))
    assert rule_of(exc) == "network"


def test_a_junction_loop_is_refused_s17_3(tmp_path: Path, links: list[Path]) -> None:
    b = tmp_path / "b"
    b.mkdir()
    a = junction(b, tmp_path / "a", links)
    b.rmdir()
    junction(a, b, links)
    with pytest.raises(NarrationError) as exc:
        get_platform().check_readable_path(str(a / "clip.wav"))
    assert rule_of(exc) == "link_loop"


def test_a_symlink_to_a_local_file_resolves_to_its_target_s17_3(clip: Path, tmp_path: Path, links: list[Path]) -> None:
    link = symlink(clip, tmp_path / "link.wav", links)
    assert same(get_platform().check_readable_path(str(link)), clip)
    relative = symlink(Path("voices") / "clip.wav", tmp_path / "relative.wav", links)
    assert same(get_platform().check_readable_path(str(relative)), clip)


def test_a_symlink_to_a_network_share_is_refused_without_following_it_s17_3(tmp_path: Path, links: list[Path]) -> None:
    link = symlink(REMOTE_TARGET, tmp_path / "remote.wav", links)
    with pytest.raises(NarrationError, match="the target of the link") as exc:
        get_platform().check_readable_path(str(link))
    assert rule_of(exc) == "network"


def test_a_symlink_to_a_folder_is_not_a_regular_file_s17_3(clip: Path, tmp_path: Path, links: list[Path]) -> None:
    link = symlink(clip.parent, tmp_path / "folder-link", links, directory=True)
    with pytest.raises(NarrationError) as exc:
        get_platform().check_readable_path(str(link))
    assert rule_of(exc) == "not_regular_file"


def test_refusals_carry_the_design_hint_and_the_path_s17_3(clip: Path) -> None:
    missing = str(clip.parent / "missing.wav")
    with pytest.raises(NarrationError) as exc:
        get_platform().check_readable_path(missing)
    assert exc.value.retryable is False
    assert exc.value.hint == codes.error_code(codes.PATH_NOT_ALLOWED).hint
    assert exc.value.details == {"path": missing, "rule": "not_found"}


# ---------------------------------------------------------------- store paths (section 17.2)
@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    return root


def test_paths_inside_the_store_are_returned_resolved_s17_2(store: Path) -> None:
    platform = get_platform()
    nested = store / "renders" / "ab" / "rn_0123456789abcdef" / "raw.wav"
    assert same(platform.check_store_path(nested, store), nested)
    assert same(platform.check_store_path(store, store), store)
    (store / "jobs").mkdir()
    (store / "jobs" / "job.json").write_text("{}", encoding="utf-8")
    assert same(platform.check_store_path(store / "jobs" / "job.json", store), store / "jobs" / "job.json")


@pytest.mark.parametrize(
    ("make", "rule"),
    [
        (lambda store: store.parent / "store2" / "x", "outside_root"),
        (lambda store: store / ".." / "x", "outside_root"),
        (lambda store: store.parent, "outside_root"),
        (lambda store: Path("store") / "x", "not_absolute"),
        (lambda store: store / "renders" / "CON", "reserved_name"),
        (lambda store: store / "com1.json", "reserved_name"),
        (lambda store: Path(str(store / "x.json") + ":stream"), "invalid_name"),
    ],
)
def test_store_paths_outside_or_badly_named_are_refused_s17_2(
    store: Path, make: Callable[[Path], Path], rule: str
) -> None:
    with pytest.raises(NarrationError) as exc:
        get_platform().check_store_path(make(store), store)
    assert rule_of(exc) == rule
    assert exc.value.hint == winpaths.STORE_PATH_HINT


def test_a_junction_inside_the_store_is_refused_even_to_the_store_s17_2(store: Path, links: list[Path]) -> None:
    (store / "renders").mkdir()
    inner = junction(store / "renders", store / "shortcut", links)
    platform = get_platform()
    for path in (inner, inner / "x.wav", inner / "deeper" / "x.wav"):
        with pytest.raises(NarrationError) as exc:
            platform.check_store_path(path, store)
        assert rule_of(exc) == "reparse_point"


def test_a_junction_out_of_the_store_is_refused_s17_2(store: Path, tmp_path: Path, links: list[Path]) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (store / "takes").mkdir()
    junction(outside, store / "takes" / "ab", links)
    with pytest.raises(NarrationError) as exc:
        get_platform().check_store_path(store / "takes" / "ab" / "delivery.wav", store)
    assert rule_of(exc) == "reparse_point"


def test_a_symlinked_file_inside_the_store_is_refused_s17_2(store: Path, tmp_path: Path, links: list[Path]) -> None:
    target = tmp_path / "elsewhere.json"
    target.write_text("{}", encoding="utf-8")
    link = symlink(target, store / "job.json", links)
    with pytest.raises(NarrationError) as exc:
        get_platform().check_store_path(link, store)
    assert rule_of(exc) == "reparse_point"


def test_the_store_root_may_be_reached_through_a_junction_s17_2(store: Path, tmp_path: Path, links: list[Path]) -> None:
    via = junction(store, tmp_path / "via", links)
    checked = get_platform().check_store_path(via / "jobs" / "job.json", via)
    assert same(checked, store / "jobs" / "job.json")


# ---------------------------------------------------------------- free disk (section 15, STORE_FULL)
def test_free_disk_bytes_measures_the_volume_s15(tmp_path: Path) -> None:
    platform = get_platform()
    free = platform.free_disk_bytes(tmp_path)
    usage = shutil.disk_usage(tmp_path)
    assert 0 < free <= usage.total
    assert abs(free - usage.free) < 2**30  # no quota here; other programs may write meanwhile
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    assert platform.free_disk_bytes(tmp_path / "file.txt") > 0
    assert platform.free_disk_bytes(tmp_path / "not" / "yet" / "there") > 0
