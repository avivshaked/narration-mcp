"""The platform stand-in's path checks and free-disk answer (``standin.py``), which WP31's tests reach through
``host.platform``. They run on every OS, and give the same answer on each."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Platform

from .standin import DEFAULT_FREE_BYTES, StandInPlatform, check_as_text


def refused(check: object) -> str:
    """The rule of the ``PATH_NOT_ALLOWED`` that ``check()`` raises."""
    assert callable(check)
    with pytest.raises(NarrationError) as info:
        check()
    assert info.value.code == codes.PATH_NOT_ALLOWED
    assert info.value.details is not None
    return str(info.value.details["rule"])


def test_the_stand_in_is_a_whole_platform() -> None:
    assert isinstance(StandInPlatform(), Platform)


def test_a_local_regular_file_is_accepted_and_resolved_s17_3(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"a clip")
    platform = StandInPlatform()
    assert platform.check_readable_path(str(clip)) == Path(os.path.realpath(clip))
    assert platform.paths_checked == [str(clip)]


@pytest.mark.parametrize(
    "path", ["\\\\server\\share\\clip.wav", "//server/share/clip.wav", "\\\\?\\UNC\\server\\share\\clip.wav"]
)
def test_a_network_path_is_refused_as_text_s17_3(path: str) -> None:
    platform = StandInPlatform()
    assert refused(lambda: platform.check_readable_path(path)) == "network"


@pytest.mark.parametrize("path", ["\\\\.\\PhysicalDrive0", "//./pipe/narration", "\\\\?\\C:\\clip.wav"])
def test_a_device_path_is_refused_as_text_s17_3(path: str) -> None:
    platform = StandInPlatform()
    assert refused(lambda: platform.check_readable_path(path)) == "device"


def test_a_reserved_device_name_is_refused_on_every_os_s17_3(tmp_path: Path) -> None:
    platform = StandInPlatform()
    assert refused(lambda: platform.check_readable_path(str(tmp_path / "nul.wav"))) == "reserved_name"


def test_a_relative_path_is_refused_s17_3() -> None:
    platform = StandInPlatform()
    assert refused(lambda: platform.check_readable_path(os.path.join("voices", "clip.wav"))) == "not_absolute"


def test_a_missing_file_and_a_folder_are_refused_s17_3(tmp_path: Path) -> None:
    platform = StandInPlatform()
    assert refused(lambda: platform.check_readable_path(str(tmp_path / "missing.wav"))) == "not_found"
    assert refused(lambda: platform.check_readable_path(str(tmp_path))) == "not_regular_file"


def test_a_path_outside_the_store_is_refused_with_its_rule_s17_2(tmp_path: Path) -> None:
    platform = StandInPlatform()
    root = tmp_path / "store"
    assert platform.check_store_path(root / "renders" / "a.wav", root) == Path(
        os.path.realpath(root / "renders" / "a.wav")
    )
    assert refused(lambda: platform.check_store_path(tmp_path / "elsewhere.wav", root)) == "outside_root"


@pytest.mark.parametrize(
    ("path", "rule"),
    [
        ("/srv/voices/clip.wav", None),
        ("//server/share/clip.wav", "network"),
        ("//./pipe/narration", "device"),
        ("/srv/voices/nul.wav", "reserved_name"),
        ("/srv/voices/a:b.wav", "invalid_name"),
        ("voices/clip.wav", "not_absolute"),
        ("", "empty"),
    ],
)
def test_the_text_rules_on_linux_ci_match_windows_s17_3(path: str, rule: str | None) -> None:
    # The POSIX branch, checked here on every OS: Linux CI runs WP31's tests through it.
    if rule is None:
        check_as_text(path, windows=False)
    else:
        assert refused(lambda: check_as_text(path, windows=False)) == rule


def test_free_disk_bytes_is_the_number_a_test_sets() -> None:
    platform = StandInPlatform()
    assert platform.free_disk_bytes(Path("anywhere")) == DEFAULT_FREE_BYTES
    platform.free_bytes = 1234
    assert platform.free_disk_bytes(Path("store")) == 1234
    assert platform.disk_asked == [Path("anywhere"), Path("store")]
