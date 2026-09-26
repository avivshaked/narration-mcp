"""The Windows path rules that need no file system (design sections 17.2 and 17.3). These run on every OS."""

from __future__ import annotations

import os

import pytest

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.platform import real_path, winpaths

BS = "\\"


def p(*parts: str) -> str:
    """A Windows path from its parts, joined with backslashes."""
    return BS.join(parts)


def _rule(exc: pytest.ExceptionInfo[NarrationError]) -> str:
    assert exc.value.code == codes.PATH_NOT_ALLOWED
    assert exc.value.retryable is False
    assert exc.value.details is not None
    return exc.value.details["rule"]


# ---------------------------------------------------------------- reserved device names
@pytest.mark.parametrize(
    "name",
    [
        "CON",
        "con",
        "Nul",
        "nul.txt",
        "NUL.tar.gz",
        "aux .txt",
        "PRN",
        "CONIN$",
        "conout$",
        "COM1",
        "com9.wav",
        "COM0",
        "LPT1",
        "lpt9.log",
        "LPT0",
        "COM\u00b9",
        "com\u00b2.wav",
        "LPT\u00b3",
        "prn:stream",
    ],
)
def test_reserved_device_names_with_any_extension_are_reserved_s17_2(name: str) -> None:
    assert winpaths.is_reserved_name(name)


@pytest.mark.parametrize(
    "name", ["CONSOLE", "con1", "COM10", "LPT", "COM", "nul_", "xcon", "clip.wav", " CON", "CON-x"]
)
def test_ordinary_names_are_not_reserved_s17_2(name: str) -> None:
    assert not winpaths.is_reserved_name(name)


# ---------------------------------------------------------------- one name
@pytest.mark.parametrize(
    ("name", "rule"),
    [
        ("", "invalid_name"),
        (".", "invalid_name"),
        ("..", "invalid_name"),
        ("clip.wav:stream", "invalid_name"),
        ("a<b", "invalid_name"),
        ('a"b', "invalid_name"),
        ("a|b", "invalid_name"),
        ("a?b", "invalid_name"),
        ("a*b", "invalid_name"),
        ("a\tb", "invalid_name"),
        ("trailing.", "invalid_name"),
        ("trailing ", "invalid_name"),
        ("nul.json", "reserved_name"),
    ],
)
def test_name_problems_s17_2(name: str, rule: str) -> None:
    problem = winpaths.name_problem(name)
    assert problem is not None
    assert problem[0] == rule


@pytest.mark.parametrize("name", ["clip.wav", "rn_0123456789abcdef", ".hidden", "a b", "é"])
def test_ordinary_names_have_no_problem_s17_2(name: str) -> None:
    assert winpaths.name_problem(name) is None


# ---------------------------------------------------------------- a caller's path, as text
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (p("C:", "voices", "clip.wav"), ("C:", ("voices", "clip.wav"))),
        ("c:/voices/clip.wav", ("C:", ("voices", "clip.wav"))),
        (p("C:", "a", "..", "b", ".", "clip.wav"), ("C:", ("b", "clip.wav"))),
        (p("C:", "a", "", "clip.wav"), ("C:", ("a", "clip.wav"))),
        (p("C:", ""), ("C:", ())),
    ],
)
def test_absolute_local_paths_parse_s17_3(path: str, expected: tuple[str, tuple[str, ...]]) -> None:
    assert winpaths.parse_absolute(path) == expected


@pytest.mark.parametrize(
    ("path", "rule"),
    [
        ("", "empty"),
        ("C:\\clip\x00.wav", "empty"),
        ("clip.wav", "not_absolute"),
        (p("voices", "clip.wav"), "not_absolute"),
        (p("", "voices", "clip.wav"), "not_absolute"),
        ("C:clip.wav", "not_absolute"),
        (p("1:", "clip.wav"), "not_absolute"),
        (p("", "", "server", "share", "clip.wav"), "network"),
        ("//server/share/clip.wav", "network"),
        (p("", "", "?", "UNC", "server", "share", "clip.wav"), "network"),
        (p("", "", "?", "unc", "server", "share", "clip.wav"), "network"),
        (p("", "??", "UNC", "server", "share", "clip.wav"), "network"),
        (p("", "", "?", "C:", "clip.wav"), "device"),
        (p("", "", ".", "C:", "clip.wav"), "device"),
        (p("", "", ".", "PhysicalDrive0"), "device"),
        (p("", "", ".", "pipe", "x"), "device"),
        ("//./C:/clip.wav", "device"),
        (p("", "??", "C:", "clip.wav"), "device"),
        (p("", "", "?", "Volume{01234567-89ab-cdef-0123-456789abcdef}", "clip.wav"), "device"),
        (p("C:", "voices", "CON"), "reserved_name"),
        (p("C:", "voices", "nul.wav"), "reserved_name"),
        (p("C:", "COM1", "clip.wav"), "reserved_name"),
        (p("C:", "voices", "lpt\u00b9.wav"), "reserved_name"),
        (p("C:", "voices", "clip.wav:stream"), "invalid_name"),
        (p("C:", "voices", "clip?.wav"), "invalid_name"),
        (p("C:", "voices.", "clip.wav"), "invalid_name"),
    ],
)
def test_refused_paths_name_their_rule_s17_3(path: str, rule: str) -> None:
    with pytest.raises(NarrationError) as exc:
        winpaths.parse_absolute(path)
    assert _rule(exc) == rule
    assert exc.value.details is not None
    assert exc.value.details["path"] == path
    assert exc.value.hint == codes.error_code(codes.PATH_NOT_ALLOWED).hint


def test_messages_name_what_was_checked_s17_3() -> None:
    with pytest.raises(NarrationError, match="the target of the link"):
        winpaths.parse_absolute(p("", "", "server", "share"), what="the target of the link")


# ---------------------------------------------------------------- link targets
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (p("", "", "?", "C:", "x"), p("C:", "x")),
        (p("", "??", "D:", "a", "b"), p("D:", "a", "b")),
        (p("", "", "?", "UNC", "server", "share"), p("", "", "server", "share")),
        (p("", "", "?", "unc", "server", "share"), p("", "", "server", "share")),
        (p("", "", "?", "Volume{0}", "x"), p("", "", "?", "Volume{0}", "x")),
        (p("C:", "plain"), p("C:", "plain")),
        ("relative", "relative"),
    ],
)
def test_verbatim_link_targets_become_ordinary_paths_s17_3(target: str, expected: str) -> None:
    assert winpaths.strip_verbatim(target) == expected


# ---------------------------------------------------------------- store paths, as text
ROOT = p("D:", "store")


@pytest.mark.parametrize(
    ("resolved", "expected"),
    [
        # what ntpath.realpath returns for a file replaced between its two looks (WP30, spike g)
        (p("", "", "?", "D:", "store", "run", "daemon.json"), p("D:", "store", "run", "daemon.json")),
        (p("", "", "?", "UNC", "server", "share", "store", "x"), p("", "", "server", "share", "store", "x")),
        (p("D:", "store", "run", "daemon.json"), p("D:", "store", "run", "daemon.json")),
        ("/srv/store/run/daemon.json", "/srv/store/run/daemon.json"),  # a POSIX realpath is never changed
    ],
)
def test_real_path_drops_the_verbatim_prefix_a_replaced_file_keeps_s17_2(
    resolved: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patched:
        patched.setattr(os.path, "realpath", lambda path: resolved)
        got = real_path("anything")
    assert got == expected


def test_a_replaced_file_inside_the_root_compares_inside_once_normalised_s17_2(monkeypatch: pytest.MonkeyPatch) -> None:
    prefixed = p("", "", "?", "D:", "store", "run", "daemon.json")
    assert winpaths.relative_names(prefixed, ROOT) is None  # the refusal WP30 saw
    with monkeypatch.context() as patched:
        patched.setattr(os.path, "realpath", lambda path: prefixed if path == "file" else p("", "", "?", ROOT))
        file, root = real_path("file"), real_path("root")
    assert winpaths.relative_names(file, root) == ("run", "daemon.json")
    assert winpaths.relative_names(p("D:", "elsewhere", "x"), root) is None  # outside stays outside


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (ROOT, ()),
        (p("D:", "store", ""), ()),
        (p("D:", "store", "renders", "ab", "rn_0123456789abcdef"), ("renders", "ab", "rn_0123456789abcdef")),
        (p("d:", "STORE", "jobs"), ("jobs",)),
        ("D:/store/scratch/x.json", ("scratch", "x.json")),
        (p("D:", "store", "a", "..", "b"), ("b",)),
    ],
)
def test_store_paths_inside_the_root_s17_2(path: str, expected: tuple[str, ...]) -> None:
    assert winpaths.split_under_root(path, ROOT) == expected


@pytest.mark.parametrize(
    ("path", "rule"),
    [
        (p("D:", "store2", "x"), "outside_root"),
        (p("D:", "stor"), "outside_root"),
        (p("D:", "store", "..", "x"), "outside_root"),
        (p("D:", "store", "a", "..", "..", "x"), "outside_root"),
        (p("E:", "store", "x"), "outside_root"),
        (p("", "", "server", "share", "store", "x"), "outside_root"),
        (p("store", "x"), "not_absolute"),
        ("D:store", "not_absolute"),
        (p("D:", "store", "renders", "CON"), "reserved_name"),
        (p("D:", "store", "aux.json"), "reserved_name"),
        (p("D:", "store", "x.json:stream"), "invalid_name"),
        (p("D:", "store", "x."), "invalid_name"),
        (p("D:", "store", "x "), "invalid_name"),
        (p("D:", "store", "a*b"), "invalid_name"),
    ],
)
def test_store_paths_refused_as_text_s17_2(path: str, rule: str) -> None:
    with pytest.raises(NarrationError) as exc:
        winpaths.split_under_root(path, ROOT)
    assert _rule(exc) == rule
    assert exc.value.hint == winpaths.STORE_PATH_HINT


def test_the_store_root_itself_is_not_checked_s17_2() -> None:
    root = p("D:", "CON", "store.")
    assert winpaths.split_under_root(p(root, "jobs"), root) == ("jobs",)


def test_relative_names_compare_whole_names_case_insensitively_s17_2() -> None:
    assert winpaths.relative_names(p("D:", "Store", "A"), p("d:", "store")) == ("A",)
    assert winpaths.relative_names(p("D:", "store2"), p("D:", "store")) is None
    assert winpaths.relative_names(p("D:", ""), p("D:", "store")) is None
    assert winpaths.join("D:", ("a", "b")) == p("D:", "a", "b")
