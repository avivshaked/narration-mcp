"""The store's layout (design section 15) and write confinement (section 17.2)."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

import pytest

from narration import platform as platform_module
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError, UnsupportedPlatform
from narration.contracts.interfaces import Platform
from narration.store import layout as layout_module
from narration.store.layout import InvalidIdError, StoreLayout, StorePathError, is_under, method_file_stem

from .standin import StandInPlatform, make_link, remove_link

RENDER_ID = "rn_77e0c4a1b2d93f08"
TAKE_ID = "tk_8c41d2e07a9b3f55"
ANALYSIS_ID = "an_0f3b91c2d5e7a468"
JOB_ID = "job_01JBXQ7Z3M8V4T2R9K6N5P0W1C"
DESIGN_ID = "01JBYQ7Z3M8V4T2R9K6N5P0W1C"
VOICE_HASH = "sha256:3f9a0c1e" + "0" * 56
AUDIO = "5b1e" + "0" * 60


@pytest.fixture
def layout(tmp_path: Path) -> StoreLayout:
    return StoreLayout(tmp_path / "store", StandInPlatform())


def test_the_stand_in_implements_the_platform_protocol() -> None:
    platform: Platform = StandInPlatform()
    assert isinstance(platform, Platform)


def test_paths_follow_the_layout_of_s15(layout: StoreLayout) -> None:
    root = layout.root
    assert layout.db_path == root / "narration.sqlite"
    assert layout.provenance_path == root / "provenance.jsonl"
    assert layout.render_dir(RENDER_ID) == root / "renders" / "77" / RENDER_ID
    assert layout.take_dir(TAKE_ID) == root / "takes" / "8c" / TAKE_ID
    assert (
        layout.analysis_path(TAKE_ID, ANALYSIS_ID)
        == root / "takes" / "8c" / TAKE_ID / "analyses" / f"{ANALYSIS_ID}.json"
    )
    assert layout.measurement_dir(VOICE_HASH, "qwen3-base-1.7b.p1") == (
        root / "measurements" / VOICE_HASH.removeprefix("sha256:") / "qwen3-base-1.7b.p1"
    )
    assert layout.design_dir(DESIGN_ID, 2) == root / "designs" / DESIGN_ID / "2"
    assert layout.profile_dir(AUDIO) == root / "profiles" / "5b" / AUDIO
    assert layout.job_dir(JOB_ID) == root / "jobs" / JOB_ID
    assert layout.engine_path("qwen3-base-1.7b.p1") == root / "engines" / "qwen3-base-1.7b.p1.json"
    assert layout.canary_clip_path("qwen3-base-1.7b.p1") == root / "engines" / "qwen3-base-1.7b.p1" / "canary.wav"
    assert (
        layout.alignment_path("ctc-forced-align+silence-snap")
        == root / "alignment" / "ctc-forced-align+silence-snap.json"
    )
    assert layout.daemon_json_path() == root / "run" / "daemon.json"


def test_scratch_paths_get_their_folder(layout: StoreLayout) -> None:
    path = layout.scratch_path("voices", f"{AUDIO}.wav")
    assert path == layout.root / "scratch" / "voices" / f"{AUDIO}.wav"
    assert path.parent.is_dir()


def test_relative_paths_round_trip(layout: StoreLayout) -> None:
    path = layout.render_dir(RENDER_ID) / "raw.wav"
    assert layout.rel(path) == f"renders/77/{RENDER_ID}/raw.wav"
    assert layout.abs(layout.rel(path)) == path


def test_method_ids_map_to_safe_distinct_file_names() -> None:
    assert (
        method_file_stem("ctc-snap/wav2vec2-large-960h-lv60-self@abc")
        == "ctc-snap%2Fwav2vec2-large-960h-lv60-self%40abc"
    )
    assert method_file_stem("A") != method_file_stem("a")
    with pytest.raises(InvalidIdError):
        method_file_stem("")


@pytest.mark.parametrize(
    "build",
    [
        lambda lay: lay.render_dir("rn_../../x"),
        lambda lay: lay.render_dir("rn_77E0C4A1B2D93F08"),
        lambda lay: lay.render_dir("tk_77e0c4a1b2d93f08"),
        lambda lay: lay.take_dir("..\\..\\x"),
        lambda lay: lay.analysis_path(TAKE_ID, "an_x/../../y"),
        lambda lay: lay.job_dir("job_01jbxq7z3m8v4t2r9k6n5p0w1c"),
        lambda lay: lay.job_dir("C:\\Windows"),
        lambda lay: lay.design_dir(DESIGN_ID, -1),
        lambda lay: lay.design_dir("../" + DESIGN_ID, 0),
        lambda lay: lay.measurement_dir(VOICE_HASH.removeprefix("sha256:"), "qwen3-base-1.7b.p1"),
        lambda lay: lay.measurement_dir(VOICE_HASH, "../qwen3"),
        lambda lay: lay.measurement_dir(VOICE_HASH, "qwen3."),
        lambda lay: lay.profile_dir("/etc/passwd"),
        lambda lay: lay.engine_path("Qwen3"),
    ],
)
def test_ids_that_break_their_grammar_build_no_path_s17_2(layout: StoreLayout, build: object) -> None:
    with pytest.raises(InvalidIdError):
        build(layout)  # type: ignore[operator]


@pytest.mark.parametrize(
    "parts",
    [("..", "x"), ("a/b",), ("a\\b",), ("C:\\x",), ("/etc",), ("x.",), (".hidden",), ("",)],
)
def test_scratch_components_cannot_escape_s17_2(layout: StoreLayout, parts: tuple[str, ...]) -> None:
    with pytest.raises(StorePathError):
        layout.scratch_path(*parts)


@pytest.mark.parametrize("rel", ["../x", "/x", "renders/../../x", "C:/x", ""])
def test_stored_relative_paths_cannot_escape_s17_2(layout: StoreLayout, rel: str) -> None:
    with pytest.raises(StorePathError):
        layout.abs(rel)


@pytest.mark.parametrize("name", ["con", "nul", "com1", "lpt9"])
def test_windows_reserved_names_are_refused_through_the_platform_s17_2(layout: StoreLayout, name: str) -> None:
    # These match the engine profile grammar, so only the platform's check stops them.
    with pytest.raises(StorePathError, match="reserved"):
        layout.engine_path(name)
    with pytest.raises(StorePathError, match="reserved"):
        layout.scratch_path(name)


def test_every_built_path_is_checked_by_the_platform_s17_2(tmp_path: Path) -> None:
    platform = StandInPlatform()
    layout = StoreLayout(tmp_path / "store", platform)
    path = layout.take_dir(TAKE_ID)
    assert platform.checked[-1] == path


def test_a_link_out_of_the_store_is_refused_s17_2(layout: StoreLayout, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = layout.root / "renders"
    if not make_link(link, outside):
        pytest.skip("this machine can make neither a symlink nor a junction")
    try:
        with pytest.raises(StorePathError, match="outside the store root"):
            layout.render_dir(RENDER_ID)
    finally:
        remove_link(link)
    assert outside.is_dir()


def test_a_link_inside_the_store_is_refused_as_a_reparse_point_s17_2(layout: StoreLayout) -> None:
    target = layout.root / "elsewhere"
    target.mkdir()
    link = layout.root / "takes"
    if not make_link(link, target):
        pytest.skip("this machine can make neither a symlink nor a junction")
    try:
        with pytest.raises(StorePathError, match="link or junction"):
            layout.take_dir(TAKE_ID)
    finally:
        remove_link(link)


# ---------------------------------------------------------------- review fixes: whole matches, long method ids,
# the platform's other errors, and the real platform


@pytest.mark.parametrize(
    "build",
    [
        lambda lay, nl: lay.render_dir(RENDER_ID + nl),
        lambda lay, nl: lay.take_dir(TAKE_ID + nl),
        lambda lay, nl: lay.analysis_path(TAKE_ID, ANALYSIS_ID + nl),
        lambda lay, nl: lay.job_dir(JOB_ID + nl),
        lambda lay, nl: lay.design_dir(DESIGN_ID + nl, 0),
        lambda lay, nl: lay.design_root(DESIGN_ID + nl),
        lambda lay, nl: lay.profile_dir(AUDIO + nl),
        lambda lay, nl: lay.measurement_dir(VOICE_HASH + nl, "qwen3-base-1.7b.p1"),
        lambda lay, nl: lay.measurement_dir(VOICE_HASH, "qwen3-base-1.7b.p1" + nl),
        lambda lay, nl: lay.engine_path("qwen3-base-1.7b.p1" + nl),
    ],
)
def test_an_id_with_a_trailing_newline_builds_no_path_s17_2(layout: StoreLayout, build: Any) -> None:
    build(layout, "")  # the id itself is fine
    with pytest.raises(InvalidIdError):
        build(layout, "\n")


def test_a_component_with_a_trailing_newline_builds_no_path_s17_2(layout: StoreLayout) -> None:
    with pytest.raises(StorePathError):
        layout.scratch_path("job_x\n")
    with pytest.raises(StorePathError):
        layout.abs("renders\n")


def test_the_id_shapes_are_the_contracts_s6() -> None:
    assert layout_module.RENDER_ID_PATTERN.pattern == names.ID_PATTERNS["render_id"]
    assert layout_module.TAKE_ID_PATTERN.pattern == names.ID_PATTERNS["take_id"]
    assert layout_module.ANALYSIS_ID_PATTERN.pattern == names.ID_PATTERNS["analysis_id"]
    assert layout_module.JOB_ID_PATTERN.pattern == names.ID_PATTERNS["job_id"]
    assert layout_module.DESIGN_ID_PATTERN.pattern == names.ID_PATTERNS["design_id"]
    ids: tuple[tuple[names.IdKind, str], ...] = (
        ("render_id", RENDER_ID),
        ("take_id", TAKE_ID),
        ("job_id", JOB_ID),
        ("design_id", DESIGN_ID),
    )
    for kind, value in ids:
        assert names.is_id(kind, value) and not names.is_id(kind, value + "\n")


# A realistic method id of the Qwen3 forced aligner: method, model, revision and snap parameters.
LONG_METHOD = (
    "ctc-forced-align+silence-snap/Qwen/Qwen3-ForcedAligner-0.6B@c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
    "+snap(frame_s=0.02,max_shift_s=0.12,pad=0.080)"
)
LONG_METHOD_STEM = (
    "ctc-forced-align+silence-snap%2F%51wen%2F%51wen3-%46orced%41ligner-0.6%42%40c7cbfc2048c462b0d63a4"
    "%hb6e6a840dd83b5ef71106e3c"
)


def test_a_long_method_id_gets_a_readable_stem_with_a_hash_s11_2(layout: StoreLayout) -> None:
    assert len(LONG_METHOD) == 146
    stem = method_file_stem(LONG_METHOD)
    assert stem == LONG_METHOD_STEM  # stable: the file name of a published benchmark
    assert len(stem) == layout_module.METHOD_STEM_MAX
    assert stem.endswith("%h" + hashlib.sha256(LONG_METHOD.encode("utf-8")).hexdigest()[:24])
    path = layout.alignment_path(LONG_METHOD)
    assert path.name == stem + ".json" and len(path.name) <= layout_module.COMPONENT_MAX
    # Two long ids that differ only past the readable prefix get different names.
    assert method_file_stem(LONG_METHOD.replace("pad=0.080", "pad=0.090")) != stem


def test_a_shortened_stem_never_splits_an_escape_or_meets_a_short_stem() -> None:
    method = "a" * 96 + "/" + "b" * 60  # the cut would fall inside "%2F"
    stem = method_file_stem(method)
    prefix = stem.split("%h", 1)[0]
    assert prefix == "a" * 96 and len(stem) <= layout_module.METHOD_STEM_MAX
    # A stem short enough to keep whole never contains the "%h" mark, whatever the id holds.
    assert "%h" not in method_file_stem("%h/%H" + "x" * 10)
    assert method_file_stem(".hidden") == "%2Ehidden"


class _PlatformRaising:
    """A platform whose ``check_store_path`` raises a chosen error."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def check_store_path(self, path: Path, root: Path) -> Path:
        raise self.error


def test_an_unsupported_platform_stays_daemon_unavailable_s17_2(tmp_path: Path) -> None:
    layout = StoreLayout(tmp_path / "store", _PlatformRaising(UnsupportedPlatform("check_store_path", "plan9")))  # type: ignore[arg-type]
    with pytest.raises(UnsupportedPlatform) as caught:
        layout.render_dir(RENDER_ID)
    assert caught.value.code == codes.DAEMON_UNAVAILABLE


def test_a_platform_refusal_keeps_its_details_s17_2(tmp_path: Path) -> None:
    refusal = NarrationError(codes.PATH_NOT_ALLOWED, "a reparse point", details={"rule": "reparse_point"})
    layout = StoreLayout(tmp_path / "store", _PlatformRaising(refusal))  # type: ignore[arg-type]
    with pytest.raises(StorePathError) as caught:
        layout.render_dir(RENDER_ID)
    assert caught.value.details == {"rule": "reparse_point"}


needs_windows = pytest.mark.skipif(
    not platform_module.is_supported(), reason="narration.platform supports Windows only"
)


@needs_windows
def test_the_real_platform_confines_the_store_s17_2(tmp_path: Path) -> None:
    layout = StoreLayout(tmp_path / "store", platform_module.get_platform())
    assert layout.render_dir(RENDER_ID) == layout.root / "renders" / "77" / RENDER_ID
    with pytest.raises(StorePathError):
        layout.engine_path("con")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = layout.root / "renders"
    if not make_link(link, outside):
        pytest.skip("this machine can make neither a symlink nor a junction")
    try:
        with pytest.raises(StorePathError):
            layout.render_dir(RENDER_ID)
    finally:
        remove_link(link)


@pytest.mark.parametrize(
    ("path", "root", "inside"),
    [
        (PureWindowsPath(r"D:\root\..\..\x"), PureWindowsPath(r"D:\root"), False),
        (PureWindowsPath(r"D:\root\x\.."), PureWindowsPath(r"D:\root"), False),
        (PureWindowsPath(r"D:\root\x"), PureWindowsPath(r"D:\root"), True),
        (PureWindowsPath(r"D:\root"), PureWindowsPath(r"D:\root"), True),
        (PureWindowsPath(r"D:\root2\x"), PureWindowsPath(r"D:\root"), False),
        (PurePosixPath("/root/../../x"), PurePosixPath("/root"), False),
        (PurePosixPath("/root/x"), PurePosixPath("/root"), True),
        (PurePosixPath("/root/x"), PurePosixPath("/root/.."), False),
    ],
)
def test_a_path_with_dot_dot_is_never_inside_s17_2(path: PurePath, root: PurePath, inside: bool) -> None:
    # is_under compares names as text, so D:\root\..\..\x has D:\root among its parents.
    assert is_under(path, root) is inside


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="only Windows' realpath has a verbatim (\\\\?\\) form; tests/platform/test_winpaths.py checks the "
    "string handling on every OS",
)
def test_confine_accepts_a_file_whose_realpath_keeps_the_verbatim_prefix_s17_2(
    layout: StoreLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    # KNOW (WP30, spike g): realpath of a file replaced during the call can come back as \\?\D:\...; confine
    # took that for a path outside the store and refused run/daemon.json.
    target = layout.root / "run" / "daemon.json"
    outside = layout.root.parent / "outside.json"
    original = os.path.realpath

    def verbatim(path: Any, **kwargs: Any) -> str:
        real = original(path, **kwargs)
        return "\\\\?\\" + real if os.fspath(path) in (str(target), str(outside)) else real

    with monkeypatch.context() as patched:
        patched.setattr(os.path, "realpath", verbatim)
        assert layout.daemon_json_path() == target
        with pytest.raises(StorePathError):
            layout.confine(outside)  # a prefixed path outside the store is still outside
