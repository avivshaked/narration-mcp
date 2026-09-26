"""The store's layout (design section 15) and write confinement (section 17.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from narration.contracts.interfaces import Platform
from narration.store.layout import InvalidIdError, StoreLayout, StorePathError, method_file_stem

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
