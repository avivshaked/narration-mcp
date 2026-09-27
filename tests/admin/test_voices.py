"""``narration-admin voices allow`` and ``voices list`` (DC-17; design sections 7.1, 16, 17.3, 17.4; plan.md WP45).

Every clip here is a tone the tests write, and every configuration a temporary file; nothing reads the
owner's own ``narration.toml``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import stat
import sys
import threading
import time
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest
import soundfile

import narration.admin.voices as voices
from narration.admin.allowlist import AllowlistEditError, add_hash, comment_text, entries, hand_edit
from narration.admin.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE
from narration.backend import NarrationBackend
from narration.config import Config, load_config
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.names import TOOL_NAMES
from narration.contracts.schemas import TOOLS
from narration.jobs.voice import MAX_CLIP_BYTES, require_synthetic
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore
from tests.backend.support import FakeLauncher, FakeMeasurements, TestPlatform, pins_for
from tests.jobs.support import ENGINE_ID, METHOD_ID, VOICE_TRANSCRIPT, engine_profile
from tests.store.standin import StandInPlatform as StoreStandIn

from .conftest import AdminRun, write_config

OTHER = "b" * 64
ANOTHER = "c" * 64
EXAMPLE = Path(__file__).resolve().parents[2] / "narration.example.toml"


def write_tone(path: Path, *, seconds: float = 2.0, rate: int = 24_000, fmt: str = "WAV") -> str:
    """A synthetic clip (a tone); returns its sha256."""
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), rate, format=fmt)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def clip(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "clips" / "designed-elsewhere.wav"
    return path, write_tone(path)


def allow(admin: AdminRun, config: Path, clip: Path | str, *extra: str, answer: str = "") -> tuple[int, str, str]:
    ran = admin("--config", str(config), "voices", "allow", *extra, str(clip), answer=answer)
    return ran.code, ran.out, ran.err


def settings_without_allowlist(config: Config) -> str:
    """Every setting but ``allow_sha256``, to compare two loads of one file."""
    return repr(dataclasses.replace(config, voices=dataclasses.replace(config.voices, allow_sha256=())))


def leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.endswith(".tmp"))


# ==================================================================== the edit keeps the file (the lead's cases)
COMMENTED = """\
# the owner's configuration
[server]
store_root = "store"   # where results go
models_root = "models"

[voices]
# clips designed before the service existed
allow_sha256 = [
  "{other}",   # first.wav
  # a comment inside the array
]  # a comment after the array

[daemon]
autostart = true
"""


def test_allow_adds_hash_and_keeps_comments_dc_17(admin: AdminRun, tmp_path: Path, clip: tuple[Path, str]) -> None:
    path, sha = clip
    config = tmp_path / "service" / "narration.toml"
    config.parent.mkdir()
    original = COMMENTED.format(other=OTHER)
    config.write_text(original, encoding="utf-8")
    before = load_config(config)
    code, out, _ = allow(admin, config, path, "--yes")
    assert code == EXIT_OK, out
    text = config.read_text(encoding="utf-8")
    added = f'  "{sha}",  # {path.resolve()}\n'
    assert text == original.replace("]  # a comment after", added + "]  # a comment after"), text
    after = load_config(config)
    assert after.voices.allow_sha256 == (OTHER, sha)
    assert settings_without_allowlist(after) == settings_without_allowlist(before)
    assert leftovers(config.parent) == []


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        ("[]", '[\n  "{sha}",  # {note}\n]'),
        ('["{other}"]', '["{other}",\n  "{sha}",  # {note}\n]'),
        ('[ "{other}", ]', '[ "{other}",\n  "{sha}",  # {note}\n]'),
        ('[\n    "{other}"  # first.wav\n]', '[\n    "{other}",  # first.wav\n    "{sha}",  # {note}\n]'),
        ('[\n  "{other}"]', '[\n  "{other}",\n  "{sha}",  # {note}\n]'),
        ("[  # the list\n]", '[  # the list\n  "{sha}",  # {note}\n]'),
    ],
    ids=["empty", "one-line", "one-line-trailing-comma", "last-item-without-comma", "bracket-on-item-line", "comment"],
)
def test_allow_edits_one_line_empty_and_multi_line_arrays_dc_17(array: str, expected: str) -> None:
    note = "clip one.wav"
    text = (
        f'[server]\nstore_root = "s"\nmodels_root = "m"\n[voices]\nallow_sha256 = {array.format(other=OTHER)}\n'
        "[daemon]\nautostart = true\n"
    )
    edited = add_hash(text, "a" * 64, note)
    assert edited is not None
    assert edited == text.replace(array.format(other=OTHER), expected.format(other=OTHER, sha="a" * 64, note=note))
    listed = tomllib.loads(edited)["voices"]["allow_sha256"]
    assert listed == ([OTHER] if "{other}" in array else []) + ["a" * 64]


def test_allow_adds_a_missing_key_below_the_voices_header_comments_dc_17(
    admin: AdminRun, tmp_path: Path, clip: tuple[Path, str]
) -> None:
    path, sha = clip
    extra = "\n[voices]\n# clips designed elsewhere\n\n# the daemon's settings\n[daemon]\nautostart = true\n"
    config = write_config(tmp_path / "service", extra)
    original = config.read_text(encoding="utf-8")
    assert allow(admin, config, path, "--yes")[0] == EXIT_OK
    block = f'allow_sha256 = [\n  "{sha}",  # {path.resolve()}\n]\n'
    assert config.read_text(encoding="utf-8") == original.replace(
        "# clips designed elsewhere\n", "# clips designed elsewhere\n" + block
    )
    assert load_config(config).voices.allow_sha256 == (sha,)


@pytest.mark.parametrize("final_newline", [True, False], ids=["newline-at-end", "no-newline-at-end"])
def test_allow_adds_a_missing_voices_table_at_the_end_dc_17(
    admin: AdminRun, tmp_path: Path, clip: tuple[Path, str], final_newline: bool
) -> None:
    path, sha = clip
    config = tmp_path / "service" / "narration.toml"
    config.parent.mkdir()
    original = '[server]\nstore_root = "store"\nmodels_root = "models"\n\n[daemon]\nidle_exit_min = 5'
    original += "\n" if final_newline else ""
    config.write_text(original, encoding="utf-8")
    before = load_config(config)
    assert allow(admin, config, path, "--yes")[0] == EXIT_OK
    text = config.read_text(encoding="utf-8")
    assert text.startswith(original)
    assert text[len(original) :] == ("" if final_newline else "\n") + (
        f'\n[voices]\nallow_sha256 = [\n  "{sha}",  # {path.resolve()}\n]\n'
    )
    after = load_config(config)
    assert after.voices.allow_sha256 == (sha,)
    assert settings_without_allowlist(after) == settings_without_allowlist(before)


def test_allow_keeps_crlf_line_endings_and_the_tables_after_voices_dc_17(
    admin: AdminRun, tmp_path: Path, clip: tuple[Path, str]
) -> None:
    path, sha = clip
    config = tmp_path / "service" / "narration.toml"
    config.parent.mkdir()
    lines = [
        "[server]",
        'store_root = "store"',
        'models_root = "models"',
        "",
        "[voices]",
        "allow_sha256 = [",
        f'  "{OTHER}",  # first.wav',
        "]",
        "[daemon]",
        "autostart = false",
        "[limits]",
        "max_clip_seconds = 20",
        "",
    ]
    original = "\r\n".join(lines).encode("utf-8")
    config.write_bytes(original)
    before = load_config(config)
    assert allow(admin, config, path, "--yes")[0] == EXIT_OK
    data = config.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n"), "every line still ends in CRLF"
    new_line = f'  "{sha}",  # {path.resolve()}\r\n'.encode()
    assert data == original.replace(b"]\r\n[daemon]", new_line + b"]\r\n[daemon]")
    after = load_config(config)
    assert after.voices.allow_sha256 == (OTHER, sha)
    assert (after.daemon.autostart, after.limits.max_clip_seconds) == (False, 20)
    assert settings_without_allowlist(after) == settings_without_allowlist(before)


SHA = "a" * 64


@pytest.mark.parametrize(
    ("text", "old", "new"),
    [
        (
            f'[server]\r\nstore_root = "s"\n[voices]\nallow_sha256 = [\n  "{OTHER}",\n]\n',
            f'"{OTHER}",\n]',
            f'"{OTHER}",\n  "{SHA}",  # x.wav\n]',
        ),
        (
            f'[server]\nstore_root = "s"\r\n[voices]\r\nallow_sha256 = [\r\n  "{OTHER}",\r\n]\r\n',
            f'"{OTHER}",\r\n]',
            f'"{OTHER}",\r\n  "{SHA}",  # x.wav\r\n]',
        ),
        (
            '[server]\r\nstore_root = "s"\n[voices]\n',
            "[voices]\n",
            f'[voices]\nallow_sha256 = [\n  "{SHA}",  # x.wav\n]\n',
        ),
        (
            '[server]\nstore_root = "s"\r\n',
            '"s"\r\n',
            f'"s"\r\n\r\n[voices]\r\nallow_sha256 = [\r\n  "{SHA}",  # x.wav\r\n]\r\n',
        ),
    ],
    ids=["lf-array-crlf-first-line", "crlf-array-lf-first-line", "lf-header", "crlf-last-line"],
)
def test_new_lines_end_as_the_line_where_they_go_in_dc_17(text: str, old: str, new: str) -> None:
    """The review's TOML F3: in a file with mixed endings, the edit takes the ending of the edit site."""
    assert text.count(old) == 1
    assert add_hash(text, SHA, "x.wav") == text.replace(old, new)


def test_allow_stores_lower_case_and_compares_without_case_dc_17() -> None:
    text = "[voices]\nallow_sha256 = []\n"
    upper = ("ab" * 32).upper()
    edited = add_hash(text, upper, "x.wav")
    assert edited is not None
    assert tomllib.loads(edited)["voices"]["allow_sha256"] == ["ab" * 32]
    assert add_hash(edited, upper, "x.wav") is None, "the same hash in another case is already listed"
    listed_upper = f'[voices]\nallow_sha256 = ["{upper}"]\n'
    assert add_hash(listed_upper, "ab" * 32, "x.wav") is None
    with pytest.raises(AllowlistEditError):
        add_hash(text, "not-a-hash", "x.wav")


def test_an_upper_case_hash_in_the_file_is_refused_as_the_service_refuses_it_dc_17(
    admin: AdminRun, tmp_path: Path, clip: tuple[Path, str]
) -> None:
    path, _ = clip
    config = write_config(tmp_path / "service", f'[voices]\nallow_sha256 = ["{OTHER.upper()}"]\n')
    original = config.read_bytes()
    code, out, err = allow(admin, config, path, "--yes")
    assert code == EXIT_USAGE
    assert "lower-case" in err and out == ""
    assert config.read_bytes() == original


def test_the_whole_example_configuration_keeps_every_line_and_value_dc_17(
    admin: AdminRun, tmp_path: Path, clip: tuple[Path, str]
) -> None:
    path, sha = clip
    config = tmp_path / "service" / "narration.toml"
    config.parent.mkdir()
    original = EXAMPLE.read_bytes()
    config.write_bytes(original)
    before = load_config(config)
    assert allow(admin, config, path, "--yes")[0] == EXIT_OK
    text = config.read_bytes().decode("utf-8")
    old_lines = original.decode("utf-8").splitlines()
    new_lines = text.splitlines()
    assert [line for line in old_lines if line != "allow_sha256 = []"] == [
        line for line in new_lines if line not in {"allow_sha256 = [", f'  "{sha}",  # {path.resolve()}', "]"}
    ]
    after = load_config(config)
    assert after.voices.allow_sha256 == (sha,)
    assert settings_without_allowlist(after) == settings_without_allowlist(before)


@pytest.mark.parametrize(
    "text",
    [
        '[voice_design]\ndesign_text = """\n[voices]\nallow_sha256 = []\n"""\n[voices]\nallow_sha256 = []\n',
        "[voice_design]\ndesign_text = '''\n[voices] # not a header\n'''\n[voices]\nallow_sha256 = [] # the list\n",
        '[ "voices" ]  # quoted\n"allow_sha256" = [\'' + OTHER + "']\n",
        "voices.allow_sha256 = []\n[server]\nstore_root = 's'\n",
        'when = 1979-05-27 07:32:00Z\ntext = "a [b] # c \\" d"\n[voices]\nallow_sha256 = [ # x ]\n]\n',
        '[server]\nstore_root = "s"\n[workers.qa]\nproject = "q"\n[voices]\n    allow_sha256 = ["' + OTHER + '"]\n',
    ],
    ids=["basic-multiline-decoy", "literal-multiline-decoy", "quoted-keys", "dotted-key", "tricky-values", "indented"],
)
def test_allow_follows_toml_and_is_not_fooled_by_text_inside_strings_dc_17(text: str) -> None:
    edited = add_hash(text, "a" * 64, "clip.wav")
    assert edited is not None
    expected = tomllib.loads(text)
    expected["voices"]["allow_sha256"].append("a" * 64)
    assert tomllib.loads(edited) == expected
    assert [e.note for e in entries(edited)][-1] == "clip.wav"


def test_the_note_cannot_break_out_of_its_comment_dc_17() -> None:
    text = "[voices]\nallow_sha256 = []\n"
    sneaky = "clip.wav\n[server]\nstore_root = 'elsewhere'\u2028\x00"
    edited = add_hash(text, "a" * 64, sneaky)
    assert edited is not None
    assert tomllib.loads(edited) == {"voices": {"allow_sha256": ["a" * 64]}}
    assert "\n[server]" not in edited
    assert comment_text("a\tb\nc") == "a b\ufffdc"


def test_an_inline_voices_table_is_refused_with_a_value_that_pastes_into_valid_toml_dc_17(
    admin: AdminRun, tmp_path: Path, clip: tuple[Path, str]
) -> None:
    path, sha = clip
    config = tmp_path / "service" / "narration.toml"
    config.parent.mkdir()
    inline = "voices = { allow_sha256 = [] }\n"
    config.write_text(f'{inline}[server]\nstore_root = "s"\nmodels_root = "m"\n', encoding="utf-8")
    original = config.read_bytes()
    code, _, err = allow(admin, config, path, "--yes")
    assert code == EXIT_FAILED
    assert config.read_bytes() == original
    assert "nothing changed" in err and "inside the brackets" in err
    value = err.rstrip().splitlines()[-1].strip()
    assert value == f'"{sha}"', "the bare value: a comment would swallow the rest of the one-line table"
    pasted = inline.replace("[]", f"[{value}]")
    assert tomllib.loads(pasted) == {"voices": {"allow_sha256": [sha]}}


@pytest.mark.parametrize(
    ("text", "kind", "hint"),
    [
        ("[voices]\nallow_sha256 = [\n]\n", "line", '"{sha}",  # clip.wav'),
        ('[voices]\nallow_sha256 = ["{other}"]\n', "value", '"{sha}"'),
        ("voices = {{ allow_sha256 = [] }}\n", "value", '"{sha}"'),
        ("[voices]\n", "key", 'allow_sha256 = ["{sha}"]'),
        ("[server]\n", "key", 'allow_sha256 = ["{sha}"]'),
        ("not toml = = =\n", "value", '"{sha}"'),
        (None, "value", '"{sha}"'),
    ],
    ids=["multi-line", "one-line", "inline-table", "no-key", "no-table", "not-toml", "unreadable"],
)
def test_the_hand_edit_fits_how_the_file_writes_the_list_dc_17(text: str | None, kind: str, hint: str) -> None:
    sha = "a" * 64
    found = hand_edit(None if text is None else text.format(other=OTHER), sha.upper(), "clip.wav")
    assert (found.kind, found.text) == (kind, hint.format(sha=sha))


# ==================================================================== the confirmation
@pytest.mark.parametrize(
    "answer",
    ["", "no\n", "\n", "yess\n", "  n  \n", "maybe yes\n", "y\n", " Y \n"],
    ids=["eof", "no", "empty", "typo", "n", "not-yes", "y", "Y"],
)
def test_an_unconfirmed_run_changes_nothing_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str], answer: str
) -> None:
    path, _ = clip
    original = config_path.read_bytes()
    code, out, err = allow(admin, config_path, path, answer=answer)
    assert code == EXIT_FAILED
    assert "Is this clip synthetic?" in out
    assert "nothing changed" in err
    if answer:
        assert "answer yes" in err
    assert config_path.read_bytes() == original
    assert leftovers(config_path.parent) == []


def test_no_answer_says_a_person_must_confirm_and_does_not_steer_to_yes_flag_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str]
) -> None:
    path, _ = clip
    code, _, err = allow(admin, config_path, path)  # stdin at its end, as in an agent's shell
    assert code == EXIT_FAILED
    assert "A person must confirm" in err and "must not allow a clip" in err
    assert "--yes is only for the operator's own scripts" in err
    assert "pass --yes" not in err


def test_a_closed_or_missing_stdin_changes_nothing_dc_17(
    config_path: Path, clip: tuple[Path, str], platform: StandInPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narration.admin.__main__ import main

    path, _ = clip
    original = config_path.read_bytes()
    closed = io.StringIO("yes\n")
    closed.close()
    argv = ["--config", str(config_path), "voices", "allow", str(path)]
    err = io.StringIO()
    assert main(argv, out=io.StringIO(), err=err, inp=closed, platform=lambda: platform, environ={}) == EXIT_FAILED
    assert "no answer" in err.getvalue()
    monkeypatch.setattr(sys, "stdin", None)  # pythonw: no standard input at all
    err = io.StringIO()
    assert main(argv, out=io.StringIO(), err=err, platform=lambda: platform, environ={}) == EXIT_FAILED
    assert "no answer" in err.getvalue()
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("answer", ["yes\n", "YES\r\n", "  yes \n", "Yes"])
def test_an_explicit_yes_allows_the_clip_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str], answer: str
) -> None:
    path, sha = clip
    code, out, _ = allow(admin, config_path, path, answer=answer)
    assert code == EXIT_OK, out
    assert load_config(config_path).voices.allow_sha256 == (sha,)


def test_yes_flag_skips_the_question_dc_17(admin: AdminRun, config_path: Path, clip: tuple[Path, str]) -> None:
    path, sha = clip
    code, out, _ = allow(admin, config_path, path, "--yes")
    assert code == EXIT_OK
    assert "Is this clip synthetic?" not in out
    assert load_config(config_path).voices.allow_sha256 == (sha,)


def test_allow_prints_the_clips_path_length_and_sha256_before_asking_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str]
) -> None:
    path, sha = clip
    _, out, _ = allow(admin, config_path, path, answer="no\n")
    question = out.index("Is this clip synthetic?")
    assert out.index(str(path.resolve())) < question
    assert out.index("2.00 s (24000 Hz, 1 channel)") < question
    assert out.index(sha) < question
    assert out.index(str(config_path)) < question


# ==================================================================== what is listed already, what to restart
def test_a_hash_already_listed_changes_nothing_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str]
) -> None:
    path, sha = clip
    assert allow(admin, config_path, path, "--yes")[0] == EXIT_OK
    once = config_path.read_bytes()
    code, out, _ = allow(admin, config_path, path)  # no answer to give: it must not ask
    assert code == EXIT_OK
    assert "already in [voices] allow_sha256" in out and "Is this clip synthetic?" not in out
    assert config_path.read_bytes() == once
    assert load_config(config_path).voices.allow_sha256 == (sha,)


@pytest.mark.parametrize("autostart", [True, False])
def test_allow_ends_by_saying_what_to_restart_dc_17(
    admin: AdminRun, tmp_path: Path, clip: tuple[Path, str], autostart: bool
) -> None:
    path, _ = clip
    config = write_config(tmp_path / "service", f"[daemon]\nautostart = {str(autostart).lower()}\n")
    _, out, _ = allow(admin, config, path, "--yes")
    advice = out[out.index("Added it to") :]
    assert "narration-mcp" in advice and "measure_voice" in advice
    assert "narration-admin daemon stop" in advice
    assert ("narration-admin daemon start" in advice) is not autostart
    assert str(config) in advice
    # The daemon first: a front-end restarted before it would admit jobs the old daemon then fails.
    assert advice.index("daemon stop") < advice.index("restart or reconnect")


def test_allow_warns_about_a_clip_the_limit_would_refuse_dc_17(
    admin: AdminRun, tmp_path: Path, clip: tuple[Path, str]
) -> None:
    path, sha = clip
    config = write_config(tmp_path / "service", "[limits]\nmax_clip_seconds = 1\n")
    code, _, err = allow(admin, config, path, "--yes")
    assert code == EXIT_OK
    assert "max_clip_seconds is 1" in err and "UNSUPPORTED_AUDIO" in err
    assert load_config(config).voices.allow_sha256 == (sha,)


# ==================================================================== the clip: section 17.3's check, a readable WAV
def test_allow_reads_the_clip_through_the_path_check_s17_3(
    admin: AdminRun,
    config_path: Path,
    clip: tuple[Path, str],
    platform: StandInPlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, sha = clip
    monkeypatch.chdir(path.parent)
    assert allow(admin, config_path, path.name, "--yes")[0] == EXIT_OK
    assert platform.paths_checked == [str(path)], "a relative path is taken from the working folder, then checked"
    assert load_config(config_path).voices.allow_sha256 == (sha,)


def test_allow_refuses_a_network_path_before_reading_anything_s17_3(
    admin: AdminRun, config_path: Path, platform: StandInPlatform
) -> None:
    original = config_path.read_bytes()
    code, _, err = allow(admin, config_path, r"\\server\share\clip.wav", "--yes")
    assert code == EXIT_FAILED
    assert codes.PATH_NOT_ALLOWED in err
    assert config_path.read_bytes() == original


def _not_wav(tmp_path: Path, kind: str) -> Path:
    path = tmp_path / "clips" / f"{kind}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "text":
        path.write_text("not audio at all", encoding="utf-8")
    elif kind == "empty-file":
        path.write_bytes(b"")
    elif kind == "flac":
        write_tone(path, fmt="FLAC")
    elif kind == "no-audio":
        soundfile.write(str(path), np.zeros(0, dtype=np.float32), 24_000, format="WAV")
    elif kind == "too-large":
        with path.open("wb") as handle:
            handle.truncate(MAX_CLIP_BYTES + 1)
    elif kind == "folder":
        path.mkdir()
    return path


@pytest.mark.parametrize(
    ("kind", "says"),
    [
        ("text", "not an audio file"),
        ("empty-file", "not an audio file"),
        ("flac", "FLAC file, not a WAV"),
        ("no-audio", "no audio"),
        ("too-large", "20 MB"),
        ("missing", codes.PATH_NOT_ALLOWED),
        ("folder", codes.PATH_NOT_ALLOWED),
    ],
)
def test_allow_refuses_a_file_that_is_not_a_readable_wav_dc_17(
    admin: AdminRun, config_path: Path, tmp_path: Path, kind: str, says: str
) -> None:
    path = tmp_path / "clips" / "missing.wav" if kind == "missing" else _not_wav(tmp_path, kind)
    original = config_path.read_bytes()
    code, _, err = allow(admin, config_path, path, "--yes")
    assert code == EXIT_FAILED
    assert says in err and "nothing changed" in err.lower()
    assert config_path.read_bytes() == original


# ==================================================================== the configuration file and the write
def test_allow_refuses_a_configuration_that_does_not_load_dc_17(
    admin: AdminRun, tmp_path: Path, clip: tuple[Path, str]
) -> None:
    path, _ = clip
    config = write_config(tmp_path / "service", "[voices]\nallow_sha256 = [\n")
    original = config.read_bytes()
    code, out, _ = allow(admin, config, path, "--yes")
    assert code == EXIT_USAGE and out == ""
    assert config.read_bytes() == original


@pytest.fixture
def read_only_config(config_path: Path) -> Iterator[Path]:
    os.chmod(config_path, stat.S_IREAD)
    if os.access(config_path, os.W_OK):  # POSIX root may write a read-only file
        os.chmod(config_path, stat.S_IREAD | stat.S_IWRITE)
        pytest.skip("this user may write a read-only file (root), so there is no read-only file to refuse")
    try:
        yield config_path
    finally:
        os.chmod(config_path, stat.S_IREAD | stat.S_IWRITE)


def test_allow_refuses_a_read_only_configuration_before_asking_dc_17(
    admin: AdminRun, read_only_config: Path, clip: tuple[Path, str]
) -> None:
    path, sha = clip
    original = read_only_config.read_bytes()
    code, out, err = allow(admin, read_only_config, path, answer="yes\n")
    assert code == EXIT_FAILED
    assert "read-only" in err and f'allow_sha256 = ["{sha}"]' in err  # the file has no [voices] yet
    assert "Is this clip synthetic?" not in out
    assert read_only_config.read_bytes() == original


def test_allow_writes_a_temporary_file_and_renames_it_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = clip
    renames: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        renames.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(voices.os, "replace", replace)
    assert allow(admin, config_path, path, "--yes")[0] == EXIT_OK
    [(src, dst)] = renames
    assert dst == config_path and src.parent == config_path.parent and src.name.endswith(".tmp")
    assert not src.exists() and leftovers(config_path.parent) == []


def test_a_failed_rename_leaves_the_file_as_it_was_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = clip
    original = config_path.read_bytes()

    def refuse(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(voices.os, "replace", refuse)
    code, _, err = allow(admin, config_path, path, "--yes")
    assert code == EXIT_FAILED
    assert "cannot write" in err and f'allow_sha256 = ["{sha}"]' in err
    assert config_path.read_bytes() == original
    assert leftovers(config_path.parent) == []


def test_a_rename_refused_for_a_moment_is_retried_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = clip
    real_replace = os.replace
    tries: list[int] = []

    def busy_once(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        tries.append(1)
        if len(tries) == 1:
            raise PermissionError(13, "The process cannot access the file")  # Windows: another program has it open
        real_replace(src, dst)

    monkeypatch.setattr(voices.os, "replace", busy_once)
    assert allow(admin, config_path, path, "--yes")[0] == EXIT_OK
    assert len(tries) == 2
    assert load_config(config_path).voices.allow_sha256 == (sha,)
    assert leftovers(config_path.parent) == []


def test_a_save_made_while_the_rename_is_retried_is_not_overwritten_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's F1: the file is read again before every rename attempt, not only before the first."""
    path, _ = clip
    saved = config_path.read_text(encoding="utf-8") + "[retention]\nretention_days = 7\n"
    real_replace = os.replace
    tries: list[int] = []

    def editor_saves_then_releases(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        tries.append(1)
        if len(tries) == 1:  # the editor holds the file, and saves it before the next attempt
            config_path.write_text(saved, encoding="utf-8")
            raise PermissionError(13, "The process cannot access the file")
        real_replace(src, dst)

    monkeypatch.setattr(voices.os, "replace", editor_saves_then_releases)
    code, _, err = allow(admin, config_path, path, "--yes")
    assert code == EXIT_FAILED and "changed while this command ran" in err
    assert len(tries) == 1, "no second rename once the file changed"
    assert config_path.read_text(encoding="utf-8") == saved
    assert leftovers(config_path.parent) == []


@pytest.mark.skipif(
    sys.platform != "win32", reason="only Windows refuses to rename over a file another program has open"
)
def test_an_editor_holding_the_file_open_keeps_its_save_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's reproduction of F1, with a real open file: a thread holds the configuration open (so the
    rename is refused and retried), writes its own content 50 ms later, and closes it 100 ms after that."""
    path, _ = clip
    saved = (config_path.read_text(encoding="utf-8") + "[retention]\nretention_days = 7\n").encode("utf-8")
    real_copymode = voices.shutil.copymode
    editors: list[threading.Thread] = []

    def edit(handle: BinaryIO) -> None:
        with handle:
            time.sleep(0.05)
            handle.seek(0)
            handle.write(saved)
            handle.truncate()
            handle.flush()
            time.sleep(0.1)

    def open_in_an_editor(src: Path, dst: Path) -> None:
        real_copymode(src, dst)  # just before the command's check and rename
        editor = threading.Thread(target=edit, args=(config_path.open("r+b"),))
        editor.start()
        editors.append(editor)

    monkeypatch.setattr(voices.shutil, "copymode", open_in_an_editor)
    try:
        code, _, err = allow(admin, config_path, path, "--yes")
    finally:
        for editor in editors:
            editor.join(timeout=5)
    assert code == EXIT_FAILED and "changed while this command ran" in err
    assert config_path.read_bytes() == saved
    assert leftovers(config_path.parent) == []


def test_a_file_changed_during_the_edit_is_not_overwritten_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = clip
    meanwhile = config_path.read_text(encoding="utf-8") + "[retention]\nretention_days = 7\n"
    real_copymode = voices.shutil.copymode

    def someone_edits(src: Path, dst: Path) -> None:
        real_copymode(src, dst)
        config_path.write_text(meanwhile, encoding="utf-8")

    monkeypatch.setattr(voices.shutil, "copymode", someone_edits)
    code, _, err = allow(admin, config_path, path, "--yes")
    assert code == EXIT_FAILED and "changed while this command ran" in err
    assert config_path.read_text(encoding="utf-8") == meanwhile
    assert leftovers(config_path.parent) == []


# ==================================================================== acceptance: measure_voice after a restart
def _front_end(config_path: Path) -> tuple[NarrationBackend, NarrationStore]:
    """``narration-mcp``'s backend as it starts: the configuration read from the file, the store opened."""
    config = load_config(config_path)
    root, models = config.server.store_root, config.server.models_root
    (root / "scratch").mkdir(parents=True, exist_ok=True)
    store = NarrationStore(root, StoreStandIn(), alignment_method_id=METHOD_ID)
    if store.get_engine_profile(ENGINE_ID) is None:
        store.put_engine_profile(engine_profile(models))
        store.set_current_engine_profile("base", ENGINE_ID)
    backend = NarrationBackend(
        config,
        store,
        TestPlatform(),
        launcher=FakeLauncher(store=store),
        pins=lambda: pins_for(models),
        measurements=FakeMeasurements(store),
        poll_s=0.02,
    )
    return backend, store


def test_a_clip_on_the_list_is_accepted_by_measure_voice_after_a_restart_dc_17(
    admin: AdminRun, config_path: Path, clip: tuple[Path, str]
) -> None:
    path, sha = clip
    voice = {"voice": {"path": str(path), "sha256": sha, "transcript": VOICE_TRANSCRIPT}}
    running, store = _front_end(config_path)
    try:
        with pytest.raises(NarrationError) as caught:
            running.measure_voice_sync(voice)
        assert caught.value.code == codes.VOICE_NOT_SYNTHETIC

        assert allow(admin, config_path, path, "--yes")[0] == EXIT_OK

        with pytest.raises(NarrationError) as still:  # a server started before the edit keeps its list
            running.measure_voice_sync(voice)
        assert still.value.code == codes.VOICE_NOT_SYNTHETIC
    finally:
        store.close()

    restarted, store = _front_end(config_path)
    try:
        queued = restarted.measure_voice_sync(voice)
        assert queued["status"] == "queued"
        job = store.get_job(queued["job_id"])
        assert job is not None and job.kind == "measure"
        # The daemon checks again, with the configuration it read when it started (the job engine and the
        # measure handler both call require_synthetic): a restarted daemon accepts the clip too.
        require_synthetic(store, load_config(config_path).voices.allow_sha256, sha)
    finally:
        store.close()


# ==================================================================== list, and never an MCP tool
def test_voices_list_prints_the_allowlist_dc_17(admin: AdminRun, config_path: Path, clip: tuple[Path, str]) -> None:
    path, sha = clip
    empty = admin("--config", str(config_path), "voices", "list")
    assert empty.code == EXIT_OK and "is empty" in empty.out
    assert allow(admin, config_path, path, "--yes")[0] == EXIT_OK
    listed = admin("--config", str(config_path), "voices", "list")
    assert listed.code == EXIT_OK
    assert f"  {sha}  {path.resolve()}" in listed.out.splitlines()
    as_json = json.loads(admin("--config", str(config_path), "voices", "list", "--json").out)
    assert as_json == {"config": str(config_path), "allow_sha256": [{"sha256": sha, "note": str(path.resolve())}]}


def test_voices_list_shows_hashes_without_a_comment_dc_17(admin: AdminRun, tmp_path: Path) -> None:
    config = write_config(tmp_path / "service", f'[voices]\nallow_sha256 = ["{OTHER}", "{ANOTHER}"]  # two\n')
    listed = admin("--config", str(config), "voices", "list", "--json")
    assert json.loads(listed.out)["allow_sha256"] == [
        {"sha256": OTHER, "note": None},
        {"sha256": ANOTHER, "note": None},
    ]


def test_voices_allow_is_never_an_mcp_tool_dc_17() -> None:
    for name in (*TOOL_NAMES, *(tool.name for tool in TOOLS)):
        assert "allow" not in name and "voices" not in name, name
