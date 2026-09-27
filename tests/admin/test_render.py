"""``narration-admin render`` (design sections 7.1, 7.3 to 7.5): one text rendered in a voice through the
service's own backend, from the terminal.

The store is the job engine's test world (``tests.jobs``): a pinned engine, a measured synthetic voice, fake
workers. The daemon is stood in for by a launcher that steps the engine's runner when ``submit_job`` asks
for a daemon, as the daemon's loop would. Every text is invented for these tests.
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from narration.admin import render as render_module
from narration.admin.__main__ import main
from narration.admin.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE
from narration.contracts.interfaces import Store
from narration.contracts.models import DaemonStatus
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.platform.testing import StandInPlatform
from tests.admin.conftest import Ran
from tests.backend.support import sha256_of, write_wav
from tests.jobs.conftest import World, anchor, make_world
from tests.jobs.support import LAMPS, VOICE_TRANSCRIPT

__all__ = ["anchor"]


class StepLauncher:
    """Asked for a daemon, it runs the engine's runner until the queue is done (or, with ``run`` False,
    does nothing, as a daemon that has not started yet)."""

    def __init__(self, world: World, *, run: bool = True) -> None:
        self.world = world
        self.run = run
        self.ensured = 0

    def running(self, store: Store) -> DaemonStatus | None:
        return None

    def ensure(self, store: Store) -> None:
        self.ensured += 1
        if self.run:
            self.world.run()


class Rig:
    def __init__(self, world: World, config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.world = world
        self.config = config
        self.platform = StandInPlatform()
        self.launcher = StepLauncher(world)
        monkeypatch.setattr(render_module, "launcher_for", lambda admin: self.launcher)

    def __call__(self, *argv: str) -> Ran:
        out, err = io.StringIO(), io.StringIO()
        code = main(
            ["--config", str(self.config), "render", *argv],
            out=out,
            err=err,
            platform=lambda: self.platform,
            environ={},
        )
        return Ran(code, out.getvalue(), err.getvalue())

    def voice(self) -> list[str]:
        return ["--voice", str(self.world.clip), "--transcript", VOICE_TRANSCRIPT]


@pytest.fixture
def rig(tmp_path: Path, anchor: tuple[float, ...], monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    world = make_world(tmp_path, anchor)
    config = tmp_path / "narration.toml"
    config.write_text(
        f"[server]\nstore_root = '{world.config.server.store_root.as_posix()}'\n"
        f"models_root = '{world.config.server.models_root.as_posix()}'\n",
        encoding="utf-8",
    )
    try:
        yield Rig(world, config, monkeypatch)
    finally:
        world.pool.close()
        world.store.close()


def test_render_prints_each_take_and_copies_the_suggested_one_s7_5(rig: Rig, tmp_path: Path) -> None:
    out = tmp_path / "delivered" / "lamps.wav"
    ran = rig(*rig.voice(), "--text", LAMPS, "--out", str(out))
    assert ran.code == EXIT_OK, ran.err
    assert "job job_" in ran.out and ": completed" in ran.out
    assert "render: passed" in ran.out or "render: warned" in ran.out
    assert "cue 0:" in ran.out and "not placed" not in ran.out
    assert f"wrote {out}" in ran.out
    shown = [line.split("sha256 ")[1] for line in ran.out.splitlines() if "sha256 " in line]
    assert sha256_of(out) in shown, "the copy is the suggested take's delivery file"
    assert rig.launcher.ensured == 1


def test_render_as_json_is_get_results_s7_5(rig: Rig) -> None:
    ran = rig(*rig.voice(), "--text", LAMPS, "--takes", "2", "--json")
    assert ran.code == EXIT_OK, ran.err
    results: dict[str, Any] = json.loads(ran.out)
    validator = Draft202012Validator(TOOLS_BY_NAME["get_results"].output_schema)
    assert [e.message for e in validator.iter_errors(results)] == []
    (segment,) = results["segments"]
    assert segment["segment_id"] == "render" and len(segment["takes"]) == 2


def test_the_same_render_again_renders_nothing_twice_s10_2(rig: Rig) -> None:
    assert rig(*rig.voice(), "--text", LAMPS).code == EXIT_OK
    rendered = len(rig.world.pool.texts())
    assert rig(*rig.voice(), "--text", LAMPS).code == EXIT_OK
    assert len(rig.world.pool.texts()) == rendered


def test_texts_can_come_from_files_s7_3(rig: Rig, tmp_path: Path) -> None:
    text, transcript = tmp_path / "text.txt", tmp_path / "transcript.txt"
    text.write_text(LAMPS + "\n", encoding="utf-8")
    transcript.write_text(VOICE_TRANSCRIPT, encoding="utf-8")
    ran = rig("--voice", str(rig.world.clip), "--transcript-file", str(transcript), "--text-file", str(text))
    assert ran.code == EXIT_OK, ran.err


def test_a_dry_run_plans_and_queues_nothing_s7_3(rig: Rig) -> None:
    ran = rig(*rig.voice(), "--text", LAMPS, "--dry-run")
    assert ran.code == EXIT_OK, ran.err
    assert "would render 1" in ran.out
    assert rig.world.store.queued_jobs() == ()
    assert rig.launcher.ensured == 0


def test_render_goes_through_the_tools_own_checks_s3_2(rig: Rig, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere" / "stranger.wav"
    write_wav(other, freq=330.0)
    ran = rig("--voice", str(other), "--transcript", VOICE_TRANSCRIPT, "--text", LAMPS)
    assert ran.code == EXIT_FAILED
    assert "VOICE_NOT_SYNTHETIC" in ran.err
    wrong = rig(*rig.voice(), "--sha256", "0" * 64, "--text", LAMPS)
    assert wrong.code == EXIT_FAILED and "VOICE_NOT_SYNTHETIC" in wrong.err
    assert rig.world.store.queued_jobs() == ()


def test_an_existing_out_file_is_kept_unless_forced(rig: Rig, tmp_path: Path) -> None:
    out = tmp_path / "kept.wav"
    out.write_bytes(b"the operator's own file")
    ran = rig(*rig.voice(), "--text", LAMPS, "--out", str(out))
    assert ran.code == EXIT_USAGE and "--force" in ran.err
    assert out.read_bytes() == b"the operator's own file"
    assert rig.launcher.ensured == 0, "nothing was submitted"
    forced = rig(*rig.voice(), "--text", LAMPS, "--out", str(out), "--force")
    assert forced.code == EXIT_OK and out.read_bytes() != b"the operator's own file"


def test_without_waiting_it_prints_the_job_s7_3(rig: Rig) -> None:
    rig.launcher.run = False
    ran = rig(*rig.voice(), "--text", LAMPS, "--wait", "0")
    assert ran.code == EXIT_OK
    job_id = ran.out.strip()
    job = rig.world.store.get_job(job_id)
    assert job is not None and job.status == "queued"


def test_a_wait_that_ends_first_says_how_to_wait_again_s7_3(rig: Rig) -> None:
    rig.launcher.run = False
    ran = rig(*rig.voice(), "--text", LAMPS, "--wait", "0.2")
    assert ran.code == EXIT_FAILED
    assert "still queued" in ran.err and "run the same command again" in ran.err
    again = rig(*rig.voice(), "--text", LAMPS, "--wait", "0")
    (job,) = rig.world.store.queued_jobs()
    assert again.out.strip() == job.job_id, "the identical request returns the same job"


def test_render_is_registered_in_the_dispatcher_s7_1(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    ran = rig("--help")  # argparse prints help to the process's stdout
    assert ran.code == EXIT_OK
    shown = capsys.readouterr().out
    assert "--voice" in shown and "not available" not in shown


def test_the_request_is_the_tools_request(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    sha = write_wav(clip)
    parser_args = argparse.Namespace(
        voice=str(clip),
        sha256=None,
        transcript="A quiet morning.",
        transcript_file=None,
        text="The ferry leaves.",
        text_file=None,
        takes=2,
        dry_run=False,
        segment_id="p01",
    )
    request = render_module.build_request(parser_args)
    assert request["voice"] == {"path": str(clip), "sha256": sha, "transcript": "A quiet morning."}
    assert request["segments"] == [{"segment_id": "p01", "text": "The ferry leaves."}]
    assert request["options"] == {"takes": 2}
    validator = Draft202012Validator(TOOLS_BY_NAME["submit_job"].input_schema)
    assert [e.message for e in validator.iter_errors(request)] == []
