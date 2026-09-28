"""Flow F over the wire (design sections 3.1, 3.6, 7.6, 7.7, 8 Flow F, 17.4): ``design_voice`` → candidates with
clips, transcripts and profiles → the agent shortlists from the profiles → ``measure_voice`` on the chosen clip,
with no allowlist edit; and ``profile_voice`` on any WAV.

The whole path runs here: the MCP front-end over JSON-RPC, the backend, the store, and the job engine's runner with
the ``design``, ``profile`` and ``measure`` handlers on the fake workers. The daemon is stood in for by stepping
the runner. Every description is invented for these tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread
import pytest

from narration.backend.service import RUNNABLE_KINDS
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.jobs.voice import require_synthetic
from narration.mcp import build_front_end
from tests.backend.support import FakeLauncher, make_backend, sha256_of, write_wav
from tests.backend.test_first_narration import call_ok
from tests.mcp.wire import ERAS, Era, Wire, open_wire

from .support import NEGATED, WARM, DesignWorld, make_world


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[DesignWorld, Any, FakeLauncher]]:
    world = make_world(tmp_path)
    backend, _, launcher = make_backend(world)
    try:
        yield world, backend, launcher
    finally:
        world.close()


def test_this_build_runs_design_and_profile_jobs_s7_1() -> None:
    assert {"design", "profile"} <= RUNNABLE_KINDS
    assert "pronunciation" in RUNNABLE_KINDS  # WP35's (narration.audition)


@pytest.mark.parametrize("era", ERAS)
def test_flow_f_design_then_measure_over_the_wire_s8(served: tuple[DesignWorld, Any, FakeLauncher], era: Era) -> None:
    world, backend, launcher = served
    front = build_front_end(backend, log_path=world.root / "narration-mcp.log")

    async def body(wire: Wire) -> None:
        # 1. design_voice: queued under a new design id; the lint warns and never refuses (section 3.5).
        submitted = await call_ok(
            wire, "design_voice", {"name": "harbour narrator", "description": NEGATED, "takes": 2}
        )
        assert submitted["status"] == "queued"
        assert [f["phrase"] for f in submitted["lint"]["findings"]] == ["not theatrical", "never rushed", "no rasp"]
        assert launcher.ensured == [1]
        design_id, job_id = submitted["design_id"], submitted["job_id"]

        await anyio.to_thread.run_sync(lambda: world.run())

        finished = await call_ok(wire, "get_job", {"job_id": job_id, "wait_s": 5})
        assert (finished["status"], finished["outcome"]) == ("completed", "all_passed")

        # 2. The results: each candidate's clip, exact transcript, seed, lint, profile and flags.
        results = await call_ok(wire, "get_results", {"job_id": job_id})
        design = results["design"]
        assert design["design_id"] == design_id
        assert [c["index"] for c in design["candidates"]] == [0, 1]
        for candidate in design["candidates"]:
            assert Path(candidate["clip"]["path"]).is_file()
            assert sha256_of(Path(candidate["clip"]["path"])) == candidate["clip"]["sha256"]
            assert candidate["transcript_check"]["ok"] is True
            assert candidate["description"] == NEGATED
            assert len(candidate["lint"]["findings"]) == 3
            assert candidate["profile"]["measurements"]["speaking_rate_wpm"] > 0
            assert Path(candidate["profile"]["pictures"]["pitch"]).is_file()
            assert candidate["flags"] == []

        # The design resource lists the same candidates (section 7.7).
        read = await wire.result("resources/read", {"uri": f"narration://designs/{design_id}"})
        body_ = json.loads(read["contents"][0]["text"])
        assert [c["clip"]["sha256"] for c in body_["candidates"]] == [c["clip"]["sha256"] for c in design["candidates"]]

        # 3. The person chooses; the caller copies the clip into its own folder and measures it. The service
        #    designed it, so the synthetic-voices rule accepts it with no allowlist edit (section 17.4).
        chosen = design["candidates"][1]
        kept = world.root / "caller" / "narrator.wav"
        kept.parent.mkdir(parents=True)
        kept.write_bytes(Path(chosen["clip"]["path"]).read_bytes())
        voice = {"path": str(kept), "sha256": chosen["clip"]["sha256"], "transcript": chosen["transcript"]}
        measuring = await call_ok(wire, "measure_voice", {"voice": voice})
        assert measuring["status"] == "queued"
        await anyio.to_thread.run_sync(lambda: world.run(max_steps=4000))
        measured = await call_ok(wire, "get_job", {"job_id": measuring["job_id"]})
        assert measured["status"] == "completed", measured

    async def main() -> None:
        async with open_wire(front.server, era) as wire:
            await body(wire)

    anyio.run(main)


def test_profile_voice_over_the_wire_s3_6(served: tuple[DesignWorld, Any, FakeLauncher]) -> None:
    world, backend, _ = served
    front = build_front_end(backend, log_path=world.root / "narration-mcp.log")
    audio = world.root / "elsewhere" / "take.wav"
    sha = write_wav(audio, seconds=3.0)

    async def body(wire: Wire) -> None:
        submitted = await call_ok(wire, "profile_voice", {"audio": {"path": str(audio), "sha256": sha}})
        await anyio.to_thread.run_sync(lambda: world.run())
        results = await call_ok(wire, "get_results", {"job_id": submitted["job_id"]})
        profile = results["profile"]
        assert profile["audio_sha256"] == sha
        assert profile["measurements"]["duration_s"] == pytest.approx(3.0)
        assert profile["measurements"]["speaking_rate_wpm"] is None
        assert Path(profile["pictures"]["spectrogram"]).is_file()

    async def main() -> None:
        async with open_wire(front.server, ERAS[0]) as wire:
            await body(wire)

    anyio.run(main)


def test_a_clip_the_service_did_not_design_is_still_refused_s17_4(
    served: tuple[DesignWorld, Any, FakeLauncher],
) -> None:
    """Designing voices and profiling a stranger's clip never make that clip one the service may clone: only the
    design job's own candidates reach the provenance list."""
    world, backend, _ = served
    stranger = world.root / "elsewhere" / "stranger.wav"
    sha = write_wav(stranger, freq=330.0)
    job = world.design(takes=1, description=WARM)
    world.run()
    assert world.job(job.job_id).status == "completed"
    profiled = backend.profile_voice_sync({"audio": {"path": str(stranger), "sha256": sha}})
    world.run()
    assert world.job(profiled["job_id"]).status == "completed"
    assert not world.store.is_provenance(sha)
    lines = (world.store.root / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    (candidate,) = world.store.get_design(str(job.request["design_id"]))
    assert [json.loads(line)["clip_sha256"] for line in lines] == [candidate.clip.sha256]
    voice = {"path": str(stranger), "sha256": sha, "transcript": "Hello."}
    reply = anyio.run(lambda: _call(backend, {"voice": voice}))
    assert reply["code"] == codes.VOICE_NOT_SYNTHETIC and reply["field"] == "voice.sha256"
    with pytest.raises(NarrationError) as caught:
        require_synthetic(world.store, world.config.voices.allow_sha256, sha)  # the daemon's own check, too
    assert caught.value.code == codes.VOICE_NOT_SYNTHETIC


async def _call(backend: Any, args: dict[str, Any]) -> dict[str, Any]:
    try:
        await backend.measure_voice(args)
    except NarrationError as exc:
        return {"code": exc.code, "field": exc.field}
    return {"code": None, "field": None}
