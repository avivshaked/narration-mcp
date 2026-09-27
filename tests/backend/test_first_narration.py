"""Milestone M1, "first narration" (plan.md WP36): a caller measures an allowlisted synthetic voice, submits
paragraphs of cues with hints and takes, polls the job, and gets QA'd, cue-aligned takes back: each with its
delivery WAV's path and sha256, its cue times, its verdict, and the job's report.

The whole path runs here: the MCP front-end (WP17) over JSON-RPC, this backend, the store, and the job engine
(WP31) with the fake worker. The daemon is stood in for by stepping the engine's runner, as the daemon's loop
does. Every text is invented for these tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread
import pytest
from jsonschema import Draft202012Validator

from narration.contracts.schemas import TOOLS_BY_NAME
from narration.mcp import build_front_end
from tests.jobs.support import KETTLE, LAMPS, ORCHARD
from tests.mcp.wire import ERAS, Era, Wire, open_wire

from .conftest import Service
from .support import sha256_of


def schema_failures(tool: str, structured: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(TOOLS_BY_NAME[tool].output_schema)
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in validator.iter_errors(structured)]


async def call_ok(wire: Wire, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """A tool call that must succeed; its structured result, checked against the tool's output schema."""
    reply = await wire.call(tool, arguments)
    assert "result" in reply, reply
    result = reply["result"]
    assert not result.get("isError"), result
    structured = result["structuredContent"]
    assert schema_failures(tool, structured) == []
    assert json.loads(result["content"][0]["text"]) == structured
    return structured


def cue_request(service: Service) -> dict[str, Any]:
    """Two paragraphs of cues, a hint, two takes each."""
    return {
        "voice": service.voice(),
        "hints": [{"term": "lamplighter", "respell": "lamp lighter"}],
        "segments": [
            {"segment_id": "p01", "cues": [{"text": LAMPS}, {"text": KETTLE}]},
            {"segment_id": "p02", "cues": [{"text": ORCHARD}]},
        ],
        "options": {"takes": 2},
        "label": "first narration",
    }


def check_takes(results: dict[str, Any], *, cues: dict[str, int]) -> None:
    """Every take: a delivery file whose sha256 is the one reported, placed cue times, a QA verdict."""
    assert [s["segment_id"] for s in results["segments"]] == list(cues)
    for segment in results["segments"]:
        assert segment["status"] in ("passed", "warned"), segment
        assert len(segment["takes"]) == 2
        assert segment["suggested_take_id"] in {t["take_id"] for t in segment["takes"]}
        for take in segment["takes"]:
            delivery = Path(take["delivery"]["path"])
            assert delivery.is_file()
            assert sha256_of(delivery) == take["delivery"]["sha256"]
            assert take["delivery"]["sample_rate"] == 48_000
            assert len(take["cues"]) == cues[segment["segment_id"]]
            for cue in take["cues"]:
                assert cue["start_s"] is not None and cue["end_s"] is not None
                assert 0.0 <= cue["start_s"] < cue["end_s"] <= take["delivery"]["duration_s"] + 1e-6
            assert take["qa"]["verdict"] in ("pass", "warn", "fail")
            assert take["alignment"]["method"]
            for flag in take["qa"]["flags"] + take["flags"]:
                assert flag.get("segment_id") in (None, segment["segment_id"])


@pytest.mark.parametrize("era", ERAS)
def test_first_narration_over_the_wire_m1(service: Service, era: Era) -> None:
    front = build_front_end(service.backend, log_path=service.world.root / "narration-mcp.log")

    async def body(wire: Wire) -> None:
        # 1. The voice's measurement: this test voice was measured before, so it comes back at once.
        measured = await call_ok(wire, "measure_voice", {"voice": service.voice()})
        assert measured["status"] == "completed"
        assert measured["measurement"]["max_segment_chars"] == 400
        voice_hash = measured["voice_hash"]
        done = await call_ok(wire, "get_job", {"job_id": measured["job_id"]})
        assert done["status"] == "completed"

        # 2. The narration job: queued, and a daemon asked for once the job was committed.
        submitted = await call_ok(wire, "submit_job", cue_request(service))
        job_id = submitted["job_id"]
        assert submitted["status"] == "queued"
        assert submitted["voice_hash"] == voice_hash
        assert submitted["plan"]["renders_needed"] == 4  # two segments, two takes each
        assert service.launcher.ensured == [1]
        waiting = await call_ok(wire, "get_job", {"job_id": job_id})
        assert waiting["status"] == "queued"
        assert waiting["queue_position"] == 0
        assert waiting["poll_after_s"] >= 0

        # The daemon's loop runs the job (fake workers).
        await anyio.to_thread.run_sync(lambda: service.world.run())

        finished = await call_ok(wire, "get_job", {"job_id": job_id, "wait_s": 5, "include_segments": True})
        assert finished["status"] == "completed", finished
        assert finished["poll_after_s"] == 0
        assert {s["segment_id"] for s in finished["segments"]} == {"p01", "p02"}

        # 3. The results: QA'd, cue-aligned takes.
        results = await call_ok(wire, "get_results", {"job_id": job_id})
        assert results["job"]["status"] == "completed"
        assert results["job"]["label"] == "first narration"
        assert results["voice"] == {"voice_hash": voice_hash, "clip_sha256": service.world.clip_sha256}
        check_takes(results, cues={"p01": 2, "p02": 1})
        report = Path(results["report_md"])
        assert report.is_file() and report.read_text(encoding="utf-8").strip()
        assert (report.parent / "report.json").is_file()

        # The report resource says the same.
        read = await wire.result("resources/read", {"uri": f"narration://jobs/{job_id}/report"})
        assert read["contents"][0]["text"].strip()

    async def main() -> None:
        async with open_wire(front.server, era) as wire:
            await body(wire)

    anyio.run(main)


def test_same_request_again_is_answered_from_the_cache_s10_2(service: Service) -> None:
    backend = service.backend
    first = backend.submit_job_sync(cue_request(service))
    service.world.run()
    assert service.world.job(first["job_id"]).status == "completed"

    again = backend.submit_job_sync(cue_request(service))
    assert again["job_id"] != first["job_id"], "a finished job is not active: the same request is a new job"
    plan = again["plan"]
    assert (plan["renders_needed"], plan["deliveries_needed"], plan["analyses_needed"]) == (0, 0, 0)
    assert plan["segments_cached"] == plan["segments_total"] == 2
    rendered = len(service.world.pool.texts())
    service.world.run()
    assert len(service.world.pool.texts()) == rendered, "nothing is rendered again"
    results = backend.get_results_sync({"job_id": again["job_id"]})
    assert all(not t["fresh"] for s in results["segments"] for t in s["takes"])
    check_takes(results, cues={"p01": 2, "p02": 1})
