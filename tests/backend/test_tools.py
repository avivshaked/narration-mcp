"""The other tools and the results (design sections 4.1, 7.4 to 7.7, 11, 12, 15; plan.md DC-2), each pinned
by the rule it checks. The backend is called directly, against the job engine's fake workers.

Every text is invented for these tests.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from typing import Any

import anyio
import anyio.to_thread
import pytest
from jsonschema import Draft202012Validator

from narration.backend.assemble import restamp_exact, restamp_flag
from narration.backend.measures import StoreMeasurements
from narration.backend.service import NarrationBackend
from narration.config import VoicesConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import ExactResult, Flag, SegmentIn
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.contracts.serial import from_json
from narration.jobs.plan import VoiceSpec
from narration.jobs.voice import clip_path
from narration.text import TextPipeline
from tests.jobs.support import ENGINE_HASH, ENGINE_ID, KETTLE, LAMPS, METHOD_ID, ORCHARD, VOICE_TRANSCRIPT, voice_hash

from .conftest import Service
from .support import daemon_status, write_wav


def valid(tool: str, structured: dict[str, Any]) -> dict[str, Any]:
    validator = Draft202012Validator(TOOLS_BY_NAME[tool].output_schema)
    failures = [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in validator.iter_errors(structured)]
    assert failures == []
    return structured


def run(fn: Callable[[], Any]) -> Any:
    async def main() -> Any:
        return await fn()

    return anyio.run(main)


def get_job(backend: NarrationBackend, args: dict[str, Any], progress: Any = None) -> dict[str, Any]:
    return valid("get_job", run(lambda: backend.get_job(args, progress)))


# ======================================================================== get_job (section 7.4, DC-2)


def test_get_job_reports_a_queued_job_with_its_place_and_poll_after_dc2(service: Service) -> None:
    first = service.backend.submit_job_sync(service.request(LAMPS))
    second = service.backend.submit_job_sync(service.request(KETTLE))
    now = get_job(service.backend, {"job_id": second["job_id"]})
    assert (now["status"], now["queue_position"]) == ("queued", 1)
    assert now["eta_s"] is not None and now["eta_s"] > 0
    assert now["poll_after_s"] > 0
    assert get_job(service.backend, {"job_id": first["job_id"]})["queue_position"] == 0


def test_get_job_waits_at_most_wait_s_s7_4(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    started = time.monotonic()
    now = get_job(service.backend, {"job_id": submitted["job_id"], "wait_s": 0.3})
    assert now["status"] == "queued"
    assert 0.25 <= time.monotonic() - started < 5


def test_get_job_long_poll_ends_when_the_status_changes_with_progress_s7_4(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS, ORCHARD))
    seen: list[tuple[float, float | None, str | None]] = []

    async def progress(done: float, total: float | None, message: str | None) -> None:
        seen.append((done, total, message))

    async def main() -> dict[str, Any]:
        out: dict[str, Any] = {}
        async with anyio.create_task_group() as tg:
            tg.start_soon(anyio.to_thread.run_sync, lambda: service.world.run())
            out = await service.backend.get_job({"job_id": submitted["job_id"], "wait_s": 60}, progress)
        return out

    started = time.monotonic()
    out = valid("get_job", anyio.run(main))
    assert out["status"] != "queued"
    assert time.monotonic() - started < 60
    assert seen, "a progress notification is sent while the job waits"
    done = [d for d, _, _ in seen]
    assert done == sorted(done)


def test_cancelling_the_wait_ends_the_wait_never_the_job_s7_4(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))

    async def main() -> None:
        with anyio.move_on_after(0.2):
            await service.backend.get_job({"job_id": submitted["job_id"], "wait_s": 30}, None)

    anyio.run(main)
    assert service.world.job(submitted["job_id"]).status == "queued"


def test_an_unknown_job_is_not_found_s14(service: Service) -> None:
    with pytest.raises(NarrationError) as caught:
        get_job(service.backend, {"job_id": "job_01JBXQ7Z3M8V4T2R9K6N5P0W1D"})
    assert (caught.value.code, caught.value.field) == (codes.NOT_FOUND, "job_id")


# ======================================================================== get_results (sections 7.5, 11, 12)


def test_results_of_an_unfinished_job_list_its_segments_as_planned_s7_5(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    results = valid("get_results", service.backend.get_results_sync({"job_id": submitted["job_id"]}))
    assert results["job"]["status"] == "queued"
    assert [(s["segment_id"], s["status"], s["takes"]) for s in results["segments"]] == [("p01", "planned", [])]
    assert "report_md" not in results


def test_results_leave_words_and_transcripts_out_when_asked_s7_5(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    service.world.run()
    full = service.backend.get_results_sync({"job_id": submitted["job_id"], "include_transcripts": True})
    take = full["segments"][0]["takes"][0]
    assert take["cues"][0]["words"], "words are in by default"
    assert take["qa"]["transcript"]
    lean = valid(
        "get_results", service.backend.get_results_sync({"job_id": submitted["job_id"], "include_words": False})
    )
    lean_take = lean["segments"][0]["takes"][0]
    assert lean_take["cues"][0].get("words", []) == []
    assert lean_take["qa"].get("transcript") is None
    assert lean_take["cues"][0]["start_s"] == take["cues"][0]["start_s"]


def test_fit_is_reported_only_for_a_segment_with_scene_seconds_s12(service: Service) -> None:
    body = service.request(LAMPS, KETTLE)
    body["segments"][0]["scene_seconds"] = 30.0
    submitted = service.backend.submit_job_sync(body)
    service.world.run()
    results = valid("get_results", service.backend.get_results_sync({"job_id": submitted["job_id"]}))
    with_scene, without = results["segments"]
    assert with_scene["takes"][0]["fit"]["scene_seconds"] == 30.0
    assert "fit" not in without["takes"][0], "no fit of any kind without scene_seconds (R4)"


def test_a_cached_analysis_is_restamped_for_this_request_s7_5(service: Service) -> None:
    """Section 10.2: the analysis key has no segment id and no offsets as sent, so a cached analysis is
    restamped with this request's segment id and exact-span offsets."""
    service.world.faults({"kind": "wrong_word", "word": 2, "replacement": "barber", "when": {"text_contains": "canal"}})

    def request(segment_id: str, text: str, start: int) -> dict[str, Any]:
        cue = {"text": text, "exact": [{"start": start, "end": start + 5}]}
        body = service.request(LAMPS, ids=[segment_id], max_retakes=0)
        body["segments"] = [{"segment_id": segment_id, "cues": [cue]}]
        return body

    first = service.backend.submit_job_sync(request("p01", LAMPS, 16))  # "walks"
    service.world.run()
    spaced = LAMPS.replace("The ", "The  ", 1)  # the same spoken text, other offsets as sent
    second = service.backend.submit_job_sync(request("opening", spaced, 17))
    service.world.run()

    for job_id, segment_id, start in ((first["job_id"], "p01", 16), (second["job_id"], "opening", 17)):
        results = valid("get_results", service.backend.get_results_sync({"job_id": job_id}))
        (segment,) = results["segments"]
        take = segment["takes"][0]
        mismatch = [f for f in take["qa"]["flags"] if f["code"] == codes.EXACT_SPAN_MISMATCH]
        assert mismatch, take["qa"]["flags"]
        assert {f["segment_id"] for f in take["qa"]["flags"]} == {segment_id}
        assert (mismatch[0]["details"]["start"], mismatch[0]["details"]["end"]) == (start, start + 5)
        assert (take["qa"]["exact"][0]["start"], take["qa"]["exact"][0]["end"]) == (start, start + 5)
    first_take = service.backend.get_results_sync({"job_id": first["job_id"]})["segments"][0]["takes"][0]
    second_take = service.backend.get_results_sync({"job_id": second["job_id"]})["segments"][0]["takes"][0]
    assert first_take["analysis_id"] == second_take["analysis_id"], "the analysis was reused, not made again"
    assert second_take["fresh"] is False


def test_restamp_flag_sets_segment_and_offsets_from_the_request_s7_5() -> None:
    segment = from_json(SegmentIn, {"segment_id": "p07", "cues": [{"text": KETTLE, "exact": [{"start": 2, "end": 8}]}]})
    (text,) = TextPipeline().plan_request([segment], [], strict_text=False)
    spans = {(c.index, e.words): e for c in text.cues for e in c.exact}
    words = next(iter(spans))[1]
    flag = Flag(
        code=codes.EXACT_SPAN_MISMATCH,
        severity="fail",
        message="expected copper",
        cue=0,
        details={"cue": 0, "words": list(words), "start": 90, "end": 99},
    )
    stamped = restamp_flag(flag, "p07", spans)
    assert stamped.segment_id == "p07"
    assert stamped.details is not None and (stamped.details["start"], stamped.details["end"]) == (2, 8)
    exact = ExactResult(cue=0, start=90, end=99, words=words, expected="copper", heard="cop", match="different")
    assert (restamp_exact(exact, spans).start, restamp_exact(exact, spans).end) == (2, 8)
    other = Flag(code=codes.WER_HIGH, severity="warn", message="many words differ")
    assert restamp_flag(other, "p07", spans).segment_id == "p07"


def test_the_report_is_written_beside_the_job_s15(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    service.world.run()
    results = service.backend.get_results_sync({"job_id": submitted["job_id"]})
    folder = service.world.store.job_dir(submitted["job_id"])
    assert results["report_md"] == str(folder / "report.md")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report
    again = service.backend.get_results_sync({"job_id": submitted["job_id"]})
    assert again["report_md"] == results["report_md"]


# ======================================================================== cancel_job (sections 7.6, 8)


def test_cancelling_a_queued_job_cancels_it_at_once_s7_6(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    out = valid("cancel_job", service.backend.cancel_job_sync({"job_id": submitted["job_id"], "reason": "changed"}))
    assert out == {"status": "cancelled", "completed": True}
    assert service.world.job(submitted["job_id"]).status == "cancelled"


def test_cancelling_a_running_job_asks_the_daemon_to_stop_it_s8(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    assert service.world.store.claim_job(submitted["job_id"], "a-daemon") is not None
    out = valid("cancel_job", service.backend.cancel_job_sync({"job_id": submitted["job_id"]}))
    assert out == {"status": "cancelling", "completed": False}
    assert service.world.job(submitted["job_id"]).status == "cancelling"


def test_a_finished_job_is_not_cancellable_s14(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    service.world.run()
    with pytest.raises(NarrationError) as caught:
        service.backend.cancel_job_sync({"job_id": submitted["job_id"]})
    assert caught.value.code == codes.JOB_NOT_CANCELLABLE


# ======================================================================== check_text (section 7.6; R6 to R8)


def test_check_text_echoes_each_segment_and_checks_its_length_for_the_voice_r7(service: Service) -> None:
    service.world.measure(max_segment_chars=40)
    args = {"voice": service.voice(), "segments": [{"segment_id": "p01", "text": LAMPS}]}
    out = valid("check_text", service.backend.check_text_sync(args))
    (segment,) = out["segments"]
    assert segment["segment_id"] == "p01"
    assert segment["max_segment_chars"] == 40
    assert segment["over_by_chars"] == segment["spoken_chars"] - 40 > 0
    assert segment["est_duration_s"] > 0
    assert codes.SEGMENT_TOO_LONG in {w["code"] for w in segment["warnings"]}
    assert out["measurement"]["max_segment_chars"] == 40
    assert not clip_path(service.world.store, service.world.clip_sha256).exists(), "check_text reads no file"


def test_check_text_without_a_voice_has_no_length_check_r7(service: Service) -> None:
    out = valid("check_text", service.backend.check_text_sync({"segments": [{"segment_id": "p01", "text": LAMPS}]}))
    assert (out["voice_hash"], out["measurement"]) == (None, None)
    assert out["segments"][0]["max_segment_chars"] is None


# ======================================================================== measure_voice (sections 3.2, 7.6)


def test_a_measured_voice_is_answered_at_once_s3_2(service: Service) -> None:
    out = valid("measure_voice", service.backend.measure_voice_sync({"voice": service.voice()}))
    assert out["status"] == "completed"
    assert out["measurement"]["engine_profile"] == {"id": ENGINE_ID, "hash": ENGINE_HASH}
    results = valid("get_results", service.backend.get_results_sync({"job_id": out["job_id"]}))
    assert results["measurement_result"]["measurement"]["voice_hash"] == out["voice_hash"]
    assert results["measurement_result"]["path"].endswith("measurement.json")
    handle = service.world.job(out["job_id"]).result
    assert handle is not None
    assert set(handle) == {
        "voice_hash",
        "engine_profile",
        "measurement_key",
        "path",
        "max_segment_chars",
        "max_segment_seconds",
        "ladder_stopped_at",
    }, "the handle a measure job leaves (WP33)"
    assert service.launcher.ensured == [], "nothing to run, so no daemon"


def test_a_new_voice_is_queued_for_measuring_s3_2(service: Service) -> None:
    other = service.world.root / "elsewhere" / "new-voice.wav"
    sha = write_wav(other, freq=260.0)
    service.backend.config = dataclasses.replace(service.backend.config, voices=VoicesConfig(allow_sha256=(sha,)))
    voice = service.voice(path=str(other), sha256=sha)
    out = valid("measure_voice", service.backend.measure_voice_sync({"voice": voice}))
    assert out["status"] == "queued"
    assert service.world.job(out["job_id"]).kind == "measure"
    assert service.launcher.ensured == [1]
    assert clip_path(service.world.store, sha).is_file(), "the clip is in the store before a worker sees it"
    again = service.backend.measure_voice_sync({"voice": voice})
    assert again["job_id"] == out["job_id"], "the same voice while measuring is the same job"


def test_measuring_a_clip_not_designed_or_allowed_is_refused_s17_4(service: Service) -> None:
    other = service.world.root / "elsewhere" / "stranger.wav"
    sha = write_wav(other, freq=330.0)
    with pytest.raises(NarrationError) as caught:
        service.backend.measure_voice_sync({"voice": service.voice(path=str(other), sha256=sha)})
    assert (caught.value.code, caught.value.field) == (codes.VOICE_NOT_SYNTHETIC, "voice.sha256")


# ======================================================================== get_server_status (sections 4.1, 7.6)


def test_status_with_no_daemon_says_stopped_and_admits_dc2(service: Service) -> None:
    out = valid("get_server_status", service.backend.get_server_status_sync({}))
    assert out["daemon"]["state"] == "stopped"
    assert out["gpu"]["name"] is None
    assert out["admission"]["accepting"] is True
    assert [p["id"] for p in out["engine_profiles"]] == [ENGINE_ID]
    assert out["alignment"]["method_id"] == METHOD_ID
    assert out["limits"]["max_queued_jobs"] == 20


def test_status_lists_the_queue_and_the_daemons_view_dc2(service: Service) -> None:
    first = service.backend.submit_job_sync(service.request(LAMPS))
    service.launcher.status = daemon_status(state="busy", holder="qwen", job_id=first["job_id"])
    service.backend.submit_job_sync(service.request(KETTLE))
    out = valid("get_server_status", service.backend.get_server_status_sync({}))
    assert out["daemon"]["state"] == "busy"
    assert out["daemon"]["current_job"]["job_id"] == first["job_id"]
    assert (out["gpu"]["name"], out["gpu"]["holder"]) == ("Test GPU", "qwen")
    assert out["queue"]["length"] == 2
    assert [j["queue_position"] for j in out["queue"]["jobs"]] == [0, 1]


def test_release_gpu_with_no_daemon_releases_nothing_s7_6(service: Service) -> None:
    out = valid("release_gpu", service.backend.release_gpu_sync({}))
    assert out == {"released": False, "holder_before": None, "busy_job": None}


@pytest.mark.parametrize("tool", ["design_voice", "profile_voice", "audition_pronunciation"])
def test_tools_whose_handlers_are_not_built_say_so_s14(service: Service, tool: str) -> None:
    with pytest.raises(NarrationError) as caught:
        run(lambda: getattr(service.backend, tool)({}))
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED
    assert not caught.value.retryable


# ======================================================================== resources (section 7.7)


def test_resources_read_the_store_s7_7(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    service.world.run()
    job_id = submitted["job_id"]
    results = service.backend.get_results_sync({"job_id": job_id})
    take_id = results["segments"][0]["takes"][0]["take_id"]
    read = service.backend.read_resource

    status = run(lambda: read("narration://status"))
    assert json.loads(status.text)["daemon"]["state"] == "stopped"
    job = run(lambda: read(f"narration://jobs/{job_id}"))
    assert json.loads(job.text)["status"] == "completed"
    report = run(lambda: read(f"narration://jobs/{job_id}/report"))
    assert report.text.strip()
    take = run(lambda: read(f"narration://takes/{take_id}"))
    assert json.loads(take.text)["take"]["take_id"] == take_id
    measurement = run(lambda: read(f"narration://measurements/{submitted['voice_hash']}"))
    assert json.loads(measurement.text)["voice_hash"] == submitted["voice_hash"]
    with pytest.raises(NarrationError) as caught:
        run(lambda: read("narration://jobs/job_01JBXQ7Z3M8V4T2R9K6N5P0W1D"))
    assert caught.value.code == codes.NOT_FOUND


def test_the_default_measurements_hash_a_voice_as_the_job_engine_does_s10_2(service: Service) -> None:
    measurements = StoreMeasurements(service.world.store, service.world.config)
    voice = VoiceSpec(path=str(service.world.clip), sha256=service.world.clip_sha256, transcript=VOICE_TRANSCRIPT)
    assert measurements.voice_hash(voice) == voice_hash(service.world.clip_sha256)
    profile = service.world.store.current_engine_profile("base")
    assert profile is not None
    assert measurements.require(voice_hash(service.world.clip_sha256), profile).voice_hash == voice_hash(
        service.world.clip_sha256
    )
    with pytest.raises(NarrationError) as caught:
        measurements.require("vh_" + "0" * 16, profile)
    assert (caught.value.code, caught.value.field) == (codes.VOICE_NOT_MEASURED, "voice")
