"""``submit_job``'s rules (design sections 3.2, 3.3, 7.3, 9.1, 16, 17.3, 17.4; plan.md DC-2, DC-6), each pinned
by the rule it checks. The backend is called directly; ``test_first_narration`` runs it over the wire.

Every text is invented for these tests.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

import numpy as np
import pytest
import soundfile

from narration.config import VoicesConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.jobs.voice import clip_path
from narration.platform import get_platform
from tests.jobs.support import ENGINE_HASH, KETTLE, LAMPS, ORCHARD

from .conftest import Service
from .support import make_backend, sha256_of, write_wav


def limits(service: Service, **changes: Any) -> None:
    config = service.backend.config
    service.backend.config = dataclasses.replace(config, limits=dataclasses.replace(config.limits, **changes))


def allow(service: Service, *sha256: str) -> None:
    service.backend.config = dataclasses.replace(service.backend.config, voices=VoicesConfig(allow_sha256=sha256))


def refused(service: Service, body: dict[str, Any]) -> NarrationError:
    with pytest.raises(NarrationError) as caught:
        service.backend.submit_job_sync(body)
    return caught.value


def jobs_in(service: Service) -> int:
    return len(service.world.store.queued_jobs())


# ======================================================================== the job (section 7.3, DC-6)


def test_identical_request_while_active_is_the_same_job_s7_3(service: Service) -> None:
    first = service.backend.submit_job_sync(service.request(LAMPS, KETTLE))
    again = service.backend.submit_job_sync({**service.request(LAMPS, KETTLE), "label": "another name"})
    assert again["job_id"] == first["job_id"], "the label is opaque and never part of the request's identity"
    assert jobs_in(service) == 1


def test_a_different_request_is_a_different_job_s7_3(service: Service) -> None:
    first = service.backend.submit_job_sync(service.request(LAMPS))
    other = service.backend.submit_job_sync(service.request(LAMPS, takes=2))
    assert other["job_id"] != first["job_id"]


def test_request_is_kept_with_the_services_defaults_s7_3(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    job = service.world.job(submitted["job_id"])
    assert job.request["options"] == {"takes": 1, "max_retakes": 2, "strict_text": False, "priority": "batch"}
    assert job.priority == "batch"
    assert len(job.request_sha256) == 64


def test_idempotency_key_on_a_different_request_is_refused_dc6(service: Service) -> None:
    key = {"idempotency_key": "chapter-one-take"}
    first = service.backend.submit_job_sync(service.request(LAMPS, **{"options": key}))
    retry = service.backend.submit_job_sync(service.request(LAMPS, **{"options": key}))
    assert retry["job_id"] == first["job_id"]
    error = refused(service, service.request(KETTLE, **{"options": key}))
    assert (error.code, error.field) == (codes.INVALID_ARGUMENT, "options.idempotency_key")
    assert jobs_in(service) == 1


def test_daemon_is_asked_for_only_after_the_job_is_committed_s4(service: Service) -> None:
    service.backend.submit_job_sync(service.request(LAMPS))
    assert service.launcher.ensured == [1], "ensure ran once, with the job already in the store"


def test_daemon_that_cannot_start_leaves_the_job_queued_and_says_when_to_retry_dc2(service: Service) -> None:
    service.launcher.error = NarrationError(codes.DAEMON_UNAVAILABLE, "no start", retry_after_s=30.0)
    error = refused(service, service.request(LAMPS))
    assert error.code == codes.DAEMON_UNAVAILABLE
    assert error.retryable and error.retry_after_s == 30.0
    assert error.details is not None
    job_id = error.details["job_id"]
    assert service.world.job(job_id).status == "queued"
    service.launcher.error = None
    again = service.backend.submit_job_sync(service.request(LAMPS))
    assert again["job_id"] == job_id, "the retry finds the job it queued"


# ======================================================================== admission (DC-2, section 16)


def test_submissions_over_the_rate_cap_are_rate_limited_with_retry_after_dc2(service: Service) -> None:
    limits(service, max_submits_per_min=1)
    service.backend.submit_job_sync(service.request(LAMPS))
    error = refused(service, service.request(KETTLE))
    assert error.code == codes.RATE_LIMITED
    assert error.retryable and error.retry_after_s is not None and 0 < error.retry_after_s <= 60
    assert jobs_in(service) == 1


def test_a_full_queue_is_refused_with_retry_after_dc2(service: Service) -> None:
    limits(service, max_queued_jobs=1)
    service.backend.submit_job_sync(service.request(LAMPS))
    error = refused(service, service.request(KETTLE))
    assert error.code == codes.QUEUE_FULL
    assert error.retryable and error.retry_after_s is not None and error.retry_after_s > 0
    assert error.details is not None and error.details["length"] == 1


def test_a_full_disk_is_store_full_and_retryable_dc2(service: Service) -> None:
    service.platform.free_bytes = 0
    error = refused(service, service.request(LAMPS))
    assert error.code == codes.STORE_FULL
    assert error.retryable and error.retry_after_s is not None and error.retry_after_s > 0
    assert jobs_in(service) == 0


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"max_segments_per_job": 1}, "segments"),
        ({"max_chars_per_segment": 20}, "segments[0]"),
        ({"max_chars_per_job": 100}, "segments"),
        ({"max_hints_per_job": 0}, "hints"),
    ],
)
def test_limits_come_from_the_config_s16(service: Service, changes: dict[str, int], field: str) -> None:
    limits(service, **changes)
    body = service.request(LAMPS, KETTLE, hints=[{"term": "canal", "respell": "ka nal"}])
    error = refused(service, body)
    assert (error.code, error.field, error.retryable) == (codes.LIMIT_EXCEEDED, field, False)
    assert error.hint
    assert jobs_in(service) == 0


# ======================================================================== the voice clip (sections 17.3, 17.4)


def test_clip_is_copied_into_the_store_before_any_worker_sees_it_s17_3(service: Service) -> None:
    submitted = service.backend.submit_job_sync(service.request(LAMPS))
    copy = clip_path(service.world.store, service.world.clip_sha256)
    assert copy.is_file() and sha256_of(copy) == service.world.clip_sha256
    service.world.clip.unlink()  # the caller's file is gone; the job runs from the copy
    service.world.run()
    assert service.world.job(submitted["job_id"]).status == "completed"


def test_a_clip_neither_designed_nor_allowed_is_refused_s17_4(service: Service) -> None:
    other = service.world.root / "elsewhere" / "stranger.wav"
    sha = write_wav(other, freq=330.0)
    error = refused(service, service.request(LAMPS, voice=service.voice(path=str(other), sha256=sha)))
    assert (error.code, error.field) == (codes.VOICE_NOT_SYNTHETIC, "voice.sha256")
    assert not clip_path(service.world.store, sha).exists(), "a refused clip is never copied"


def test_a_file_whose_sha256_is_not_the_one_sent_is_refused_s17_3(service: Service) -> None:
    other = service.world.root / "elsewhere" / "changed.wav"
    write_wav(other, freq=330.0)
    error = refused(service, service.request(LAMPS, voice=service.voice(path=str(other))))
    assert (error.code, error.field) == (codes.VOICE_FILE_MISMATCH, "voice.sha256")
    assert error.details is not None and error.details["expected"] == service.world.clip_sha256
    assert error.details["actual"] == sha256_of(other)


def test_a_relative_path_is_refused_s17_3(service: Service) -> None:
    error = refused(service, service.request(LAMPS, voice=service.voice(path="voice/clip.wav")))
    assert (error.code, error.field) == (codes.PATH_NOT_ALLOWED, "voice.path")


@pytest.mark.skipif(sys.platform != "win32", reason="the platform's path check is Windows only (section 17.3)")
def test_a_network_path_is_refused_by_the_platform_s17_3(service: Service) -> None:
    backend, _, _ = make_backend(service.world)
    backend.platform = get_platform()
    body = service.request(LAMPS, voice=service.voice(path="\\\\fileserver\\share\\clip.wav"))
    with pytest.raises(NarrationError) as caught:
        backend.submit_job_sync(body)
    assert (caught.value.code, caught.value.field) == (codes.PATH_NOT_ALLOWED, "voice.path")


def test_a_clip_over_30_seconds_is_unsupported_audio_s17_3(service: Service) -> None:
    long = service.world.root / "elsewhere" / "long.wav"
    sha = write_wav(long, seconds=31.0)
    allow(service, sha)
    error = refused(service, service.request(LAMPS, voice=service.voice(path=str(long), sha256=sha)))
    assert (error.code, error.field) == (codes.UNSUPPORTED_AUDIO, "voice.path")
    assert not clip_path(service.world.store, sha).exists()


def test_a_clip_that_is_not_a_wav_is_unsupported_audio_s17_3(service: Service) -> None:
    flac = service.world.root / "elsewhere" / "clip.wav"
    flac.parent.mkdir(parents=True)
    soundfile.write(str(flac), np.zeros(24_000, dtype=np.float32), 24_000, format="FLAC")
    sha = sha256_of(flac)
    allow(service, sha)
    error = refused(service, service.request(LAMPS, voice=service.voice(path=str(flac), sha256=sha)))
    assert (error.code, error.field) == (codes.UNSUPPORTED_AUDIO, "voice.path")


def test_an_unmeasured_voice_is_refused_and_told_to_measure_s3_2(service: Service) -> None:
    other = service.world.root / "elsewhere" / "new-voice.wav"
    sha = write_wav(other, freq=260.0)
    allow(service, sha)
    error = refused(service, service.request(LAMPS, voice=service.voice(path=str(other), sha256=sha)))
    assert (error.code, error.field, error.retryable) == (codes.VOICE_NOT_MEASURED, "voice", False)
    assert "measure_voice" in error.hint
    assert jobs_in(service) == 0


# ======================================================================== the engine and the text


def test_an_engine_other_than_the_one_expected_is_engine_changed_s7_3(service: Service) -> None:
    error = refused(service, service.request(LAMPS, expect_engine_profile="sha256:" + "0" * 64))
    assert (error.code, error.field) == (codes.ENGINE_CHANGED, "expect_engine_profile")
    assert error.details == {"expected": "sha256:" + "0" * 64, "current": ENGINE_HASH}
    accepted = service.backend.submit_job_sync(service.request(LAMPS, expect_engine_profile=ENGINE_HASH))
    assert accepted["engine_profile"]["hash"] == ENGINE_HASH


def test_a_control_is_refused_by_every_current_engine_s3_3(service: Service) -> None:
    body = service.request(LAMPS)
    body["segments"][0]["controls"] = {"pace": 1.1}
    error = refused(service, body)
    assert (error.code, error.field) == (codes.CONTROL_UNSUPPORTED, "segments[0].controls.pace")


def test_an_over_long_segment_is_warned_about_never_refused_s3_2(service: Service) -> None:
    service.world.measure(max_segment_chars=40)
    submitted = service.backend.submit_job_sync(service.request(LAMPS, KETTLE))
    assert submitted["status"] == "queued"
    too_long = [w for w in submitted["warnings"] if w["code"] == codes.SEGMENT_TOO_LONG]
    assert {w["segment_id"] for w in too_long} == {"p01", "p02"}


def test_text_warnings_are_returned_and_strict_text_refuses_them_s9_1(service: Service) -> None:
    text = "The old tower is 40 metres tall, or so the ferry keeper says."
    submitted = service.backend.submit_job_sync(service.request(text))
    assert [w["code"] for w in submitted["warnings"]] == ["WRITTEN_FORM_TOKEN"]
    error = refused(service, service.request(text, **{"options": {"strict_text": True}}))
    assert (error.code, error.field) == (codes.TEXT_REFUSED, "segments[0].text")


def test_refused_characters_are_refused_whatever_strict_text_says_s9_1(service: Service) -> None:
    error = refused(service, service.request("The ferry [leaves] at dawn."))
    assert error.code == codes.TEXT_REFUSED


# ======================================================================== dry_run (section 7.3)


def test_dry_run_plans_and_queues_nothing_s7_3(service: Service) -> None:
    planned = service.backend.submit_job_sync(service.request(LAMPS, ORCHARD, **{"options": {"dry_run": True}}))
    assert (planned["job_id"], planned["status"]) == (None, "planned")
    assert planned["plan"]["renders_needed"] == 2
    assert planned["plan"]["est_audio_s"] > 0
    assert [t["segment_id"] for t in planned["text"]] == ["p01", "p02"]
    assert all(t["max_segment_chars"] == 400 for t in planned["text"])
    assert jobs_in(service) == 0
    assert service.launcher.ensured == []
    assert not clip_path(service.world.store, service.world.clip_sha256).exists(), "a dry run copies nothing"


def test_plan_counts_only_what_the_cache_does_not_hold_s10_2(service: Service) -> None:
    service.backend.submit_job_sync(service.request(LAMPS))
    service.world.run()
    dry = {"options": {"dry_run": True}}
    planned = service.backend.submit_job_sync(service.request(LAMPS, KETTLE, **dry))["plan"]
    assert (planned["segments_cached"], planned["renders_needed"]) == (1, 1)
    assert (planned["deliveries_needed"], planned["analyses_needed"]) == (1, 1)
    renamed = service.backend.submit_job_sync(service.request(LAMPS, ids=["opening"], **dry))["plan"]
    assert renamed["segments_cached"] == 1, "the segment id is in no key (section 10.3)"


def test_plan_without_the_analysis_pins_counts_every_analysis_s10_2(service: Service) -> None:
    service.backend.submit_job_sync(service.request(LAMPS))
    service.world.run()
    backend, _, _ = make_backend(service.world, pins=False)
    planned = backend.submit_job_sync(service.request(LAMPS, **{"options": {"dry_run": True}}))["plan"]
    assert (planned["renders_needed"], planned["deliveries_needed"], planned["analyses_needed"]) == (0, 0, 1)


def test_the_clip_path_in_the_request_is_read_only_through_the_platform_s17_3(service: Service) -> None:
    service.backend.submit_job_sync(service.request(LAMPS))
    assert service.platform.read_paths == [str(service.world.clip)]
