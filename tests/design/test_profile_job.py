"""The ``profile`` job (design sections 3.6, 7.6, 15 and 17.3) on the fake workers: any WAV the owner can read, by
path and sha256, profiled on the CPU."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from narration.contracts import codes, names
from narration.design.work import ProfileReplyError, profile_record
from tests.backend.support import write_wav

from .support import DesignWorld


def profiled(world: DesignWorld, job_id: str) -> dict[str, Any]:
    job = world.job(job_id)
    assert job.status == "completed", (job.status, job.error)
    assert job.result is not None
    return job.result


def test_any_readable_wav_is_profiled_with_its_pictures_s3_6_s17_3(world: DesignWorld) -> None:
    audio = world.root / "elsewhere" / "long-take.wav"
    sha = write_wav(audio, seconds=40.0)  # neither a synthetic voice nor a 30 s clip: neither is required
    job = world.profile(audio, sha)
    world.run()
    assert profiled(world, job.job_id) == {"audio_sha256": sha, "profile_version": names.PROFILE_VERSION}
    record = world.store.get_profile(sha, names.PROFILE_VERSION)
    assert record is not None
    assert record.measurements.duration_s == pytest.approx(40.0)
    assert record.measurements.speaking_rate_wpm is None, "profile_voice is sent no transcript"
    for picture in (record.pictures.spectrogram, record.pictures.pitch):
        assert Path(picture).is_file() and Path(picture).parent == world.store.profile_dir(sha)
    assert world.job(job.job_id).outcome == "all_passed"


def test_a_profile_runs_on_the_cpu_and_never_takes_the_gpu_s7_6(world: DesignWorld) -> None:
    audio = world.root / "elsewhere" / "take.wav"
    sha = write_wav(audio)
    job = world.profile(audio, sha)
    world.run()
    profiled(world, job.job_id)
    loads = [p for g, o, p in world.pool.requests if g == "qa" and o == "load"]
    assert loads == [{"device": "cpu", "models": {}}]
    assert world.pool.gpu_holder is None
    request = next(p for _, o, p in world.pool.requests if o == "profile")
    assert "transcript" not in request
    assert Path(request["wav"]).is_relative_to(world.store.root), "a worker reads only inside the store"


def test_a_profile_uses_the_qa_group_as_it_is_when_it_is_loaded_s4(world: DesignWorld) -> None:
    design = world.design(takes=1)
    world.run()
    assert world.job(design.job_id).status == "completed"
    loads = len(world.pool.loads)
    audio = world.root / "elsewhere" / "take.wav"
    sha = write_wav(audio)
    job = world.profile(audio, sha)
    world.run()
    profiled(world, job.job_id)
    assert len(world.pool.loads) == loads, "no load: the resident QA group's worker profiles as it is"


def test_the_same_bytes_are_answered_from_the_cache_s15(world: DesignWorld) -> None:
    audio = world.root / "elsewhere" / "take.wav"
    sha = write_wav(audio)
    first = world.profile(audio, sha)
    world.run()
    copy = world.root / "another" / "copy.wav"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(audio.read_bytes())
    again = world.profile(copy, sha)
    world.run()
    assert profiled(world, again.job_id) == profiled(world, first.job_id)
    assert len(world.requests("profile")) == 1


@pytest.mark.parametrize(
    ("case", "code", "field"),
    [
        ("changed", codes.VOICE_FILE_MISMATCH, "audio.sha256"),
        ("gone", codes.PATH_NOT_ALLOWED, "audio.path"),
    ],
)
def test_audio_that_changed_or_went_fails_the_job_naming_the_field_s14(
    world: DesignWorld, case: str, code: str, field: str
) -> None:
    audio = world.root / "elsewhere" / "take.wav"
    sha = write_wav(audio)
    job = world.profile(audio, sha)
    if case == "changed":
        write_wav(audio, freq=440.0)
    else:
        audio.unlink()
    world.run()
    done = world.job(job.job_id)
    assert done.status == "failed" and done.error is not None
    assert (done.error.code, done.error.field, done.error.retryable) == (code, field, False)
    assert done.error.hint
    assert world.requests("profile") == []


class _Silent:
    """A QA client whose ``profile`` reply measures no brightness, as the real worker's is for an all-zero file."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def request(self, op: str, payload: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]:
        reply = self.inner.request(op, payload, timeout_s=timeout_s)
        if op == "profile":
            reply["measurements"]["spectral_centroid_hz"] = None
        return reply


def test_silent_audio_is_unsupported_audio_s14(world: DesignWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = world.root / "elsewhere" / "silence.wav"
    sha = write_wav(audio)
    pool = world.pool
    real = pool.client
    monkeypatch.setattr(
        pool, "client", lambda group, **kw: _Silent(real(group, **kw)) if group == "qa" else real(group, **kw)
    )
    job = world.profile(audio, sha)
    world.run()
    done = world.job(job.job_id)
    assert done.status == "failed" and done.error is not None
    assert (done.error.code, done.error.field) == (codes.UNSUPPORTED_AUDIO, "audio.path")
    assert done.error.hint


def test_a_profile_reply_is_read_as_the_record_s3_6() -> None:
    measurements = {
        "duration_s": 2.0,
        "pitch_median_hz": 120.0,
        "pitch_p10_hz": 100.0,
        "pitch_p90_hz": 150.0,
        "pitch_range_st": 7.02,
        "speaking_rate_wpm": None,
        "pause_ratio": 0.1,
        "loudness_lufs": -20.0,
        "spectral_centroid_hz": 900.0,
        "hnr_db": 18.0,
        "cpps_db": 12.0,
    }
    pictures = {"spectrogram": "/s/spectrogram.png", "pitch": "/s/pitch.png"}
    record = profile_record("a" * 64, {"measurements": measurements, "pictures": pictures, "method": {}})
    assert record.profile_version == names.PROFILE_VERSION
    assert record.measurements.pitch_median_hz == 120.0 and record.pictures.pitch == "/s/pitch.png"
    with pytest.raises(ProfileReplyError):
        profile_record("a" * 64, {"measurements": {**measurements, "jitter": 0.1}, "pictures": pictures})
    with pytest.raises(ProfileReplyError):
        profile_record("a" * 64, {"measurements": measurements, "pictures": {"pitch": "/s/pitch.png"}})
