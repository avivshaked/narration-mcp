"""The fake worker's outputs, as the server's own code sees them (plan.md WP16).

These need numpy, scipy and soundfile, which the server has and ``narration_worker`` does not: the fake's
WAV reads like any other, and its QA ops still hear a take after delivery post-processing (section 13:
trim, 48 kHz, static gain, fades, PCM_24).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pytest
import soundfile as sf
from narration_worker.fake.audio import HEAD
from narration_worker.fake.handler import EMBEDDING_DIM, PROFILE_KEYS, PROFILE_PICTURES
from narration_worker.fake.wav import read_wav
from scipy.signal import resample_poly

from narration.contracts import worker as protocol
from narration.contracts.models import ProfileMeasurements, ProfilePictures
from narration.workers import SubprocessWorkerClient

from .conftest import ClientFactory, call_cap

TIMEOUT = 30.0
VOICE = "sha256:" + "ef" * 32
TEXT = "Some of them thrive, and some of them simply disappear, and the water keeps no record of either."


def _required(typed_dict: Any) -> set[str]:
    """A reply's required keys, besides id and ok."""
    return set(typed_dict.__required_keys__) - {"id", "ok"}


def test_optional_reply_keys_are_optional_at_run_time_appA() -> None:
    """Contracts 1.1: the protocol module keeps ``NotRequired`` visible to ``TypedDict`` (WP16's request)."""
    assert "error" not in protocol.Reply.__required_keys__
    assert "new_tokens" in protocol.AudioReply.__optional_keys__
    assert _required(protocol.AudioReply) == {"sample_rate", "samples", "gen_s", "hit_token_cap"}


@pytest.fixture
def ready(make_client: ClientFactory, store: Path) -> SubprocessWorkerClient:
    client = make_client()
    client.start()
    client.request("load", {"device": "cpu"}, timeout_s=TIMEOUT)
    clip = store / "scratch" / "clip.wav"
    design = client.request(
        "design",
        {
            "description": "A warm voice.",
            "design_text": "Far below the surface.",
            "language": "English",
            "seed": 9,
            "max_new_tokens": call_cap("Far below the surface.", design=True),
            "out_path": str(clip),
        },
        timeout_s=TIMEOUT,
    )
    assert _required(protocol.AudioReply) <= set(design)
    client.request(
        "prepare_voice",
        {"voice_hash": VOICE, "ref_wav": str(clip), "ref_text": "Far below the surface.", "x_vector_only_mode": False},
        timeout_s=TIMEOUT,
    )
    return client


def _deliver(raw: Path, out: Path, *, gain: float, trim_s: float) -> None:
    """Post-processing as section 13 describes it, closely enough for the fake's ears."""
    samples, rate = sf.read(raw, dtype="float64")
    delivered = resample_poly(samples, 2, 1)[round(trim_s * 2 * rate) :] * gain
    fade = int(0.01 * 2 * rate)
    delivered[:fade] *= np.linspace(0.0, 1.0, fade)
    delivered[-fade:] *= np.linspace(1.0, 0.0, fade)
    sf.write(out, np.clip(delivered, -1.0, 1.0), 2 * rate, subtype="PCM_24")


def test_soundfile_reads_the_fakes_wav_as_float32_mono_appA(ready: SubprocessWorkerClient, store: Path) -> None:
    raw = store / "scratch" / "take.wav"
    reply = ready.request(
        "synthesize",
        {
            "voice_hash": VOICE,
            "engine_text": TEXT,
            "language": "English",
            "seed": 5,
            "max_new_tokens": call_cap(TEXT),
            "out_path": str(raw),
        },
        timeout_s=TIMEOUT,
    )
    assert _required(protocol.AudioReply) <= set(reply)
    info = sf.info(raw)
    assert (info.samplerate, info.channels, info.subtype, info.frames) == (24_000, 1, "FLOAT", reply["samples"])
    data, _ = sf.read(raw, dtype="float32")
    assert np.array_equal(data, np.asarray(read_wav(raw).samples, dtype=np.float32))
    assert float(np.max(np.abs(data))) == pytest.approx(0.5, abs=1e-3)


def test_the_fake_hears_a_post_processed_delivery_file_s13(ready: SubprocessWorkerClient, store: Path) -> None:
    raw = store / "scratch" / "take.wav"
    ready.request(
        "synthesize",
        {
            "voice_hash": VOICE,
            "engine_text": TEXT,
            "language": "English",
            "seed": 5,
            "max_new_tokens": call_cap(TEXT),
            "out_path": str(raw),
        },
        timeout_s=TIMEOUT,
    )
    delivery = store / "takes" / "delivery.wav"
    delivery.parent.mkdir()
    trim_s = HEAD / 24_000 - 0.08
    _deliver(raw, delivery, gain=1.7, trim_s=trim_s)

    def hear(path: Path) -> dict[str, Any]:
        reply = ready.request(
            "transcribe",
            {"wav": str(path), "language": "English", "word_timestamps": True, "long_form": True},
            timeout_s=TIMEOUT,
        )
        assert _required(protocol.TranscribeReply) <= set(reply)
        return reply

    from_raw, from_delivery = hear(raw), hear(delivery)
    assert from_delivery["text"] == from_raw["text"] == TEXT
    for a, b in zip(from_raw["words"], from_delivery["words"], strict=True):
        assert b["start_s"] == pytest.approx(a["start_s"] - trim_s, abs=0.003)
        assert b["end_s"] == pytest.approx(a["end_s"] - trim_s, abs=0.003)

    def embed(path: Path) -> list[float]:
        reply = ready.request("embed", {"wav": str(path), "device": "cpu"}, timeout_s=TIMEOUT)
        assert _required(protocol.EmbedReply) <= set(reply) and reply["dim"] == EMBEDDING_DIM
        return reply["embedding"]

    assert embed(delivery) == embed(raw)

    tokens = [ch for word in "Some of them".upper().split() for ch in [*word, "|"]][:-1]
    aligned = ready.request("align", {"wav": str(delivery), "tokens": tokens}, timeout_s=TIMEOUT)
    assert _required(protocol.AlignReply) <= set(aligned)
    assert aligned["num_frames"] == (round(sf.info(delivery).duration * 16_000) - 400) // 320 + 1

    pitch = ready.request("f0", {"wav": str(delivery), "fmin_hz": 60.0, "fmax_hz": 600.0}, timeout_s=TIMEOUT)
    assert _required(protocol.F0Reply) <= set(pitch)
    assert math.isclose(len(pitch["f0_hz"]) * 0.01, sf.info(delivery).duration, abs_tol=0.011)


def test_the_profile_reply_has_the_profile_records_keys_s3_6(ready: SubprocessWorkerClient, store: Path) -> None:
    reply = ready.request(
        "profile",
        {"wav": str(store / "scratch" / "clip.wav"), "out_dir": str(store / "scratch" / "profile"), "transcript": None},
        timeout_s=TIMEOUT,
    )
    assert _required(protocol.ProfileReply) <= set(reply)
    assert tuple(reply["measurements"]) == PROFILE_KEYS
    assert set(reply["measurements"]) == {f.name for f in dataclasses.fields(ProfileMeasurements)}
    assert set(reply["pictures"]) == set(PROFILE_PICTURES) == {f.name for f in dataclasses.fields(ProfilePictures)}
    assert reply["measurements"]["speaking_rate_wpm"] is None  # no transcript given
    hints = get_type_hints(ProfileMeasurements)
    for name, value in reply["measurements"].items():
        assert value is None or isinstance(value, float | int), name
        assert value is not None or type(None) in getattr(hints[name], "__args__", ()), name
