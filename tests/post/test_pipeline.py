"""The delivery pipeline end to end (design section 13; the take record of App. B; flags of section 14)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile
from jsonschema import Draft202012Validator

from narration.config import DeliveryConfig
from narration.contracts import codes
from narration.contracts.interfaces import DeliveryOutput, DeliveryProcessor
from narration.contracts.models import DeliveryAudio, TakeRecord
from narration.contracts.schemas import record_schema
from narration.contracts.serial import from_json, to_json
from narration.post import GAIN_HIGH_DB, DeliveryPipeline, deliver, delivery_tools, fade_edges
from narration.post import pcm as pcm_module
from narration.post.loudness import integrated_loudness, true_peak_dbtp
from narration.post.pcm import decode_wav, lsb_dbfs
from narration.post.resample import resample, resampled_length
from narration.post.trim import find_trim

from .signals import Audio, reference_loudness, reference_true_peak, speech_like, write_raw

PROFILE = DeliveryConfig()


def _deliver(x: Audio, rate: int = 24000, profile: DeliveryConfig = PROFILE) -> tuple[Audio, DeliveryOutput]:
    made = deliver(x, rate, profile)
    audio, sample_rate = decode_wav(made.wav)
    assert sample_rate == profile.sample_rate
    return audio, made.output


def test_pipeline_is_a_delivery_processor_s13() -> None:
    assert isinstance(DeliveryPipeline(), DeliveryProcessor)
    assert DeliveryPipeline().tools == delivery_tools()


def test_delivery_is_a_48k_pcm24_mono_wav_s13(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw.wav", tmp_path / "delivery.wav"
    write_raw(str(raw), speech_like(), 24000)
    result = DeliveryPipeline().process(raw, out, PROFILE)
    info = soundfile.info(str(out))
    assert (info.format, info.subtype, info.channels, info.samplerate) == ("WAV", "PCM_24", 1, 48000)
    assert info.frames == result.samples
    assert result.sample_rate == 48000
    assert result.duration_s == result.samples / 48000


@pytest.mark.parametrize("rate", [24000, 22050, 16000, 44100, 48000])
@pytest.mark.parametrize(("head_s", "tail_s"), [(0.5, 0.7), (0.04, 0.3), (0.2, 0.05), (0.0, 0.0)])
def test_delivery_length_formula_holds_to_the_sample_s13(rate: int, head_s: float, tail_s: float) -> None:
    x = speech_like(rate, head_s=head_s, bursts=(1.2, 0.9), gaps=(0.4,), tail_s=tail_s)
    output = deliver(x, rate, PROFILE).output
    head, tail = round(output.trim.head_s * rate), round(output.trim.tail_s * rate)
    pad = round(output.trim.pad_s * rate)
    assert output.trim.pad_s == 0.08
    kept = x.shape[0] - max(head - pad, 0) - max(tail - pad, 0)
    assert output.samples == resampled_length(kept, rate, 48000)
    if head >= pad and tail >= pad:  # the design's formula, exactly as section 13 writes it
        assert kept == x.shape[0] - (head - pad) - (tail - pad)


def test_delivery_trim_record_states_the_silence_found_s13() -> None:
    output = deliver(speech_like(24000, head_s=0.5, tail_s=0.7), 24000, PROFILE).output
    assert (output.trim.head_s, output.trim.tail_s, output.trim.pad_s) == (0.5, 0.7, 0.08)
    assert output.trim.rule == "p95_frame_rms - 40 dB; head_s/tail_s found, pad_s kept"


def test_delivery_gain_reaches_minus_16_lufs_s13() -> None:
    audio, output = _deliver(speech_like(24000, level=0.05))
    assert output.loudness.target_lufs == -16.0
    assert not output.loudness.ceiling_applied
    assert reference_loudness(audio) == pytest.approx(-16.0, abs=0.05)
    assert output.loudness.measured_lufs == pytest.approx(-16.0, abs=0.01)


def test_delivery_loudness_record_is_measured_on_the_final_file_s13() -> None:
    audio, output = _deliver(speech_like(24000, level=0.2, bursts=(2.0, 1.0), gaps=(0.5,)))
    measured = integrated_loudness(audio, 48000)
    peak = true_peak_dbtp(audio, 48000)
    assert measured is not None and peak is not None
    assert output.loudness.measured_lufs == round(measured, 4)
    assert output.loudness.true_peak_dbtp == round(peak, 4)


def test_delivery_gain_is_stated_exactly_as_applied_s13() -> None:
    x = speech_like(24000, level=0.03)
    audio, output = _deliver(x)
    points = find_trim(x, 24000, rel_db=-40.0, pad_s=0.08)
    expected = fade_edges(resample(x[points.start : points.stop], 24000, 48000), 480)
    expected *= 10.0 ** (output.loudness.gain_db / 20.0)
    assert np.max(np.abs(audio - expected)) <= 0.5 / 2**23 + 1e-12


def test_true_peak_ceiling_wins_over_the_loudness_target_s13() -> None:
    x = speech_like(24000, level=0.02)
    x[30000:30010] = 0.9  # a click: far louder in peak than in loudness
    audio, output = _deliver(x)
    loud = output.loudness
    assert loud.ceiling_applied
    assert loud.true_peak_dbtp <= -1.0
    peak = true_peak_dbtp(audio, 48000)
    assert peak is not None and peak <= -1.0
    assert reference_true_peak(audio) <= -1.0 + 0.05
    assert loud.measured_lufs < -16.0
    (flag,) = [f for f in output.flags if f.code == codes.LOUDNESS_UNDER_TARGET]
    assert flag.severity == "info"
    assert flag.retake_trigger is False
    assert flag.details is not None
    assert flag.details["shortfall_db"] == round(-16.0 - loud.measured_lufs, 4)
    assert flag.details["ceiling_dbtp"] == -1.0


def test_true_peak_ceiling_lowers_the_gain_only_as_far_as_needed_s13() -> None:
    x = speech_like(24000, level=0.02)
    x[30000:30010] = 0.9
    _, output = _deliver(x)
    one_step_louder = dataclasses.replace(PROFILE, true_peak_dbtp=output.loudness.true_peak_dbtp + 0.001)
    # With the ceiling just above the peak reached, the same gain (or at most 0.001 dB more) is allowed.
    again = deliver(x, 24000, one_step_louder).output
    assert 0.0 <= again.loudness.gain_db - output.loudness.gain_db <= 0.0011


def test_delivery_below_the_ceiling_has_no_loudness_flag_s13() -> None:
    _, output = _deliver(speech_like(24000, level=0.2))
    assert not output.loudness.ceiling_applied
    assert output.flags == ()


@pytest.mark.parametrize(("level", "flagged"), [(0.002, True), (0.05, True), (0.2, False)])
def test_gain_above_12db_is_flagged_s13(level: float, flagged: bool) -> None:
    _, output = _deliver(speech_like(24000, level=level))
    high = [f for f in output.flags if f.code == codes.GAIN_HIGH]
    assert bool(high) == flagged
    assert (output.loudness.gain_db > GAIN_HIGH_DB) == flagged
    if flagged:
        assert high[0].severity == "info"
        assert high[0].retake_trigger is False
        assert high[0].details == {"gain_db": output.loudness.gain_db, "limit_db": 12.0}


def test_delivery_fades_are_10ms_s13() -> None:
    ramp = fade_edges(np.ones(2000), 480)
    assert ramp[0] == 0.0 and ramp[-1] == 0.0
    assert np.all(np.diff(ramp[:480]) > 0) and np.all(ramp[480:1520] == 1.0)
    assert np.array_equal(ramp[::-1], ramp)
    audio, _ = _deliver(speech_like(24000, head_s=0.02, tail_s=0.02))
    assert audio[0] == 0.0 and audio[-1] == 0.0


def test_delivery_fades_overlap_on_a_very_short_take_s13() -> None:
    ramp = fade_edges(np.ones(300), 480)
    assert ramp[0] == 0.0 and ramp[-1] == 0.0
    assert np.array_equal(ramp[::-1], ramp)


def test_delivery_nonfinite_raw_samples_count_as_zero_s13() -> None:
    x = speech_like(24000)
    y = x.copy()
    y[20000:20005] = [np.nan, np.inf, -np.inf, np.nan, np.nan]
    x[20000:20005] = 0.0
    assert deliver(y, 24000, PROFILE).wav == deliver(x, 24000, PROFILE).wav


def test_delivery_of_digital_silence_gets_no_gain_s13() -> None:
    audio, output = _deliver(np.zeros(24000))
    assert output.samples == 48000
    assert not np.any(audio)
    assert output.loudness.gain_db == 0.0
    assert output.loudness.measured_lufs == -70.0
    assert output.loudness.true_peak_dbtp == lsb_dbfs(24)
    assert not output.loudness.ceiling_applied
    assert output.flags == ()


def test_delivery_of_an_empty_raw_is_empty_s13() -> None:
    audio, output = _deliver(np.zeros(0))
    assert output.samples == 0 and audio.shape[0] == 0


def test_delivery_shorter_than_one_gating_block_is_levelled_s13() -> None:
    # 0.22 s is measured as one 400 ms block (silence after it), which asks for more gain than the peak
    # allows here: the ceiling wins, and the record and flag say so.
    audio, output = _deliver(speech_like(24000, head_s=0.05, bursts=(0.12,), tail_s=0.05))
    assert output.samples < 19200
    assert np.all(np.isfinite(audio))
    loud = output.loudness
    padded = np.concatenate([audio, np.zeros(19200 - audio.shape[0])])
    assert loud.measured_lufs == round(integrated_loudness(padded, 48000) or 0.0, 4)
    assert loud.ceiling_applied and loud.true_peak_dbtp <= -1.0
    (flag,) = [f for f in output.flags if f.code == codes.LOUDNESS_UNDER_TARGET]
    assert flag.details is not None
    assert loud.measured_lufs == round(-16.0 - flag.details["shortfall_db"], 4)


def test_delivery_refuses_stereo_raw_audio_s13(tmp_path: Path) -> None:
    raw = tmp_path / "stereo.wav"
    soundfile.write(str(raw), np.zeros((4800, 2), dtype=np.float32), 24000, subtype="FLOAT")
    with pytest.raises(ValueError, match="mono"):
        DeliveryPipeline().process(raw, tmp_path / "out.wav", PROFILE)


def test_delivery_refuses_an_unsupported_subtype_s13() -> None:
    with pytest.raises(ValueError, match="PCM_24"):
        deliver(speech_like(), 24000, dataclasses.replace(PROFILE, subtype="FLOAT"))


@pytest.mark.parametrize(("subtype", "bits"), [("PCM_16", 16), ("PCM_32", 32)])
def test_delivery_other_pcm_subtypes_follow_the_profile_s13(tmp_path: Path, subtype: str, bits: int) -> None:
    raw, out = tmp_path / "raw.wav", tmp_path / "out.wav"
    write_raw(str(raw), speech_like(), 24000)
    DeliveryPipeline().process(raw, out, dataclasses.replace(PROFILE, subtype=subtype))
    assert soundfile.info(str(out)).subtype == subtype


def test_delivery_is_written_atomically_s13(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw, out = tmp_path / "raw.wav", tmp_path / "delivery.wav"
    write_raw(str(raw), speech_like(), 24000)
    DeliveryPipeline().process(raw, out, PROFILE)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["delivery.wav", "raw.wav"]

    def refuse(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pcm_module.os, "replace", refuse)
    second = tmp_path / "second.wav"
    with pytest.raises(OSError, match="disk full"):
        DeliveryPipeline().process(raw, second, PROFILE)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["delivery.wav", "raw.wav"]


def _quiet_with_a_click() -> Audio:
    # Quiet speech needs over +12 dB; a click meets the ceiling first, at a gain still over +12 dB.
    x = speech_like(24000, level=0.002)
    x[30000:30004] = 0.05
    return x


def test_delivery_flags_use_the_contract_codes_s14() -> None:
    output = deliver(_quiet_with_a_click(), 24000, PROFILE).output
    assert {f.code for f in output.flags} == {codes.LOUDNESS_UNDER_TARGET, codes.GAIN_HIGH}
    for flag in output.flags:
        assert flag.severity in codes.flag_code(flag.code).severities
        assert flag.retake_trigger is codes.is_retake_trigger(flag.code, flag.severity)


def test_delivery_output_fits_the_take_record_schema_s13() -> None:
    output = deliver(_quiet_with_a_click(), 24000, PROFILE).output
    assert len(output.flags) == 2
    take = TakeRecord(
        take_id="tk_0123456789abcdef",
        delivery_key="sha256:" + "0" * 64,
        render_id="rn_0123456789abcdef",
        delivery=DeliveryAudio(
            path="delivery.wav",
            sha256="0" * 64,
            sample_rate=output.sample_rate,
            samples=output.samples,
            duration_s=output.duration_s,
        ),
        trim=output.trim,
        loudness=output.loudness,
        tools=delivery_tools(),
        flags=output.flags,
    )
    data = to_json(take)
    json.dumps(data, allow_nan=False)
    Draft202012Validator(record_schema(TakeRecord)).validate(data)
    assert from_json(TakeRecord, data) == take
