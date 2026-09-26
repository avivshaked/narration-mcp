"""The voice profile's measures, validated on synthetic signals with known values (design section 3.6 as changed
by DC-1; plan.md WP22).

Every signal here is made by the test from a seed: harmonic tones at a known f0, the same tone in white noise at a
known signal-to-noise ratio, glides whose pitch percentiles are known in closed form, tones at a known level for
loudness, and tone bursts with known pauses. No audio file is read.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from narration_worker_qa import voice

SR = voice.ANALYSIS_RATE


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def harmonic(f0: float, seconds: float, *, rolloff: float = 1.0, seed: int = 0, sr: int = SR) -> np.ndarray:
    """A harmonic tone at ``f0``: every harmonic below 95 % of Nyquist, amplitude 1/k**rolloff, random phases."""
    rng = _rng(seed)
    t = np.arange(round(seconds * sr)) / sr
    x = np.zeros_like(t)
    k = 1
    while k * f0 < sr / 2 * 0.95:
        x += np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) / k**rolloff
        k += 1
    return x / np.abs(x).max() * 0.5


def with_noise(x: np.ndarray, snr_db: float, seed: int = 1) -> np.ndarray:
    """``x`` plus white noise at ``snr_db`` (power ratio over the whole signal)."""
    noise = _rng(seed).standard_normal(x.shape[0])
    noise *= math.sqrt(float(np.mean(x**2)) / 10 ** (snr_db / 10) / float(np.mean(noise**2)))
    return x + noise


def glide(start_hz: float, octaves: float, seconds: float) -> np.ndarray:
    """A harmonic tone whose pitch rises exponentially by ``octaves`` over ``seconds``."""
    t = np.arange(round(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(start_hz * 2 ** (octaves * t / seconds)) / SR
    return 0.3 * np.sum([np.sin(k * phase) / k for k in range(1, 20)], axis=0)


# ---------------------------------------------------------------------- pitch


def test_pitch_track_of_a_steady_tone_is_its_f0_s3_6() -> None:
    track = voice.pitch_track(harmonic(150.0, 2.0), 50.0, 400.0)
    assert track.hop_s == 0.01
    voiced = track.f0_hz[np.isfinite(track.f0_hz)]
    assert voiced.size > 0.9 * track.f0_hz.size
    assert np.median(voiced) == pytest.approx(150.0, rel=0.005)
    assert float(np.median(track.voiced_probability)) > 0.5  # the edge frames, half padding, are less sure


def test_pitch_percentiles_of_a_glide_are_known_in_closed_form_s3_6() -> None:
    # Over an exponential glide of one octave, a fraction q of the frames lies below 100 * 2**q Hz.
    stats = voice.pitch_stats(voice.pitch_track(glide(100.0, 1.0, 3.0), 50.0, 400.0).f0_hz)
    assert stats is not None
    assert stats.median_hz == pytest.approx(100 * 2**0.5, rel=0.01)
    assert stats.p10_hz == pytest.approx(100 * 2**0.1, rel=0.01)
    assert stats.p90_hz == pytest.approx(100 * 2**0.9, rel=0.01)
    assert stats.range_st == pytest.approx(9.6, abs=0.15)  # 12 semitones * (0.9 - 0.1)


def test_silence_has_no_pitch_s3_6() -> None:
    track = voice.pitch_track(np.zeros(SR), 50.0, 400.0)
    assert not np.isfinite(track.f0_hz).any()
    assert voice.pitch_stats(track.f0_hz) is None


@pytest.mark.parametrize(("fmin", "fmax"), [(20.0, 400.0), (400.0, 400.0), (500.0, 400.0), (50.0, 9000.0)])
def test_pitch_track_refuses_a_range_it_cannot_track_s3_6(fmin: float, fmax: float) -> None:
    with pytest.raises(ValueError, match="fmin_hz"):
        voice.pitch_track(np.zeros(SR), fmin, fmax)


def test_track_reply_is_null_where_unvoiced_s3_6() -> None:
    track = voice.PitchTrack(0.01, np.array([np.nan, 120.123456, 130.0]), np.array([0.01, 0.9, 0.8]))
    f0, probability = voice.track_reply(track)
    assert f0 == [None, 120.123, 130.0]
    assert probability == [0.01, 0.9, 0.8]


# ---------------------------------------------------------------------- speech, pauses, speaking rate


def _bursts() -> np.ndarray:
    """0.3 s of silence, five 0.5 s tone bursts with 0.4 s of silence between them, then 0.2 s of silence."""
    parts = [np.zeros(int(0.3 * SR))]
    for i in range(5):
        parts.append(harmonic(150.0, 0.5, seed=i))
        if i < 4:
            parts.append(np.zeros(int(0.4 * SR)))
    parts.append(np.zeros(int(0.2 * SR)))
    return np.concatenate(parts)


def test_speech_span_and_pauses_of_tone_bursts_s3_6() -> None:
    speech = voice.find_speech(_bursts())
    assert speech.start_s == pytest.approx(0.3)
    assert speech.end_s == pytest.approx(0.3 + 4.1)
    assert speech.pause_s == pytest.approx(1.6)
    assert speech.pause_ratio == pytest.approx(1.6 / 4.1)


def test_a_short_gap_is_not_a_pause_s3_6() -> None:
    gap = np.zeros(int(0.1 * SR))  # 5 frames, under MIN_PAUSE_FRAMES
    x = np.concatenate([harmonic(150.0, 0.5), gap, harmonic(150.0, 0.5)])
    assert voice.find_speech(x).pause_s == 0.0


def test_silence_is_all_pause_s3_6() -> None:
    speech = voice.find_speech(np.zeros(SR))
    assert (speech.start_s, speech.end_s, speech.span_s, speech.pause_ratio) == (None, None, 0.0, 1.0)


def test_speaking_rate_counts_spoken_words_over_the_voiced_span_s3_6() -> None:
    speech = voice.find_speech(_bursts())
    # 8 words; the dash and the lone comma are not words (the server's rule).
    assert voice.speaking_rate_wpm("one two three , four five six seven — eight", speech) == pytest.approx(8 / 4.1 * 60)
    assert voice.speaking_rate_wpm(None, speech) is None
    assert voice.speaking_rate_wpm("", speech) is None
    assert voice.speaking_rate_wpm("... —", speech) is None


@pytest.mark.parametrize(
    ("text", "words"),
    [("Rain came over the ridge.", 5), ("It's 42 — and more", 4), ("« quoted » word", 2), ("", 0), ("  ", 0)],
)
def test_spoken_words_are_counted_as_the_server_counts_them_s3_6(text: str, words: int) -> None:
    assert voice.spoken_words(text) == words


# ---------------------------------------------------------------------- loudness


@pytest.mark.parametrize("rate", [16_000, 24_000, 48_000])
def test_loudness_of_a_997_hz_sine_s3_6(rate: int) -> None:
    # BS.1770: a 997 Hz sine at amplitude A in one channel reads 20*log10(A) - 3.01 LKFS.
    t = np.arange(rate * 5) / rate
    measured = voice.loudness_lufs(0.1 * np.sin(2 * np.pi * 997 * t), rate)
    assert measured == pytest.approx(20 * math.log10(0.1) - 3.01, abs=0.1)


def test_loudness_of_silence_is_none_s3_6() -> None:
    assert voice.loudness_lufs(np.zeros(SR * 2), SR) is None


def test_loudness_of_a_clip_shorter_than_a_block_is_measured_s3_6() -> None:
    t = np.arange(int(0.2 * SR)) / SR
    measured = voice.loudness_lufs(0.1 * np.sin(2 * np.pi * 997 * t), SR)
    assert measured is not None and measured < 20 * math.log10(0.1) - 3.01  # measured as if silence followed


# ---------------------------------------------------------------------- spectral centroid


def test_spectral_centroid_of_tones_s3_6() -> None:
    t = np.arange(SR * 2) / SR
    one = 0.3 * np.sin(2 * np.pi * 1000 * t)
    two = 0.3 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1500 * t)
    assert voice.spectral_centroid_hz(one, voice.speech_frames(one)) == pytest.approx(1000.0, rel=0.005)
    assert voice.spectral_centroid_hz(two, voice.speech_frames(two)) == pytest.approx(1000.0, rel=0.005)


def test_spectral_centroid_without_speech_uses_every_frame_with_energy_s3_6() -> None:
    t = np.arange(SR) / SR
    quiet = 1e-5 * np.sin(2 * np.pi * 2000 * t)  # below the -70 dBFS floor: no speech frame
    assert not voice.speech_frames(quiet).any()
    assert voice.spectral_centroid_hz(quiet, voice.speech_frames(quiet)) == pytest.approx(2000.0, rel=0.005)
    assert voice.spectral_centroid_hz(np.zeros(SR), voice.speech_frames(np.zeros(SR))) is None


# ---------------------------------------------------------------------- HNR


@pytest.mark.parametrize("f0", [100.0, 150.0, 220.0])
@pytest.mark.parametrize("snr", [5.0, 10.0, 15.0, 20.0])
def test_hnr_of_a_tone_in_white_noise_is_its_snr_s3_6(f0: float, snr: float) -> None:
    # For a periodic signal in white noise, Boersma's r is the periodic share of the power (DC-1).
    assert voice.hnr_db(with_noise(harmonic(f0, 2.0), snr)) == pytest.approx(snr, abs=1.0)


def test_hnr_is_high_for_a_clean_tone_and_none_without_voicing_s3_6() -> None:
    clean = voice.hnr_db(harmonic(150.0, 2.0))
    assert clean is not None and clean > 30.0
    assert voice.hnr_db(np.zeros(SR)) is None
    noise = voice.hnr_db(_rng(5).standard_normal(SR * 2) * 0.1)
    assert noise is None or noise < 0.0


# ---------------------------------------------------------------------- CPPS


@pytest.mark.parametrize("f0", [100.0, 150.0, 220.0])
def test_cepstral_peak_is_at_the_period_s3_6(f0: float) -> None:
    _, quefrency, _ = voice.cepstral_peaks(harmonic(f0, 1.0, rolloff=2.0))
    assert float(np.median(quefrency)) == pytest.approx(1.0 / f0, rel=0.03)


def test_cpps_grows_with_the_signal_to_noise_ratio_s3_6() -> None:
    # CPPS has no closed form, so it is checked for order: a clear periodic voice (a -12 dB/octave source, like
    # a glottal pulse train) ranks above the same voice in more noise, and all rank above noise alone.
    tone = harmonic(130.0, 2.0, rolloff=2.0)
    values = []
    for snr in (0.0, 5.0, 10.0, 15.0, 20.0):
        x = with_noise(tone, snr)
        value = voice.cpps_db(x, voice.speech_frames(x))
        assert value is not None
        values.append(value)
    assert values == sorted(values) and values[-1] - values[0] > 3.0
    noise = _rng(7).standard_normal(SR * 2) * 0.1
    noise_cpps = voice.cpps_db(noise, voice.speech_frames(noise))
    assert noise_cpps is not None and noise_cpps < values[0]


def test_cpps_without_speech_is_none_s3_6() -> None:
    assert voice.cpps_db(np.zeros(SR), voice.speech_frames(np.zeros(SR))) is None


# ---------------------------------------------------------------------- the profile


def test_profile_measures_every_key_at_any_rate_and_channel_count_s3_6() -> None:
    x16 = with_noise(harmonic(140.0, 3.0, rolloff=2.0), 20.0)
    for rate in (16_000, 24_000, 48_000):
        from scipy.signal import resample_poly

        audio = x16 if rate == SR else resample_poly(x16, rate // 1000, 16)
        stereo = np.stack([audio, audio], axis=1)
        for signal in (audio, stereo):
            measurements, track, x = voice.measure(signal, rate, "one two three four five six")
            assert tuple(measurements) == voice.PROFILE_KEYS
            assert measurements["duration_s"] == pytest.approx(3.0, abs=1e-3)
            assert measurements["pitch_median_hz"] == pytest.approx(140.0, rel=0.01)
            assert measurements["hnr_db"] == pytest.approx(20.0, abs=1.5)
            assert all(isinstance(v, float) for v in measurements.values())
            assert x.shape[0] == pytest.approx(3.0 * SR, abs=2)
            assert track.hop_s == 0.01


def test_profile_keys_are_the_servers_profile_measurements_s3_6() -> None:
    fake = pytest.importorskip("narration_worker.fake.handler")
    assert voice.PROFILE_KEYS == fake.PROFILE_KEYS  # the fake mirrors the server's models.ProfileMeasurements
    assert voice.PICTURES == fake.PROFILE_PICTURES


def test_profile_of_silence_is_measured_without_a_crash_s3_6() -> None:
    measurements, _, _ = voice.measure(np.zeros(SR * 2), SR, "some words here")
    assert measurements["duration_s"] == 2.0
    assert measurements["pause_ratio"] == 1.0
    for key in ("pitch_median_hz", "speaking_rate_wpm", "loudness_lufs", "hnr_db", "cpps_db", "spectral_centroid_hz"):
        assert measurements[key] is None, key


def test_method_names_every_measure_and_the_versions_s3_6() -> None:
    method = voice.method()
    for key in ("version", "f0", "pitch", "speech", "speaking_rate", "pause_ratio", "loudness", "hnr", "cpps"):
        assert method[key], key
    assert method["version"] == voice.METHOD_VERSION
    assert "librosa.pyin" in method["f0"] and "pyloudnorm" in method["loudness"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pictures_are_pngs_written_through_temporary_names_and_reproducible_s3_6(tmp_path: Path) -> None:
    x = with_noise(harmonic(140.0, 2.0), 20.0)
    measurements, track, x16 = voice.measure(x, SR, None)
    first = voice.draw_pictures(x16, track, measurements, tmp_path / "a")
    second = voice.draw_pictures(x16, track, measurements, tmp_path / "b")
    assert set(first) == set(voice.PICTURES)
    for name in voice.PICTURES:
        path = Path(first[name])
        assert path.name == f"{name}.png" and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert _sha256(path) == _sha256(Path(second[name]))
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["pitch.png", "spectrogram.png"]


def test_pictures_of_silence_are_drawn_s3_6(tmp_path: Path) -> None:
    measurements, track, x16 = voice.measure(np.zeros(SR), SR, None)
    pictures = voice.draw_pictures(x16, track, measurements, tmp_path)
    assert all(Path(p).is_file() for p in pictures.values())
