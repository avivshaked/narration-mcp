"""The QA worker's CTC aligner (design section 11.2 steps 2 and 3; plan.md WP15).

Run from the worker's venv: ``workers/qa/.venv/Scripts/python.exe -m pytest workers/qa/tests``. The default
tests need torch and torchaudio but no model. The ``model`` tests load the pinned wav2vec2 snapshot from
``NARRATION_MODELS_ROOT`` on the CPU, and skip cleanly without it.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from narration_worker.protocol import WORKER_ERROR_CODES
from narration_worker_qa import align as qa

REPO = "facebook/wav2vec2-large-960h-lv60-self"
REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
SNAPSHOT = Path("models--facebook--wav2vec2-large-960h-lv60-self") / "snapshots" / REVISION

torch.set_num_threads(4)


def _emission(frames: int, peaks: dict[int, int], classes: int = 6) -> Any:
    """Log-probabilities with the blank (0) likely everywhere and ``peaks`` {frame: token} much likelier."""
    logits = torch.zeros(frames, classes)
    logits[:, 0] = 4.0
    for frame, token in peaks.items():
        logits[frame, token] = 12.0
    return torch.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------- frames, resampling, the guard


def _encoder_frames(samples: int) -> int:
    """The feature encoder's output length, convolution by convolution (kernels and strides of the model)."""
    length = samples
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2), strict=True):
        if length < kernel:
            return 0
        length = (length - kernel) // stride + 1
    return length


@pytest.mark.parametrize("samples", [0, 399, 400, 719, 720, 1039, 1040, 16_000, 104_960, 316_160, 316_161])
def test_ctc_frames_is_the_encoders_output_length_s11_2(samples: int) -> None:
    assert qa.ctc_frames(samples) == _encoder_frames(samples)


@pytest.mark.parametrize("rate", [16_000, 22_050, 24_000, 44_100, 48_000])
@pytest.mark.parametrize("samples", [1, 1_000, 48_001])
def test_resampled_length_is_what_to_mono_16k_returns_s11_2(rate: int, samples: int) -> None:
    audio = np.zeros((samples, 2), dtype=np.float32)
    assert qa.to_mono_16k(audio, rate).shape == (qa.resampled_length(samples, rate),)


def test_to_mono_16k_averages_channels_s11_2() -> None:
    left = np.full(1600, 0.5, dtype=np.float32)
    mono = qa.to_mono_16k(np.stack([left, -left / 5], axis=1), 16_000)
    assert mono.dtype == np.float32 and np.allclose(mono, 0.2)


def test_guard_needs_frames_for_tokens_plus_repeats_s11_2() -> None:
    qa.check_guard(5, [1, 2, 2, 3])
    with pytest.raises(qa.AlignmentFailure) as caught:
        qa.check_guard(4, [1, 2, 2, 3])
    assert caught.value.code == "ALIGNMENT_ERROR"
    assert caught.value.details == {"reason": "too_short", "frames": 4, "tokens": 4, "repeats": 1}
    with pytest.raises(qa.AlignmentFailure) as caught:
        qa.check_guard(100, [])
    assert caught.value.details["reason"] == "no_tokens"


# ---------------------------------------------------------------------- forced alignment


def test_forced_align_gives_one_span_per_token_s11_2() -> None:
    spans = qa.align_emission(torch, _emission(60, {10: 1, 20: 2, 30: 2, 40: 3}), [1, 2, 2, 3], 0)
    assert [(s["token_index"], s["start_frame"], s["end_frame"]) for s in spans] == [
        (0, 10, 11),
        (1, 20, 21),
        (2, 30, 31),
        (3, 40, 41),
    ]
    assert all(0.99 < s["score"] <= 1.0 for s in spans)


def test_forced_align_is_deterministic_s11_2() -> None:
    torch.manual_seed(7)
    emission = torch.log_softmax(torch.randn(80, 6), dim=-1)
    assert qa.align_emission(torch, emission, [1, 3, 3, 2, 5], 0) == qa.align_emission(
        torch, emission, [1, 3, 3, 2, 5], 0
    )


def test_too_few_frames_is_an_alignment_error_before_forced_align_s11_2() -> None:
    with pytest.raises(qa.AlignmentFailure) as caught:
        qa.align_emission(torch, _emission(4, {}), [1, 2, 2, 3], 0)
    assert caught.value.details == {"reason": "too_short", "frames": 4, "tokens": 4, "repeats": 1}


def test_forced_align_raising_is_an_alignment_error_s11_2() -> None:
    with pytest.raises(qa.AlignmentFailure) as caught:
        qa.align_emission(torch, _emission(20, {}), [1, 0, 2], 0)  # the blank as a target
    details = caught.value.details
    assert details["reason"] == "forced_align_raised"
    assert details["type"] == "ValueError"
    assert (details["frames"], details["tokens"], details["repeats"]) == (20, 3, 0)


# ---------------------------------------------------------------------- the wildcard (plan.md DC-11)


def test_wildcard_token_is_the_extra_column_s11_2_dc11() -> None:
    vocab = {"<pad>": 0, "|": 4, "A": 7}
    assert qa.token_ids(["A", "|", "*", "|", "A"], vocab, wildcard_id=32) == [7, 4, 32, 4, 7]


def test_wildcard_column_is_one_minus_the_blank_s11_2_dc11() -> None:
    emission = _emission(6, {2: 1, 3: 4})
    extended = qa.with_wildcard(torch, emission, 0)
    assert extended.shape == (6, 7)
    assert torch.equal(extended[:, :6], emission)
    expected = torch.log1p(-emission[:, 0].exp())
    assert torch.allclose(extended[:, 6], expected)
    silent = torch.log_softmax(torch.tensor([[100.0, 0.0, 0.0]]), dim=-1)
    assert float(qa.with_wildcard(torch, silent, 0)[0, 3]) == pytest.approx(
        math.log(qa.WILDCARD_FLOOR), abs=0.05
    )  # float32


def test_wildcard_absorbs_speech_outside_the_alphabet_s11_2_dc11() -> None:
    # A at frame 10, then speech the targets cannot spell (class 5) at frames 20-29, then B at frame 40
    logits = torch.zeros(60, 6)
    logits[:, 0] = 4.0
    logits[10, 1] = logits[40, 2] = 12.0
    logits[20:30, 5] = 12.0
    extended = qa.with_wildcard(torch, torch.log_softmax(logits, dim=-1), 0)
    spans = qa.align_emission(torch, extended, [1, 6, 2], 0)
    assert [(s["start_frame"], s["end_frame"]) for s in spans] == [(10, 11), (20, 30), (40, 41)]
    assert spans[1]["score"] > 0.99


# ---------------------------------------------------------------------- requests the aligner refuses


def test_token_outside_the_vocabulary_is_an_invalid_request_s11_2() -> None:
    vocab = {"<pad>": 0, "|": 4, "A": 7}
    assert qa.token_ids(["A", "|", "A"], vocab) == [7, 4, 7]
    for bad in (["A", "b"], ["<pad>"], ["<unk>"], ["AB"], ["A", "*"], ["**"]):
        with pytest.raises(qa.InvalidRequest) as caught:
            qa.token_ids(bad, vocab)
        assert caught.value.code == "INVALID_REQUEST" and caught.value.details["field"] == "tokens"


def test_unreadable_audio_is_unsupported_audio_s11_2(tmp_path: Path) -> None:
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a wav file")
    with pytest.raises(qa.UnreadableAudio) as caught:
        qa.read_audio(junk)
    assert caught.value.code == "UNSUPPORTED_AUDIO"
    nan = tmp_path / "nan.wav"
    sf.write(str(nan), np.array([0.0, np.nan, 0.0], dtype=np.float32), 16_000, subtype="FLOAT")
    with pytest.raises(qa.UnreadableAudio):
        qa.read_audio(nan)


def test_load_refuses_a_gpu_a_missing_snapshot_and_a_misnamed_one_s11_2(tmp_path: Path) -> None:
    aligner = qa.Wav2Vec2Aligner(torch)
    with pytest.raises(qa.InvalidRequest, match="CPU only"):
        aligner.load(REPO, REVISION, tmp_path, device="cuda:0")
    with pytest.raises(qa.BackendMissing):
        aligner.load(REPO, REVISION, tmp_path / REVISION)
    with pytest.raises(qa.InvalidRequest, match="not named by the revision"):
        aligner.load(REPO, REVISION, tmp_path)
    empty = tmp_path / REVISION
    empty.mkdir()
    with pytest.raises(qa.BackendMissing, match="vocabulary"):
        aligner.load(REPO, REVISION, empty)
    assert not aligner.loaded


def test_align_before_load_is_not_loaded_s11_2(tmp_path: Path) -> None:
    aligner = qa.Wav2Vec2Aligner(torch)
    with pytest.raises(qa.NotLoaded):
        aligner.align(np.zeros(16_000, dtype=np.float32), 16_000, ["A"])
    with pytest.raises(qa.NotLoaded):
        aligner.align_file(tmp_path / "x.wav", ["A"])


def test_every_error_carries_a_protocol_code_s11_2() -> None:
    kinds = (qa.AlignmentFailure, qa.InvalidRequest, qa.UnreadableAudio, qa.BackendMissing, qa.NotLoaded)
    assert {k.code for k in kinds} <= set(WORKER_ERROR_CODES)
    assert qa.AlignmentFailure.code == "ALIGNMENT_ERROR"


# ---------------------------------------------------------------------- with the pinned model (CPU)


@pytest.fixture(scope="module")
def aligner() -> qa.Wav2Vec2Aligner:
    root = os.environ.get("NARRATION_MODELS_ROOT")
    if not root:
        pytest.skip("NARRATION_MODELS_ROOT is not set: the pinned wav2vec2 snapshot is needed")
    snapshot = Path(root) / SNAPSHOT
    if not snapshot.is_dir():
        pytest.skip(f"the pinned snapshot is not installed under NARRATION_MODELS_ROOT: {SNAPSHOT.as_posix()}")
    model = qa.Wav2Vec2Aligner(torch)
    model.load(REPO, REVISION, snapshot)
    return model


@pytest.mark.model
def test_model_loads_offline_on_the_cpu_s11_2(aligner: qa.Wav2Vec2Aligner) -> None:
    assert aligner.loaded


@pytest.mark.model
def test_too_short_audio_raises_the_guard_not_a_crash_s11_2(aligner: qa.Wav2Vec2Aligner) -> None:
    tokens = list("WINDS|CAME|OVER|THE|RIDGE")  # invented: 25 tokens, no repeat
    with pytest.raises(qa.AlignmentFailure) as caught:
        aligner.align(np.zeros(4_800, dtype=np.float32), 48_000, tokens)  # 0.1 s
    assert caught.value.details == {"reason": "too_short", "frames": 4, "tokens": 25, "repeats": 0}


@pytest.mark.model
def test_silence_aligns_without_a_crash_and_scores_low_s11_2(aligner: qa.Wav2Vec2Aligner, tmp_path: Path) -> None:
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros(48_000, dtype=np.float32), 48_000, subtype="FLOAT")
    reply = aligner.align_file(wav, list("RAIN|CAME"))
    assert (reply["model"], reply["revision"], reply["device"], reply["frame_s"]) == (REPO, REVISION, "cpu", 0.02)
    assert reply["num_frames"] == qa.ctc_frames(16_000) == 49
    assert [s["token_index"] for s in reply["spans"]] == list(range(9))
    letters = [s["score"] for s, t in zip(reply["spans"], "RAIN|CAME", strict=True) if t != "|"]
    assert sum(letters) / len(letters) < 0.5


@pytest.mark.model
def test_wildcard_aligns_with_the_model_s11_2_dc11(aligner: qa.Wav2Vec2Aligner) -> None:
    reply = aligner.align(np.zeros(48_000, dtype=np.float32), 48_000, [*"RAIN", "|", "*", "|", *"CAME"])
    assert [s["token_index"] for s in reply["spans"]] == list(range(11))
    assert 0.0 <= reply["spans"][5]["score"] <= 1.0
