"""Delivery on real Qwen3-TTS clone takes (design section 13; plan.md WP13's acceptance).

The takes are the bakeoff's six clone renders of the names probe (plan.md section 1.2), each a
concatenation of eight paragraph takes whose sample bounds its JSON sidecar lists. The bakeoff is local
evidence, found through ``NARRATION_BAKEOFF_ROOT``; without it these tests skip. Its files are only read:
each paragraph is post-processed in memory, and the one test of the file path deletes what it wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import soundfile

from narration.config import DeliveryConfig
from narration.contracts import codes
from narration.post import DeliveryPipeline, deliver
from narration.post.pcm import decode_wav, read_mono
from narration.post.resample import resampled_length

from .signals import reference_loudness, reference_true_peak

pytestmark = pytest.mark.evidence

TAKES = "outputs/qwen3-tts-1.7b-clone-*/r48_names_probe.wav"
PROFILE = DeliveryConfig()
LOUDNESS_TOLERANCE_LU = 0.1
"""EBU Tech 3341's tolerance for a loudness meter."""
TRUE_PEAK_TOLERANCE_DB = 0.1
"""How far the independent interpolator may read above the pinned 4x meter."""


@dataclass(frozen=True)
class Paragraph:
    render: str
    segment: str
    start: int
    end: int


@pytest.fixture(scope="module")
def renders() -> dict[Path, list[Paragraph]]:
    root = os.environ.get("NARRATION_BAKEOFF_ROOT")
    if not root:
        pytest.skip("NARRATION_BAKEOFF_ROOT is not set: the bakeoff's clone takes are local evidence")
    found: dict[Path, list[Paragraph]] = {}
    for wav in sorted(Path(root).glob(TAKES)):
        sidecar = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        found[wav] = [
            Paragraph(wav.parent.name, str(s["id"]), int(s["start_sample"]), int(s["end_sample"]))
            for s in sidecar["segments"]
        ]
    if not found:
        pytest.skip(f"no clone takes matching {TAKES} under NARRATION_BAKEOFF_ROOT")
    return found


def test_evidence_is_the_six_clone_takes_of_eight_paragraphs(renders: dict[Path, list[Paragraph]]) -> None:
    assert len(renders) == 6
    for wav, paragraphs in renders.items():
        info = soundfile.info(str(wav))
        assert (info.samplerate, info.channels) == (24000, 1)
        assert len(paragraphs) == 8
        assert all(0 <= p.start < p.end <= info.frames for p in paragraphs)


def test_evidence_delivery_meets_loudness_and_true_peak_s13(renders: dict[Path, list[Paragraph]]) -> None:
    problems: list[str] = []
    for wav, paragraphs in renders.items():
        audio, rate = read_mono(wav)
        for p in paragraphs:
            raw = audio[p.start : p.end]
            made = deliver(raw, rate, PROFILE)
            out = made.output
            delivery, delivery_rate = decode_wav(made.wav)
            where = f"{p.render}/{p.segment}"
            head, tail, pad = round(out.trim.head_s * rate), round(out.trim.tail_s * rate), round(0.08 * rate)
            kept = raw.shape[0] - max(head - pad, 0) - max(tail - pad, 0)
            if (delivery_rate, delivery.shape[0]) != (48000, out.samples) or out.samples != resampled_length(
                kept, rate, 48000
            ):
                problems.append(f"{where}: length {out.samples} breaks the formula")
            loud = out.loudness
            reference = reference_loudness(delivery)
            # pyloudnorm also counts a final partial block, which the whole-block reference leaves out: up to
            # 0.06 LU on these paragraphs, inside EBU Tech 3341's tolerance.
            if abs(reference - loud.measured_lufs) > LOUDNESS_TOLERANCE_LU:
                problems.append(f"{where}: record {loud.measured_lufs} vs reference {reference:.4f} LUFS")
            if loud.ceiling_applied != any(f.code == codes.LOUDNESS_UNDER_TARGET for f in out.flags):
                problems.append(f"{where}: ceiling_applied and LOUDNESS_UNDER_TARGET disagree")
            if not loud.ceiling_applied and abs(reference - PROFILE.target_lufs) > LOUDNESS_TOLERANCE_LU:
                problems.append(f"{where}: {reference:.3f} LUFS is off the {PROFILE.target_lufs} target")
            if loud.ceiling_applied and reference > PROFILE.target_lufs:
                problems.append(f"{where}: the ceiling applied yet {reference:.3f} LUFS is over the target")
            if loud.true_peak_dbtp > PROFILE.true_peak_dbtp:
                problems.append(f"{where}: true peak {loud.true_peak_dbtp} dBTP is over the ceiling")
            independent = reference_true_peak(delivery)
            if independent > PROFILE.true_peak_dbtp + TRUE_PEAK_TOLERANCE_DB:
                problems.append(f"{where}: independent true peak {independent:.3f} dBTP is over the ceiling")
    assert problems == []


def test_evidence_delivery_is_deterministic_s13(renders: dict[Path, list[Paragraph]]) -> None:
    for wav, paragraphs in renders.items():
        audio, rate = read_mono(wav)
        p = paragraphs[0]
        assert deliver(audio[p.start : p.end], rate, PROFILE).wav == deliver(audio[p.start : p.end], rate, PROFILE).wav


def test_evidence_delivery_is_the_same_in_another_process_s13(renders: dict[Path, list[Paragraph]]) -> None:
    wav, paragraphs = next(iter(renders.items()))
    p = paragraphs[1]
    audio, rate = read_mono(wav)
    here = hashlib.sha256(deliver(audio[p.start : p.end], rate, PROFILE).wav).hexdigest()
    code = (
        "import hashlib, sys\n"
        "from pathlib import Path\n"
        "from narration.config import DeliveryConfig\n"
        "from narration.post import deliver\n"
        "from narration.post.pcm import read_mono\n"
        "audio, rate = read_mono(Path(sys.argv[1]))\n"
        "raw = audio[int(sys.argv[2]) : int(sys.argv[3])]\n"
        "print(hashlib.sha256(deliver(raw, rate, DeliveryConfig()).wav).hexdigest())\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(wav), str(p.start), str(p.end)],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert done.stdout.strip() == here


def test_evidence_process_reads_a_render_by_path_s13(renders: dict[Path, list[Paragraph]], tmp_path: Path) -> None:
    wav = next(iter(renders))
    out = tmp_path / "delivery.wav"
    try:
        result = DeliveryPipeline().process(wav, out, PROFILE)
        info = soundfile.info(str(out))
        assert (info.samplerate, info.subtype, info.frames) == (48000, "PCM_24", result.samples)
        assert result.loudness.true_peak_dbtp <= PROFILE.true_peak_dbtp
    finally:
        out.unlink(missing_ok=True)
