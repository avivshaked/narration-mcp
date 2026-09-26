"""The fake worker's outputs and planted faults (plan.md WP16), through the protocol, in real processes.

Every test starts its workers with ``WorkerProcess`` in a ``with`` block, so each process is killed and reaped
whatever happens, and every wait has a timeout.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from narration_worker.fake.audio import SAMPLE_RATE
from narration_worker.fake.faults import SPEC_ENV
from narration_worker.fake.handler import PROFILE_KEYS
from narration_worker.fake.registry import RECORD_SCHEMA, Registry, fake_dir
from narration_worker.fake.wav import read_wav
from narration_worker.testing.client import WorkerProcess, check_reply

TIMEOUT = 30.0
TEXT = "Far below the surface, where the light grows thin, small things live quiet lives."
VOICE = "sha256:" + "ab" * 32


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    (root / "scratch").mkdir(parents=True)
    return root


def _argv(store: Path) -> list[str]:
    return [sys.executable, "-m", "narration_worker", "--role", "fake", "--store", str(store), "--cpu-threads", "2"]


def _env(spec: dict[str, Any] | None, store: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != SPEC_ENV}
    if spec is not None:
        path = store.parent / "fake-spec.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        env[SPEC_ENV] = str(path)
    return env


def fake(store: Path, spec: dict[str, Any] | None = None) -> WorkerProcess:
    return WorkerProcess(_argv(store), env=_env(spec, store))


@pytest.fixture
def worker(store: Path) -> Iterator[WorkerProcess]:
    with fake(store) as process:
        _ok(process.request("load", device="cpu", timeout_s=TIMEOUT))
        yield process


def _ok(reply: dict[str, Any]) -> dict[str, Any]:
    assert reply["ok"] is True, reply
    return reply


def _error(reply: dict[str, Any], code: str) -> dict[str, Any]:
    assert reply["ok"] is False and reply["error"]["code"] == code, reply
    return reply["error"]


def _design(worker: WorkerProcess, store: Path, name: str, description: str = "A calm, low voice.") -> Path:
    clip = store / "scratch" / f"{name}.wav"
    _ok(
        worker.request(
            "design",
            description=description,
            design_text="Some of them thrive, and some of them simply disappear.",
            language="English",
            seed=2001,
            out_path=str(clip),
            timeout_s=TIMEOUT,
        )
    )
    return clip


def _voice(worker: WorkerProcess, store: Path, voice_hash: str = VOICE, name: str = "clip") -> Path:
    clip = _design(worker, store, name)
    _ok(
        worker.request(
            "prepare_voice",
            voice_hash=voice_hash,
            ref_wav=str(clip),
            ref_text="Some of them thrive, and some of them simply disappear.",
            x_vector_only_mode=False,
            timeout_s=TIMEOUT,
        )
    )
    return clip


def _say(worker: WorkerProcess, store: Path, name: str, text: str = TEXT, seed: int = 7) -> tuple[Path, dict[str, Any]]:
    out = store / "scratch" / f"{name}.wav"
    reply = worker.request(
        "synthesize",
        voice_hash=VOICE,
        engine_text=text,
        language="English",
        seed=seed,
        out_path=str(out),
        timeout_s=TIMEOUT,
    )
    return out, reply


def _transcribe(worker: WorkerProcess, wav: Path) -> dict[str, Any]:
    return worker.request(
        "transcribe", wav=str(wav), language="English", word_timestamps=True, long_form=True, timeout_s=TIMEOUT
    )


def _embed(worker: WorkerProcess, wav: Path) -> list[float]:
    return _ok(worker.request("embed", wav=str(wav), device="cpu", timeout_s=TIMEOUT))["embedding"]


def _cos(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------- outputs


def test_synthesize_writes_float32_mono_wav_whose_length_follows_the_text_appA(
    worker: WorkerProcess, store: Path
) -> None:
    _voice(worker, store)
    wav, reply = _say(worker, store, "long")
    _ok(reply)
    with wav.open("rb") as handle:
        header = handle.read(36)
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    assert int.from_bytes(header[20:22], "little") == 3  # IEEE float
    assert int.from_bytes(header[22:24], "little") == 1  # mono
    assert int.from_bytes(header[34:36], "little") == 32
    audio = read_wav(wav)
    assert audio.sample_rate == reply["sample_rate"] == SAMPLE_RATE
    assert len(audio.samples) == reply["samples"]
    assert reply["hit_token_cap"] is False
    assert reply["new_tokens"] == math.ceil(reply["samples"] * 12 / SAMPLE_RATE)
    assert 4.0 < audio.duration_s < 8.0  # 14 words at about narration pace
    _, short = _say(worker, store, "short", text="Far below.")
    assert _ok(short)["samples"] < reply["samples"] / 4


def test_the_same_request_gives_the_same_bytes_in_another_process_appA(tmp_path: Path) -> None:
    digests = []
    for run in ("a", "b"):
        store = tmp_path / run / "store"
        (store / "scratch").mkdir(parents=True)
        with fake(store) as worker:
            _ok(worker.request("load", device="cpu", timeout_s=TIMEOUT))
            clip = _voice(worker, store)
            wav, reply = _say(worker, store, f"take-{run}")
            _ok(reply)
            digests.append(
                (hashlib.sha256(clip.read_bytes()).hexdigest(), hashlib.sha256(wav.read_bytes()).hexdigest())
            )
            reply.pop("id")
            digests.append(json.dumps(reply, sort_keys=True))
    assert digests[0] == digests[2]
    assert digests[1] == digests[3]


GOLDEN_DESIGN_SHA256 = "faa8a531e385afce0a7af3f1a79458d600ac54c8f6959ebc43aac1016dd03746"


def test_the_fake_voice_is_pinned_to_the_byte_on_every_platform_appA(worker: WorkerProcess, store: Path) -> None:
    """Integer synthesis: this hash holds on Windows and Linux alike. Change it only on purpose."""
    clip = _design(worker, store, "golden")
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == GOLDEN_DESIGN_SHA256


def test_an_utterance_id_clash_steps_to_the_next_id_appA(worker: WorkerProcess, store: Path) -> None:
    """Utterance ids are 32 bits, so two utterances can share one: the second steps on and is still heard."""
    _voice(worker, store)
    _ok(_say(worker, store, "first")[1])
    records = fake_dir(store) / "utterances"
    mine = next(p for p in records.glob("*.json") if json.loads(p.read_text(encoding="utf-8"))["text"] == TEXT)
    foreign = json.loads(mine.read_text(encoding="utf-8"))
    foreign["text"] = "Another utterance entirely."
    mine.write_text(json.dumps(foreign), encoding="utf-8")  # another utterance now holds this one's id
    second, reply = _say(worker, store, "second")
    _ok(reply)
    stepped = records / f"{(int(mine.stem, 16) + 1) % 2**32:08x}.json"
    assert json.loads(stepped.read_text(encoding="utf-8"))["id"] == stepped.stem
    assert json.loads(mine.read_text(encoding="utf-8"))["text"] == "Another utterance entirely."
    assert _ok(_transcribe(worker, second))["text"] == TEXT
    third, _ = _say(worker, store, "third")
    assert third.read_bytes() == second.read_bytes()  # the same utterance finds its record at the stepped id


def test_utterance_ids_step_past_a_clash_and_wrap_around_appA(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    top = 2**32 - 1

    def record(text: str) -> dict[str, Any]:
        return {"schema": RECORD_SCHEMA, "text": text}

    assert registry.claim(top, record("a")) == top
    assert registry.claim(top, record("b")) == 0
    assert registry.claim(top, record("b")) == 0
    assert registry.claim(top, record("a")) == top
    third = record("c")
    assert registry.claim(top, third) == 1 and third["id"] == "00000001"
    assert [(registry.get(i) or {}).get("text") for i in (top, 0, 1)] == ["a", "b", "c"]
    assert not list(registry.root.glob(".*.tmp"))


def test_different_seeds_give_different_takes_s10_3(worker: WorkerProcess, store: Path) -> None:
    _voice(worker, store)
    a, _ = _say(worker, store, "seed1", seed=1)
    b, _ = _say(worker, store, "seed2", seed=2)
    assert a.read_bytes() != b.read_bytes()


def test_transcribe_hears_the_text_back_with_word_times_s11_1(worker: WorkerProcess, store: Path) -> None:
    _voice(worker, store)
    wav, _ = _say(worker, store, "take")
    reply = _ok(_transcribe(worker, wav))
    assert reply["text"] == TEXT
    words = reply["words"]
    assert [w["text"] for w in words] == TEXT.split()
    assert words[0]["start_s"] == pytest.approx(0.12, abs=0.002)
    for before, after in itertools.pairwise(words):
        assert before["start_s"] < before["end_s"] < after["start_s"]
    assert words[-1]["end_s"] < read_wav(wav).duration_s
    assert reply["revision"] and len(reply["revision"]) == 40


def test_a_clone_sounds_like_its_clip_and_not_like_another_voice_s11_1(worker: WorkerProcess, store: Path) -> None:
    clip = _voice(worker, store)
    take1, _ = _say(worker, store, "t1", seed=1)
    take2, _ = _say(worker, store, "t2", seed=2)
    other = _design(worker, store, "other", description="A bright, quick voice.")
    e_clip, e1, e2, e_other = (_embed(worker, p) for p in (clip, take1, take2, other))
    assert len(e1) == 512
    assert math.fsum(x * x for x in e1) == pytest.approx(1.0, abs=1e-9)
    assert 0.97 < _cos(e1, e_clip) < 1.0
    assert 0.97 < _cos(e1, e2) < 1.0
    assert abs(_cos(e1, e_other)) < 0.3
    assert _embed(worker, take1) == e1


def test_synthesize_needs_prepare_voice_appA(worker: WorkerProcess, store: Path) -> None:
    _, reply = _say(worker, store, "unprepared")
    _error(reply, "VOICE_NOT_PREPARED")


def test_unload_forgets_prepared_voices_appA(worker: WorkerProcess, store: Path) -> None:
    _voice(worker, store)
    _ok(worker.request("unload", timeout_s=TIMEOUT))
    _ok(worker.request("load", device="cpu", timeout_s=TIMEOUT))
    _, reply = _say(worker, store, "after-reload")
    _error(reply, "VOICE_NOT_PREPARED")


def test_paths_outside_the_store_are_refused_s17_2(worker: WorkerProcess, store: Path, tmp_path: Path) -> None:
    _voice(worker, store)
    reply = worker.request(
        "synthesize",
        voice_hash=VOICE,
        engine_text=TEXT,
        language="English",
        seed=1,
        out_path=str(tmp_path / "outside.wav"),
        timeout_s=TIMEOUT,
    )
    _error(reply, "INVALID_REQUEST")
    assert not (tmp_path / "outside.wav").exists()


def test_a_low_token_cap_cuts_the_take_s11_1(store: Path) -> None:
    with fake(store) as worker:
        _ok(
            worker.request(
                "load",
                device="cpu",
                settings={"non_streaming_mode": False, "generation": {"max_new_tokens": 30}},
                timeout_s=TIMEOUT,
            )
        )
        _voice(worker, store)
        wav, reply = _say(worker, store, "capped")
        _ok(reply)
        assert reply["hit_token_cap"] is True
        assert reply["new_tokens"] == 30
        assert reply["samples"] <= 30 * SAMPLE_RATE // 12
        heard = _ok(_transcribe(worker, wav))["text"].split()
        assert 0 < len(heard) < len(TEXT.split())
        assert heard == TEXT.split()[: len(heard)]


def test_unknown_audio_is_not_transcribed_unless_the_spec_names_it_s11_1(store: Path) -> None:
    silence = store / "scratch" / "foreign.wav"
    from array import array

    from narration_worker.fake.wav import write_float32_mono

    write_float32_mono(silence, array("f", [0.0] * 4800), 24_000)
    digest = hashlib.sha256(silence.read_bytes()).hexdigest()
    with fake(store) as worker:
        _ok(worker.request("load", device="cpu", timeout_s=TIMEOUT))
        _error(_transcribe(worker, silence), "UNSUPPORTED_AUDIO")
    with fake(store, {"transcripts": {digest: "a canned transcript"}}) as worker:
        _ok(worker.request("load", device="cpu", timeout_s=TIMEOUT))
        reply = _ok(_transcribe(worker, silence))
        assert reply["text"] == "a canned transcript"
        assert [w["start_s"] for w in reply["words"]] == [None, None, None]


def test_align_places_every_token_in_order_s11_2(worker: WorkerProcess, store: Path) -> None:
    _voice(worker, store)
    wav, _ = _say(worker, store, "take", text="Small things live.")
    tokens = [*"SMALL", "|", *"THINGS", "|", *"LIVE"]
    reply = _ok(worker.request("align", wav=str(wav), tokens=tokens, timeout_s=TIMEOUT))
    spans = reply["spans"]
    assert [s["token_index"] for s in spans] == list(range(len(tokens)))
    assert reply["frame_s"] == 0.02
    previous_end = 0
    for span in spans:
        assert span["start_frame"] >= previous_end and span["end_frame"] > span["start_frame"]
        previous_end = span["end_frame"]
    assert previous_end <= reply["num_frames"]
    words = _ok(_transcribe(worker, wav))["words"]
    things = spans[6:12]
    assert things[0]["start_frame"] * 0.02 == pytest.approx(words[1]["start_s"], abs=0.03)
    assert things[-1]["end_frame"] * 0.02 == pytest.approx(words[1]["end_s"], abs=0.03)
    # the two Ls of SMALL are equal neighbours: CTC needs a blank frame between them
    assert spans[4]["start_frame"] > spans[3]["end_frame"]


def test_align_guard_frames_below_tokens_plus_repeats_is_alignment_error_s11_2(
    worker: WorkerProcess, store: Path
) -> None:
    _voice(worker, store)
    wav, _ = _say(worker, store, "short", text="Hi.")
    tokens = ["A", "A"] * 200
    error = _error(worker.request("align", wav=str(wav), tokens=tokens, timeout_s=TIMEOUT), "ALIGNMENT_ERROR")
    assert error["details"]["reason"] == "too_short"
    assert error["details"]["tokens"] == 400 and error["details"]["repeats"] == 399


def test_f0_reports_the_pitch_within_the_range_asked_s3_6(worker: WorkerProcess, store: Path) -> None:
    _voice(worker, store)
    wav, _ = _say(worker, store, "take")
    reply = _ok(worker.request("f0", wav=str(wav), fmin_hz=50.0, fmax_hz=600.0, timeout_s=TIMEOUT))
    f0 = reply["f0_hz"]
    assert reply["hop_s"] == 0.01
    assert len(f0) == len(reply["voiced_probability"]) == math.ceil(read_wav(wav).duration_s / 0.01)
    voiced = [v for v in f0 if v is not None]
    assert voiced and all(120.0 <= v <= 502.5 for v in voiced)
    assert f0[0] is None  # the leading silence
    narrow = _ok(worker.request("f0", wav=str(wav), fmin_hz=50.0, fmax_hz=130.0, timeout_s=TIMEOUT))["f0_hz"]
    assert all(v is None or v <= 130.0 for v in narrow)


def test_profile_returns_every_measurement_and_two_pictures_s3_6(worker: WorkerProcess, store: Path) -> None:
    clip = _voice(worker, store)
    out_dir = store / "scratch" / "profile"
    reply = _ok(
        worker.request(
            "profile", wav=str(clip), out_dir=str(out_dir), transcript="Some of them thrive.", timeout_s=TIMEOUT
        )
    )
    assert tuple(reply["measurements"]) == PROFILE_KEYS
    assert reply["measurements"]["speaking_rate_wpm"] > 0
    assert set(reply["pictures"]) == {"spectrogram", "pitch"}
    for picture in reply["pictures"].values():
        assert Path(picture).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert Path(picture).parent == out_dir.resolve()


# ---------------------------------------------------------------------- planted faults


def _faulted(store: Path, *faults: dict[str, Any]) -> WorkerProcess:
    return fake(store, {"faults": list(faults)})


def _ready(worker: WorkerProcess, store: Path) -> None:
    _ok(worker.request("load", device="cpu", timeout_s=TIMEOUT))
    _voice(worker, store)


def test_planted_wrong_word_is_heard_in_the_transcript_wp40(store: Path) -> None:
    with _faulted(store, {"kind": "wrong_word", "word": 2, "replacement": "barber"}) as worker:
        _ready(worker, store)
        wav, _ = _say(worker, store, "take")
        words = [w["text"] for w in _ok(_transcribe(worker, wav))["words"]]
    expected = TEXT.split()
    expected[2] = "barber"
    assert words == expected


def test_planted_head_insertion_comes_before_the_first_word_wp40(store: Path) -> None:
    with _faulted(store, {"kind": "head_insertion", "words": ["and", "so"]}) as worker:
        _ready(worker, store)
        wav, _ = _say(worker, store, "take")
        words = _ok(_transcribe(worker, wav))["words"]
    assert [w["text"] for w in words[:3]] == ["and", "so", "Far"]
    assert words[1]["end_s"] < words[2]["start_s"]
    assert words[0]["probability"] < 0.9


def test_planted_end_insertion_comes_after_the_last_word_wp40(store: Path) -> None:
    with _faulted(store, {"kind": "end_insertion"}) as worker:
        _ready(worker, store)
        wav, _ = _say(worker, store, "take")
        words = [w["text"] for w in _ok(_transcribe(worker, wav))["words"]]
    assert words == [*TEXT.split(), "okay"]


def test_planted_token_cap_hit_wp40(store: Path) -> None:
    with _faulted(store, {"kind": "token_cap", "times": 1}) as worker:
        _ready(worker, store)
        _, first = _say(worker, store, "first")
        _, second = _say(worker, store, "second")
    assert _ok(first)["hit_token_cap"] is True
    assert _ok(second)["hit_token_cap"] is False
    assert first["new_tokens"] == second["new_tokens"] * 6 // 10


def test_planted_gpu_oom_fires_as_many_times_as_asked_s4(store: Path) -> None:
    with _faulted(store, {"kind": "gpu_oom", "times": 1}) as worker:
        _ready(worker, store)
        _, first = _say(worker, store, "first")
        _, retry = _say(worker, store, "retry")
    _error(first, "GPU_OOM")
    _ok(retry)


def test_planted_faults_match_on_text_and_seed_wp40(store: Path) -> None:
    fault = {"kind": "error", "code": "RENDER_FAILED", "when": {"text_contains": "quiet", "seed": 2}}
    with _faulted(store, fault) as worker:
        _ready(worker, store)
        _ok(_say(worker, store, "a", seed=1)[1])
        _error(_say(worker, store, "b", seed=2)[1], "RENDER_FAILED")
        _ok(_say(worker, store, "c", text="Nothing here.", seed=2)[1])


def test_planted_alignment_error_wp40(store: Path) -> None:
    with _faulted(store, {"kind": "alignment_error", "when": {"text_contains": "surface"}}) as worker:
        _ready(worker, store)
        wav, _ = _say(worker, store, "take")
        error = _error(
            worker.request("align", wav=str(wav), tokens=["F", "A", "R"], timeout_s=TIMEOUT), "ALIGNMENT_ERROR"
        )
    assert error["details"]["reason"] == "planted"


def test_planted_crash_exits_mid_request_and_does_not_repeat_after_a_restart_wp40(store: Path) -> None:
    with _faulted(store, {"kind": "crash", "exit_code": 7, "times": 1}) as worker:
        _ready(worker, store)
        worker.send(
            {
                "id": 99,
                "op": "synthesize",
                "voice_hash": VOICE,
                "engine_text": TEXT,
                "language": "English",
                "seed": 1,
                "out_path": str(store / "scratch" / "x.wav"),
            }
        )
        assert worker.wait(TIMEOUT) == 7
        assert "planted crash" in worker.stderr_text()
        assert all(json.loads(line)["id"] != 99 for line in worker.lines)
    with _faulted(store, {"kind": "crash", "exit_code": 7, "times": 1}) as restarted:
        _ready(restarted, store)
        _ok(_say(restarted, store, "after")[1])


def test_planted_hang_never_replies_wp40(store: Path) -> None:
    with _faulted(store, {"kind": "hang", "op": "transcribe"}) as worker:
        _ready(worker, store)
        wav, _ = _say(worker, store, "take")
        worker.send(
            {
                "id": 50,
                "op": "transcribe",
                "wav": str(wav),
                "language": "English",
                "word_timestamps": True,
                "long_form": True,
            }
        )
        with pytest.raises(AssertionError, match="no reply"):
            worker.receive(timeout_s=1.5)
        assert worker.proc.poll() is None
    assert worker.proc.poll() is not None  # stop() killed and reaped it


def test_planted_delay_postpones_the_reply_wp40(store: Path) -> None:
    with _faulted(store, {"kind": "delay", "seconds": 0.5, "op": "hello"}) as worker:
        import time

        started = time.monotonic()
        _ok(worker.request("hello", timeout_s=TIMEOUT))
        assert time.monotonic() - started >= 0.5


def test_planted_protocol_break_puts_a_foreign_line_on_stdout_appA(store: Path) -> None:
    with _faulted(store, {"kind": "protocol_break", "op": "hello"}) as worker:
        worker.send({"id": 1, "op": "hello"})
        with pytest.raises(AssertionError, match="not a protocol message"):
            worker.receive(TIMEOUT)


def test_stdout_noise_never_reaches_the_protocol_stream_appA(store: Path) -> None:
    with _faulted(store, {"kind": "stdout_noise", "op": "hello"}) as worker:
        _ok(worker.request("hello", timeout_s=TIMEOUT))
        _ok(worker.request("shutdown", timeout_s=TIMEOUT))
        assert worker.wait(TIMEOUT) == 0
        assert len(worker.lines) == 2
        for line in worker.lines:
            check_reply(line)
        stderr = worker.stderr_text()
    for source in ("sys.stdout", "sys.__stdout__", "file descriptor 1", "a child process"):
        assert f"planted noise on {source}" in stderr or f"planted noise from {source}" in stderr, source


def test_an_invalid_spec_stops_the_worker_at_start_with_exit_code_two(store: Path) -> None:
    with fake(store, {"faults": [{"kind": "explode"}]}) as worker:
        assert worker.wait(TIMEOUT) == 2
        assert "explode" in worker.stderr_text() or "kind" in worker.stderr_text()


def test_a_spec_file_edited_while_running_takes_effect_wp40(store: Path) -> None:
    with fake(store, {"faults": []}) as worker:
        _ready(worker, store)
        _ok(_say(worker, store, "before")[1])
        spec = Path(worker.env[SPEC_ENV]) if worker.env else None
        assert spec is not None
        spec.write_text(json.dumps({"faults": [{"kind": "gpu_oom", "times": 5}]}), encoding="utf-8")
        os.utime(spec, ns=(1, 1))
        _error(_say(worker, store, "after")[1], "GPU_OOM")


def test_the_spec_is_read_only_from_the_environment_appA(store: Path) -> None:
    """A request cannot carry faults: an unknown member is ignored and nothing is planted."""
    with fake(store) as worker:
        _ready(worker, store)
        reply = worker.request(
            "synthesize",
            voice_hash=VOICE,
            engine_text=TEXT,
            language="English",
            seed=1,
            out_path=str(store / "scratch" / "x.wav"),
            faults=[{"kind": "crash"}],
            timeout_s=TIMEOUT,
        )
        _ok(reply)


def test_the_fake_runs_where_the_parent_has_no_console(store: Path) -> None:
    """``python -m narration_worker`` with stdin from a closed pipe exits cleanly (end of input)."""
    proc = subprocess.run(_argv(store), input=b"", capture_output=True, timeout=TIMEOUT, check=False)
    assert proc.returncode == 0
    assert proc.stdout == b""


@pytest.mark.parametrize(("heard", "bursts"), [(3, 3), (2, 5), (5, 2), (7, 3), (1, 4)])
def test_a_given_transcript_is_spread_over_the_take_in_order_without_overlap_s11_1(heard: int, bursts: int) -> None:
    from narration_worker.fake.handler import _map_heard

    words = _map_heard([f"w{k}" for k in range(heard)], list(range(10, 10 + bursts)), set())
    assert [w["text"] for w in words] == [f"w{k}" for k in range(heard)]
    covered: list[tuple[int, int, int, int]] = [(w["burst"], w["last_burst"], w["part"], w["parts"]) for w in words]
    assert covered[0][0] == 10 and covered[-1][1] == 10 + bursts - 1
    for (first, last, part, parts), (next_first, _, next_part, _) in itertools.pairwise(covered):
        assert first <= last
        assert (last < next_first) or (last == next_first and part + 1 == next_part and parts > 1)
