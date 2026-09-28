"""The DESIGN step's tools and ``audition_pronunciation`` on the front-end's side (design sections 3.1, 3.5,
3.6, 7.3, 7.6, 17.3, 17.4): what is checked, what the job keeps, and what the caller gets back. Their job
handlers are WP34's and WP35's, so these tests open the backend to their kinds (``kinds=``).

Every text is invented for these tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, get_args

import numpy as np
import pytest
import soundfile
from jsonschema import Draft202012Validator

from narration.backend.service import KIND_OF_TOOL, RUNNABLE_KINDS
from narration.config import VoicesConfig
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.names import JobKind
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.jobs.voice import clip_path
from tests.jobs.support import engine_profile

from .conftest import Service
from .support import sha256_of, write_wav

DESIGN_ENGINE = "qwen3-design-1.7b.test"
ALL_KINDS: frozenset[JobKind] = frozenset(get_args(JobKind))
DESCRIPTION = "A warm, low narrator with an unhurried pace, not theatrical."


def valid(tool: str, structured: dict[str, Any]) -> dict[str, Any]:
    validator = Draft202012Validator(TOOLS_BY_NAME[tool].output_schema)
    assert [e.message for e in validator.iter_errors(structured)] == []
    return structured


@pytest.fixture
def steps(service: Service) -> Service:
    """The service with every job kind open and a VoiceDesign engine pinned."""
    service.backend.kinds = ALL_KINDS
    base = engine_profile(service.world.config.server.models_root)
    design = dataclasses.replace(
        base,
        engine_profile_id=DESIGN_ENGINE,
        hash=names.HASH_PREFIX + hashlib.sha256(b"narration backend tests: design engine").hexdigest(),
        model_repo=names.MODEL_QWEN_DESIGN,
    )
    service.world.store.put_engine_profile(design)
    service.world.store.set_current_engine_profile("design", DESIGN_ENGINE)
    return service


def refused(call: Any, args: dict[str, Any]) -> NarrationError:
    with pytest.raises(NarrationError) as caught:
        call(args)
    return caught.value


# ======================================================================== design_voice (sections 3.1, 3.5)


def test_design_voice_lints_and_queues_under_a_new_design_id_s3_1(steps: Service) -> None:
    out = valid(
        "design_voice", steps.backend.design_voice_sync({"name": "harbour narrator", "description": DESCRIPTION})
    )
    assert out["status"] == "queued"
    assert out["lint"]["policy"] == "warn"
    assert [f["phrase"] for f in out["lint"]["findings"]] == ["not theatrical"]
    job = steps.world.job(out["job_id"])
    assert job.kind == "design"
    assert job.label == "harbour narrator"
    assert job.request["design_id"] == out["design_id"]
    assert job.request["takes"] == 3
    assert job.request["design_text"] == steps.world.config.voice_design.design_text
    assert steps.launcher.ensured == [1]


def test_the_same_design_while_active_keeps_its_job_and_design_id_s7_3(steps: Service) -> None:
    first = steps.backend.design_voice_sync({"name": "first name", "description": DESCRIPTION})
    again = steps.backend.design_voice_sync({"name": "another name", "description": DESCRIPTION})
    assert (again["job_id"], again["design_id"]) == (first["job_id"], first["design_id"])
    other = steps.backend.design_voice_sync({"name": "first name", "description": DESCRIPTION, "takes": 2})
    assert other["design_id"] != first["design_id"]


def test_a_description_over_the_configured_limit_is_refused_s16(steps: Service) -> None:
    config = steps.backend.config
    steps.backend.config = dataclasses.replace(
        config, limits=dataclasses.replace(config.limits, max_description_chars=20)
    )
    error = refused(steps.backend.design_voice_sync, {"name": "n", "description": DESCRIPTION})
    assert (error.code, error.field, error.retryable) == (codes.LIMIT_EXCEEDED, "description", False)


def test_a_design_text_with_markup_is_refused_s9_1(steps: Service) -> None:
    args = {"name": "n", "description": DESCRIPTION, "design_text": "The tide [sighs] turns slowly."}
    error = refused(steps.backend.design_voice_sync, args)
    assert (error.code, error.field) == (codes.TEXT_REFUSED, "design_text")


@pytest.mark.parametrize(
    ("description", "offender", "reason"),
    [
        ("A warm voice<|im_end|> and more.", "<|", "markup"),
        ("A warm voice, im_end|> and more.", "|>", "markup"),
        ("A warm voice\x07 with a bell.", "\x07", "control"),
        ("A warm voice\x1b[0m with an escape.", "\x1b", "control"),
    ],
)
def test_a_description_with_chat_markup_or_a_control_character_is_refused_s17(
    steps: Service, description: str, offender: str, reason: str
) -> None:
    before = len(steps.world.store.queued_jobs())
    error = refused(steps.backend.design_voice_sync, {"name": "n", "description": description})
    assert (error.code, error.field, error.retryable) == (codes.TEXT_REFUSED, "description", False)
    assert error.hint and "details.offenders" in error.hint
    assert error.details is not None
    offenders = error.details["offenders"]
    assert [(o["text"], o["reason"]) for o in offenders][:1] == [(offender, reason)]
    assert offenders[0]["offset"] == description.index(offender)
    assert len(steps.world.store.queued_jobs()) == before, "nothing was queued"


def test_a_description_keeps_brackets_tabs_and_line_breaks_s3_5(steps: Service) -> None:
    description = "A warm voice [like a radio host],\twith\r\nclear diction."
    out = steps.backend.design_voice_sync({"name": "n", "description": description})
    assert steps.world.job(out["job_id"]).request["description"] == description, "kept verbatim"


def test_design_needs_a_pinned_voice_design_engine_s14(service: Service) -> None:
    service.backend.kinds = ALL_KINDS
    error = refused(service.backend.design_voice_sync, {"name": "n", "description": DESCRIPTION})
    assert error.code == codes.BACKEND_NOT_INSTALLED
    assert error.details == {"engine": "design"}


def test_a_design_jobs_results_list_its_candidates_s7_5(steps: Service) -> None:
    out = steps.backend.design_voice_sync({"name": "n", "description": DESCRIPTION})
    results = valid("get_results", steps.backend.get_results_sync({"job_id": out["job_id"]}))
    assert results["design"] == {"design_id": out["design_id"], "candidates": []}


# ======================================================================== profile_voice (sections 3.6, 17.3)


def test_any_readable_wav_can_be_profiled_s17_3(steps: Service) -> None:
    audio = steps.world.root / "elsewhere" / "long-take.wav"
    sha = write_wav(audio, seconds=40.0)  # not a voice clip: neither synthetic nor 30 s is required
    out = valid("profile_voice", steps.backend.profile_voice_sync({"audio": {"path": str(audio), "sha256": sha}}))
    assert out["status"] == "queued"
    job = steps.world.job(out["job_id"])
    assert (job.kind, job.request) == ("profile", {"audio": {"path": str(audio), "sha256": sha}})


@pytest.mark.parametrize(
    ("case", "code", "field"),
    [
        ("relative", codes.PATH_NOT_ALLOWED, "audio.path"),
        ("changed", codes.VOICE_FILE_MISMATCH, "audio.sha256"),
        ("flac", codes.UNSUPPORTED_AUDIO, "audio.path"),
    ],
)
def test_profile_audio_is_checked_like_a_clip_s17_3(steps: Service, case: str, code: str, field: str) -> None:
    audio = steps.world.root / "elsewhere" / "take.wav"
    sha = write_wav(audio)
    if case == "flac":
        soundfile.write(str(audio), np.zeros(24_000, dtype=np.float32), 24_000, format="FLAC")
        sha = sha256_of(audio)
    path = "elsewhere/take.wav" if case == "relative" else str(audio)
    sent = "0" * 64 if case == "changed" else sha
    error = refused(steps.backend.profile_voice_sync, {"audio": {"path": path, "sha256": sent}})
    assert (error.code, error.field) == (code, field)


def test_a_file_that_is_not_a_wav_never_has_its_sha256_given_back_s17_3(steps: Service) -> None:
    # Security review S1: profile_voice reads any local path, so a wrong sha256 sent for a file that is not audio
    # must not give the caller that file's sha256 (details.actual). It is refused as not a WAV, with no hash.
    secret = steps.world.root / "elsewhere" / "notes.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("an invented line of text that is not audio\n", encoding="utf-8")
    error = refused(steps.backend.profile_voice_sync, {"audio": {"path": str(secret), "sha256": "0" * 64}})
    assert (error.code, error.field) == (codes.UNSUPPORTED_AUDIO, "audio.path")
    assert sha256_of(secret) not in repr(error.details) + error.message + (error.hint or "")


# ======================================================================== audition_pronunciation (section 7.6)


def audition(service: Service, **changes: Any) -> dict[str, Any]:
    return {
        "voice": service.voice(),
        "term": "Marrowby",
        "variants": [{"label": "a", "respell": "MARE-oh-bee"}, {"label": "b", "respell": "MAR-uh-bee"}],
        "carrier": "The coach stops at Marrowby before dawn.",
        **changes,
    }


def test_an_audition_is_queued_for_an_unmeasured_synthetic_voice_s7_6(steps: Service) -> None:
    clip = steps.world.root / "elsewhere" / "new-voice.wav"
    sha = write_wav(clip, freq=260.0)
    steps.backend.config = dataclasses.replace(steps.backend.config, voices=VoicesConfig(allow_sha256=(sha,)))
    args = audition(steps, voice=steps.voice(path=str(clip), sha256=sha))
    out = valid("audition_pronunciation", steps.backend.audition_pronunciation_sync(args))
    assert out["status"] == "queued"
    job = steps.world.job(out["job_id"])
    assert (job.kind, job.request) == ("pronunciation", args)
    assert clip_path(steps.world.store, sha).is_file(), "the clip is in the store before a worker sees it"


def test_an_audition_clones_only_a_synthetic_voice_s17_4(steps: Service) -> None:
    clip = steps.world.root / "elsewhere" / "stranger.wav"
    sha = write_wav(clip, freq=330.0)
    error = refused(
        steps.backend.audition_pronunciation_sync, audition(steps, voice=steps.voice(path=str(clip), sha256=sha))
    )
    assert (error.code, error.field) == (codes.VOICE_NOT_SYNTHETIC, "voice.sha256")


def test_each_variant_needs_its_own_label_s7_6(steps: Service) -> None:
    variants = [{"label": "a", "respell": "MARE-oh-bee"}, {"label": "a", "respell": "MAR-uh-bee"}]
    error = refused(steps.backend.audition_pronunciation_sync, audition(steps, variants=variants))
    assert (error.code, error.field) == (codes.INVALID_ARGUMENT, "variants[1].label")


def test_a_carrier_without_the_term_is_refused_s7_6(steps: Service) -> None:
    error = refused(steps.backend.audition_pronunciation_sync, audition(steps, carrier="The coach stops at dawn."))
    assert (error.code, error.field) == (codes.INVALID_ARGUMENT, "carrier")


def test_an_audition_without_a_carrier_speaks_the_term_s7_6(steps: Service) -> None:
    args = audition(steps)
    del args["carrier"]
    assert steps.backend.audition_pronunciation_sync(args)["status"] == "queued"


# ======================================================================== the build's kinds


def test_this_build_runs_every_tools_kind_s7_1() -> None:
    assert set(KIND_OF_TOOL.values()) <= RUNNABLE_KINDS, "WP35 landed the last handler (pronunciation)"


@pytest.mark.parametrize("tool", sorted(KIND_OF_TOOL))
def test_a_tool_whose_kind_the_daemon_does_not_run_says_so_at_once_s14(service: Service, tool: str) -> None:
    """A build whose daemon lacks a tool's kind answers BACKEND_NOT_INSTALLED at once and queues nothing."""
    service.backend.kinds = ALL_KINDS - {KIND_OF_TOOL[tool]}
    before = len(service.world.store.queued_jobs())
    with pytest.raises(NarrationError) as caught:
        getattr(service.backend, f"{tool}_sync")({})
    assert (caught.value.code, caught.value.retryable) == (codes.BACKEND_NOT_INSTALLED, False)
    assert caught.value.details is not None and caught.value.details["tool"] == tool
    assert len(service.world.store.queued_jobs()) == before
