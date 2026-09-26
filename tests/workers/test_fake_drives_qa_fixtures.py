"""The fake worker can stage every QA fault fixture (``material/fixtures/qa-faults-v1``; plan.md WP18, WP40).

Each spec gives the text sent, the text the take says (``render``) and what the ASR writes (``asr``). With
a ``say`` fault (and ``token_cap`` where the spec has one), the fake renders the take and ``transcribe``
returns the spec's ASR text, with word times inside the take, so the QA logic can be tested on them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .conftest import ClientFactory

SPECS = Path(__file__).resolve().parents[2] / "material" / "fixtures" / "qa-faults-v1" / "specs.json"
TIMEOUT = 30.0
VOICE = "sha256:" + "12" * 32


def _specs() -> list[dict[str, Any]]:
    if not SPECS.is_file():
        return []
    return json.loads(SPECS.read_text(encoding="utf-8"))["specs"]


@pytest.mark.parametrize("spec", _specs(), ids=lambda spec: spec["name"])
def test_the_fake_stages_the_qa_fault_fixture_s11_1(
    make_client: ClientFactory, store: Path, spec: dict[str, Any]
) -> None:
    render, asr = spec["render"], spec["asr"]
    faults: list[dict[str, Any]] = [{"kind": "say", "text": render["text"], "heard": asr["text"]}]
    if "max_new_tokens" in render:
        faults.append({"kind": "token_cap", "max_new_tokens": render["max_new_tokens"]})
    client = make_client({"faults": faults})
    client.start()
    client.request("load", {"device": "cpu"}, timeout_s=TIMEOUT)
    clip = store / "scratch" / "clip.wav"
    client.request(
        "design",
        {
            "description": "A calm voice.",
            "design_text": "Far below the surface.",
            "language": "English",
            "seed": 1,
            "out_path": str(clip),
        },
        timeout_s=TIMEOUT,
    )
    client.request(
        "prepare_voice",
        {"voice_hash": VOICE, "ref_wav": str(clip), "ref_text": "Far below the surface.", "x_vector_only_mode": False},
        timeout_s=TIMEOUT,
    )
    take = store / "scratch" / "take.wav"
    sent = " ".join(cue["text"] for cue in spec["segment"]["cues"])
    audio = client.request(
        "synthesize",
        {"voice_hash": VOICE, "engine_text": sent, "language": "English", "seed": 1, "out_path": str(take)},
        timeout_s=TIMEOUT,
    )
    if "max_new_tokens" in render:
        assert audio["hit_token_cap"] is True
        assert audio["new_tokens"] == render["max_new_tokens"]
    heard = client.request(
        "transcribe",
        {"wav": str(take), "language": "English", "word_timestamps": True, "long_form": True},
        timeout_s=TIMEOUT,
    )
    assert heard["text"] == " ".join(asr["text"].split())
    duration = audio["samples"] / audio["sample_rate"]
    previous = 0.0
    for word in heard["words"]:
        assert previous <= word["start_s"] < word["end_s"] <= duration
        previous = word["end_s"]
