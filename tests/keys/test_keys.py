"""Keys, ids and seeds (design sections 10.2, 10.3; plan.md WP12).

The golden values below are **forever once released**: a changed byte changes every take id a caller has
recorded. Each one was derived independently of ``narration.keys``: the canonical bytes are written out by
hand here, and the pinned hex is the sha256 of those bytes (computed with ``hashlib`` alone). A test then
checks that the implementation produces the same bytes and the same key.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import unicodedata
from typing import Any

import pytest

from narration import keys
from narration.config import DeliveryConfig, MeasurementConfig
from narration.contracts import names
from narration.contracts.interfaces import AnalysisKeyInputs, KeyBuilder
from narration.contracts.models import DeliveryTools
from narration.keys import ulid
from narration.keys.ulid import is_ulid

# ---------------------------------------------------------------- the golden inputs
CLIP = "0123456789abcdef" * 4
ENGINE_PROFILE_HASH = "sha256:" + "9e21" * 16
ENGINE_TEXT = "Before dawn, the reef belongs to the Oss-a-veen shrimp."
SPOKEN = (
    "Before dawn, the reef belongs to the Ossavine shrimp. By sunrise, some three thousand two hundred of them "
    "are back in the rock."
)
CORPUS_VERSION = "narration-en.v1@sha256:" + "c07d" * 16
TOOLS = DeliveryTools(resampler="soxr 0.5.0", loudness_meter="pyloudnorm 0.1.1", post="narration.post/1")
# The golden settings are literals, not today's defaults: a golden value pins the key function, and must not
# move when a default changes (the delivery target moved from -16 to -20 LUFS in design revision 5.3).
DELIVERY_PROFILE = DeliveryConfig(
    sample_rate=48000,
    subtype="PCM_24",
    target_lufs=-16.0,
    true_peak_dbtp=-1.0,
    trim_rel_db=-40.0,
    trim_floor_dbfs=-70.0,
    trim_pad_s=0.08,
    fade_s=0.01,
)
MEASUREMENT_SETTINGS = MeasurementConfig(
    corpus="narration-en.v1",
    seeds=3,
    length_ladder_spoken_chars=(80, 150, 250, 300, 350, 400, 450, 500, 560),
    trend_band_max_chars=300,
    pace_tol_min=0.10,
    sim_warn_margin=0.01,
    sim_fail_floor=0.90,
)
ASR = "openai/whisper-large-v3@" + "1" * 40
SV = "microsoft/wavlm-base-plus-sv@" + "2" * 40

# ---------------------------------------------------------------- hand-built canonical bytes (RFC 8785)
VOICE_BYTES = (
    '{"clip_sha256":"' + CLIP + '","language":"English","model":"Qwen/Qwen3-TTS-12Hz-1.7B-Base",'
    '"schema":"narration.voice/v2","transcript":"Café by the reef.","x_vector_only_mode":false}'
).encode("utf-8")
VOICE_HASH = "sha256:cb5aaced1254ebd0d70fb5243745e240955603ff6392c32607eb0fc9acc0fb9c"

RENDER_BYTES = (
    '{"engine_profile_hash":"' + ENGINE_PROFILE_HASH + '","engine_text":"' + ENGINE_TEXT + '",'
    '"schema":"narration.render/v1","seed":1834112093,"voice_hash":"' + VOICE_HASH + '"}'
).encode("utf-8")
RENDER_KEY = "sha256:0ed0f147896bf2cca87ce7c629e305dc62d58c1dba177bcc37f3d08608ebd544"

DELIVERY_BYTES = (
    '{"delivery_profile":{"fade_s":0.01,"sample_rate":48000,"subtype":"PCM_24","target_lufs":-16,'
    '"trim_floor_dbfs":-70,"trim_pad_s":0.08,"trim_rel_db":-40,"true_peak_dbtp":-1},"raw_sha256":"' + "ab" * 32 + '",'
    '"schema":"narration.delivery-key/v1","stretch":null,"tools":{"loudness_meter":"pyloudnorm 0.1.1",'
    '"post":"narration.post/1","resampler":"soxr 0.5.0"}}'
).encode("utf-8")
DELIVERY_KEY = "sha256:0f057208e4a6f5bb823598f40d9d013cb88966514ffd0fdf0637c48cd9492c6d"

MEASUREMENT_BYTES = (
    '{"corpus_version":"' + CORPUS_VERSION + '","engine_profile_hash":"' + ENGINE_PROFILE_HASH + '",'
    '"ladder_settings":{"length_ladder_spoken_chars":[80,150,250,300,350,400,450,500,560],"pace_tol_min":0.1,'
    '"seeds":3,"sim_fail_floor":0.9,"sim_warn_margin":0.01,"trend_band_max_chars":300},'
    '"pace_method":"narration.pace/articulation-cps@1",'
    '"schema":"narration.measurement-key/v2","voice_hash":"' + VOICE_HASH + '"}'
).encode("utf-8")
MEASUREMENT_KEY = "sha256:56f527d352b6b17650c12aba0260569e94bd76a5ea60ee75a89eaec9c0e5753a"
"""``narration.measurement-key/v2`` (WP47: the key names the pace method). ``/v1``, without ``pace_method``,
was ``sha256:c2fb0d03…``, which the analysis golden below still takes as its input."""
ANALYSIS_MEASUREMENT_KEY = "sha256:c2fb0d03432c4234eac019f7f5ff7a71314041231bedeffafdccb8438f29cb2d"
"""The analysis golden's ``measurement_key`` input: any key will do, and this one keeps the golden fixed."""

ANALYSIS_BYTES = (
    '{"aligner_method_id":"ctc-forced-align+silence-snap","asr_model":"' + ASR + '",'
    '"cue_spans":[[0,53],[54,127]],"delivery_sha256":"' + "cd" * 32 + '","exact_spans":[[1,3,7]],'
    '"hints_qa":[{"align_as":null,"asr_aliases":["Osavine","Ossa vine"],"term":"Ossavine"}],'
    '"measurement_key":"' + ANALYSIS_MEASUREMENT_KEY + '","number_reader":"whisper-english-normalizer+nought@1",'
    '"qa_profile":"default.v3","schema":"narration.analysis-key/v1","spoken_text":"' + SPOKEN + '",'
    '"sv_model":"' + SV + '","text_checks_version":"text-1.1.0"}'
).encode("utf-8")
ANALYSIS_KEY = "sha256:f7fb5269b02911888ea1e767e7426c864a7751f76c0c4332fc8bf5eff4a49f0b"

# seed(VOICE_HASH, ENGINE_TEXT, attempt): the first four digest bytes 12371cc3 / 1f574721 / 25c7d02a
SEEDS = {0: 305601731, 1: 525813537, 7: 633851946}


def voice_kwargs(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": names.MODEL_QWEN_BASE,
        "clip_sha256": CLIP,
        "transcript": "Café by the reef.",  # decomposed: the key puts it in NFC
        "language": names.LANGUAGE,
        "x_vector_only_mode": False,
    }
    return base | changes


def render_kwargs(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "engine_profile_hash": ENGINE_PROFILE_HASH,
        "voice_hash": VOICE_HASH,
        "engine_text": ENGINE_TEXT,
        "seed": 1834112093,
    }
    return base | changes


def measurement_kwargs(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "voice_hash": VOICE_HASH,
        "engine_profile_hash": ENGINE_PROFILE_HASH,
        "corpus_version": CORPUS_VERSION,
        "settings": MEASUREMENT_SETTINGS,
    }
    return base | changes


def analysis_inputs(**changes: Any) -> AnalysisKeyInputs:
    base = AnalysisKeyInputs(
        delivery_sha256="cd" * 32,
        spoken_text=SPOKEN,
        cue_spans=((0, 53), (54, 127)),
        exact_spans=((1, 3, 7),),
        hints_qa=(("Ossavine", ("Ossa vine", "Osavine"), None),),
        # Literals, not the service's current versions: a golden value pins the key function, and must not
        # move when names.NUMBER_READER or another version is bumped.
        qa_profile="default.v3",
        text_checks_version="text-1.1.0",
        number_reader="whisper-english-normalizer+nought@1",
        asr_model=ASR,
        sv_model=SV,
        aligner_method_id="ctc-forced-align+silence-snap",
        measurement_key=ANALYSIS_MEASUREMENT_KEY,
    )
    return dataclasses.replace(base, **changes)


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- RFC 8785


def test_canonical_json_matches_the_rfc_8785_example_s10_2() -> None:
    # RFC 8785 section 3.2.4 (numbers, string escapes, literals).
    value = {
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 0.000000000000000000000000001],
        "string": '€$\u000f\nA\'B"\\\\"/',
        "literals": [None, True, False],
    }
    # The RFC's output, whose string member reads: "€$\u000f\nA'B\"\\\\\"/"
    expected = (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )
    assert keys.canonical_json(value) == expected.encode("utf-8")


def test_canonical_json_sorts_members_by_utf16_code_units_s10_2() -> None:
    # RFC 8785 section 3.2.3: the sorting example, including a character outside the BMP.
    value = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\u0080": "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }
    pairs = json.loads(keys.canonical_json(value), object_pairs_hook=list)
    assert [v for _, v in pairs] == [
        "Carriage Return",
        "One",
        "Control",
        "Latin Small Letter O With Diaeresis",
        "Euro Sign",
        "Emoji: Grinning Face",
        "Hebrew Letter Dalet With Dagesh",
    ]


@pytest.mark.parametrize(
    "value",
    [
        {"a": math.nan},
        {"a": math.inf},
        {"a": 2**53},
        {1: "not a string key"},
        DeliveryTools(resampler="r", loudness_meter="m", post="p"),
        {"a": object()},
    ],
)
def test_canonical_json_refuses_what_json_cannot_say_exactly_s10_2(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        keys.canonical_json(value)


# ---------------------------------------------------------------- golden keys


def test_voice_hash_golden_s10_2() -> None:
    obj = keys.voice_hash_object(**voice_kwargs())
    assert keys.canonical_json(obj) == VOICE_BYTES
    assert sha(VOICE_BYTES) == VOICE_HASH
    assert keys.voice_hash(**voice_kwargs()) == VOICE_HASH


def test_render_key_golden_s10_2() -> None:
    assert keys.canonical_json(keys.render_key_object(**render_kwargs())) == RENDER_BYTES
    assert sha(RENDER_BYTES) == RENDER_KEY
    assert keys.render_key(**render_kwargs()) == RENDER_KEY


def test_delivery_key_golden_s10_2() -> None:
    obj = keys.delivery_key_object(raw_sha256="ab" * 32, profile=DELIVERY_PROFILE, tools=TOOLS)
    assert keys.canonical_json(obj) == DELIVERY_BYTES
    assert sha(DELIVERY_BYTES) == DELIVERY_KEY
    assert keys.delivery_key(raw_sha256="ab" * 32, profile=DELIVERY_PROFILE, tools=TOOLS) == DELIVERY_KEY


def test_measurement_key_golden_s10_2() -> None:
    assert keys.canonical_json(keys.measurement_key_object(**measurement_kwargs())) == MEASUREMENT_BYTES
    assert sha(MEASUREMENT_BYTES) == MEASUREMENT_KEY
    assert keys.measurement_key(**measurement_kwargs()) == MEASUREMENT_KEY


def test_measurement_key_names_the_pace_method_s10_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """A measurement made under another pace rule is another measurement (WP47): its key differs, so
    ``measure_voice`` does not answer with it."""
    assert keys.measurement_key_object(**measurement_kwargs())["pace_method"] == names.PACE_METHOD
    monkeypatch.setattr(names, "PACE_METHOD", "narration.pace/articulation-cps@2")
    assert _differs(keys.measurement_key(**measurement_kwargs()), MEASUREMENT_KEY)


def test_analysis_key_golden_s10_2() -> None:
    assert keys.canonical_json(keys.analysis_key_object(analysis_inputs())) == ANALYSIS_BYTES
    assert sha(ANALYSIS_BYTES) == ANALYSIS_KEY
    assert keys.analysis_key(analysis_inputs()) == ANALYSIS_KEY


def test_seed_golden_s10_3() -> None:
    for attempt, expected in SEEDS.items():
        # The pinned layout, built by hand: parts joined by NUL, digest bytes 0-3 big-endian, 31 bits.
        message = b"\x00".join(
            [
                b"narration-seed/v1",
                VOICE_HASH.encode("ascii"),
                hashlib.sha256(ENGINE_TEXT.encode("utf-8")).hexdigest().encode("ascii"),
                str(attempt).encode("ascii"),
            ]
        )
        assert keys.seed_message(voice_hash=VOICE_HASH, engine_text=ENGINE_TEXT, attempt=attempt) == message
        assert int.from_bytes(hashlib.sha256(message).digest()[:4], "big") & 0x7FFFFFFF == expected
        assert keys.seed(voice_hash=VOICE_HASH, engine_text=ENGINE_TEXT, attempt=attempt) == expected


def test_ids_are_the_prefix_and_16_hex_of_their_key_s6() -> None:
    assert keys.render_id(RENDER_KEY) == "rn_0ed0f147896bf2cc"
    assert keys.take_id(DELIVERY_KEY) == "tk_0f057208e4a6f5bb"
    assert keys.analysis_id(ANALYSIS_KEY) == "an_f7fb5269b0291188"
    for bad in ("0ed0f147896bf2cca87ce7c629e305dc62d58c1dba177bcc37f3d08608ebd544", "sha256:XYZ", ""):
        with pytest.raises(ValueError):
            keys.render_id(bad)


# ---------------------------------------------------------------- what is and is not in a key


def test_hashed_objects_carry_exactly_the_members_of_s10_2() -> None:
    assert set(keys.voice_hash_object(**voice_kwargs())) == {
        "schema",
        "model",
        "clip_sha256",
        "transcript",
        "language",
        "x_vector_only_mode",
    }
    assert set(keys.render_key_object(**render_kwargs())) == {
        "schema",
        "engine_profile_hash",
        "voice_hash",
        "engine_text",
        "seed",
    }
    delivery = keys.delivery_key_object(raw_sha256="ab" * 32, profile=DELIVERY_PROFILE, tools=TOOLS)
    assert set(delivery) == {"schema", "raw_sha256", "delivery_profile", "tools", "stretch"}
    assert set(delivery["delivery_profile"]) == {f.name for f in dataclasses.fields(DeliveryConfig)}
    assert delivery["stretch"] is None
    measurement = keys.measurement_key_object(**measurement_kwargs())
    assert set(measurement) == {
        "schema",
        "voice_hash",
        "engine_profile_hash",
        "corpus_version",
        "ladder_settings",
        "pace_method",
    }
    assert set(measurement["ladder_settings"]) == {f.name for f in dataclasses.fields(MeasurementConfig)} - {"corpus"}
    analysis = keys.analysis_key_object(analysis_inputs())
    assert set(analysis) == {"schema"} | {f.name for f in dataclasses.fields(AnalysisKeyInputs)}


def test_segment_id_not_in_seed_s10_3() -> None:
    assert set(inspect.signature(keys.seed).parameters) == {"voice_hash", "engine_text", "attempt"}
    assert set(inspect.signature(KeyBuilder.seed).parameters) == {"self", "voice_hash", "engine_text", "attempt"}
    # The same text in the same voice gives the same seed and render key in any script, under any segment id.
    seed = keys.seed(voice_hash=VOICE_HASH, engine_text=ENGINE_TEXT, attempt=0)
    assert seed == SEEDS[0]
    assert "segment_id" not in keys.render_key_object(**render_kwargs(seed=seed))


def test_voice_hash_is_the_clip_bytes_and_words_not_its_path_s10_2() -> None:
    assert "path" not in inspect.signature(keys.voice_hash).parameters


def test_voice_hash_puts_the_transcript_in_nfc_itself_s10_2() -> None:
    # Callers pass the transcript as received; an NFD and an NFC spelling are one voice.
    nfd = "Café by the reef."
    nfc = "Café by the reef."
    assert unicodedata.normalize("NFC", nfd) == nfc != nfd
    assert keys.voice_hash(**voice_kwargs(transcript=nfd)) == keys.voice_hash(**voice_kwargs(transcript=nfc))
    assert keys.voice_hash(**voice_kwargs(transcript=nfc)) == VOICE_HASH
    assert keys.Keys().voice_hash(**voice_kwargs(transcript=nfd)) == VOICE_HASH


@pytest.mark.parametrize(
    "model",
    [
        names.MODEL_QWEN_DESIGN,
        names.MODEL_QWEN_BASE + "@" + "f" * 40,  # a revision
        "qwen3-base-1.7b.p1",  # an engine profile id
        ENGINE_PROFILE_HASH,
        names.MODEL_QWEN_BASE.lower(),
    ],
)
def test_voice_hash_model_is_always_the_cloning_models_repo_id_s10_2(model: str) -> None:
    # A voice's hash must survive a re-pin (measurements are kept per voice), so no pin-specific name is hashed.
    with pytest.raises(ValueError, match="model"):
        keys.voice_hash(**voice_kwargs(model=model))


def test_voice_hash_language_is_the_services_language_s10_2() -> None:
    with pytest.raises(ValueError, match="language"):
        keys.voice_hash(**voice_kwargs(language="english"))


def test_measurement_key_takes_the_corpus_from_its_version_not_the_setting_s10_2() -> None:
    renamed = dataclasses.replace(MEASUREMENT_SETTINGS, corpus="another-corpus.v9")
    assert keys.measurement_key(**measurement_kwargs(settings=renamed)) == MEASUREMENT_KEY
    with pytest.raises(ValueError, match="corpus_version"):
        keys.measurement_key(**measurement_kwargs(corpus_version="narration-en.v1"))


def test_analysis_key_does_not_depend_on_hint_order_s10_2() -> None:
    reordered = analysis_inputs(
        hints_qa=(("Zeta", (), "zeta"), ("Ossavine", ("Osavine", "Ossa vine"), None)),
    )
    swapped = analysis_inputs(
        hints_qa=(("Ossavine", ("Ossa vine", "Osavine"), None), ("Zeta", (), "zeta")),
    )
    assert keys.analysis_key(reordered) == keys.analysis_key(swapped)
    assert keys.analysis_key(analysis_inputs(hints_qa=(("Ossavine", ("Osavine", "Ossa vine"), None),))) == ANALYSIS_KEY


def _differs(a: str, b: str) -> bool:
    return a != b and keys.KEY_PATTERN.fullmatch(a) is not None and keys.KEY_PATTERN.fullmatch(b) is not None


@pytest.mark.parametrize(
    "change",
    [
        {"clip_sha256": "f" * 64},
        {"transcript": "Café by the reef!"},
        {"x_vector_only_mode": True},
    ],
)
def test_voice_hash_changes_with_each_input_s10_2(change: dict[str, Any]) -> None:
    assert _differs(keys.voice_hash(**voice_kwargs(**change)), VOICE_HASH)


@pytest.mark.parametrize(
    "change",
    [
        {"engine_profile_hash": "sha256:" + "0" * 64},
        {"voice_hash": "sha256:" + "1" * 64},
        {"engine_text": ENGINE_TEXT + " "},
        {"seed": 1834112094},
    ],
)
def test_render_key_changes_with_each_input_s10_2(change: dict[str, Any]) -> None:
    assert _differs(keys.render_key(**render_kwargs(**change)), RENDER_KEY)


_DELIVERY_CHANGES: list[dict[str, Any]] = [
    {"sample_rate": 44100},
    {"subtype": "PCM_16"},
    {"target_lufs": -23.0},
    {"true_peak_dbtp": -2.0},
    {"trim_rel_db": -35.0},
    {"trim_floor_dbfs": -60.0},
    {"trim_pad_s": 0.1},
    {"fade_s": 0.02},
]


def test_the_delivery_changes_cover_every_field_of_the_profile() -> None:
    assert {next(iter(c)) for c in _DELIVERY_CHANGES} == {f.name for f in dataclasses.fields(DeliveryConfig)}


@pytest.mark.parametrize("change", _DELIVERY_CHANGES)
def test_delivery_key_changes_with_each_profile_setting_s10_2(change: dict[str, Any]) -> None:
    profile = dataclasses.replace(DELIVERY_PROFILE, **change)
    assert _differs(keys.delivery_key(raw_sha256="ab" * 32, profile=profile, tools=TOOLS), DELIVERY_KEY)


@pytest.mark.parametrize(
    "raw, tools",
    [
        ("ac" * 32, TOOLS),
        ("ab" * 32, dataclasses.replace(TOOLS, resampler="soxr 0.5.1")),
        ("ab" * 32, dataclasses.replace(TOOLS, loudness_meter="pyloudnorm 0.2.0")),
        ("ab" * 32, dataclasses.replace(TOOLS, post="narration.post/2")),
    ],
)
def test_delivery_key_changes_with_the_raw_audio_and_the_tools_s10_2(raw: str, tools: DeliveryTools) -> None:
    assert _differs(keys.delivery_key(raw_sha256=raw, profile=DELIVERY_PROFILE, tools=tools), DELIVERY_KEY)


_LADDER_CHANGES: list[dict[str, Any]] = [
    {"seeds": 4},
    {"length_ladder_spoken_chars": (80, 150, 250)},
    {"trend_band_max_chars": 250},
    {"pace_tol_min": 0.12},
    {"sim_warn_margin": 0.02},
    {"sim_fail_floor": 0.85},
]


def test_the_ladder_changes_cover_every_field_but_corpus() -> None:
    assert {next(iter(c)) for c in _LADDER_CHANGES} == {f.name for f in dataclasses.fields(MeasurementConfig)} - {
        "corpus"
    }


@pytest.mark.parametrize(
    "change",
    [
        {"voice_hash": "sha256:" + "1" * 64},
        {"engine_profile_hash": "sha256:" + "0" * 64},
        {"corpus_version": "narration-en.v1@sha256:" + "0" * 64},
        *({"settings": dataclasses.replace(MEASUREMENT_SETTINGS, **c)} for c in _LADDER_CHANGES),
    ],
)
def test_measurement_key_changes_with_each_input_s10_2(change: dict[str, Any]) -> None:
    assert _differs(keys.measurement_key(**measurement_kwargs(**change)), MEASUREMENT_KEY)


_ANALYSIS_CHANGES: list[dict[str, Any]] = [
    {"delivery_sha256": "ce" * 32},
    {"spoken_text": SPOKEN.replace("rock", "reef")},
    {"cue_spans": ((0, 53), (53, 127))},
    {"exact_spans": ((1, 3, 6),)},
    {"hints_qa": (("Ossavine", ("Osavine",), None),)},
    {"qa_profile": "default.v4"},
    {"text_checks_version": "text-1.2.0"},
    {"number_reader": "another-reader@1"},
    {"asr_model": "openai/whisper-large-v3@" + "3" * 40},
    {"sv_model": "microsoft/wavlm-base-plus-sv@" + "4" * 40},
    {"aligner_method_id": "ctc-forced-align"},
    {"measurement_key": None},
]


def test_the_analysis_changes_cover_every_field() -> None:
    assert {next(iter(c)) for c in _ANALYSIS_CHANGES} == {f.name for f in dataclasses.fields(AnalysisKeyInputs)}


@pytest.mark.parametrize("change", _ANALYSIS_CHANGES)
def test_analysis_key_changes_with_each_input_s10_2(change: dict[str, Any]) -> None:
    assert _differs(keys.analysis_key(analysis_inputs(**change)), ANALYSIS_KEY)


@pytest.mark.parametrize(
    "hints_qa",
    [
        (("Ossavine", ("Ossa vine", "Osavine"), "o s a v i n e"),),
        (("Ossavine", ("Ossa vine", "Osavine", "Osaveen"), None),),
        (("Ossavin", ("Ossa vine", "Osavine"), None),),
    ],
)
def test_analysis_key_changes_with_each_hint_qa_input_s10_2(hints_qa: Any) -> None:
    assert _differs(keys.analysis_key(analysis_inputs(hints_qa=hints_qa)), ANALYSIS_KEY)


def test_seed_changes_with_each_input_s10_3() -> None:
    base = keys.seed(voice_hash=VOICE_HASH, engine_text=ENGINE_TEXT, attempt=0)
    assert keys.seed(voice_hash="sha256:" + "1" * 64, engine_text=ENGINE_TEXT, attempt=0) != base
    assert keys.seed(voice_hash=VOICE_HASH, engine_text=ENGINE_TEXT + ".", attempt=0) != base
    assert keys.seed(voice_hash=VOICE_HASH, engine_text=ENGINE_TEXT, attempt=1) != base
    assert all(0 <= s <= 0x7FFFFFFF for s in SEEDS.values())


# ---------------------------------------------------------------- malformed inputs


@pytest.mark.parametrize(
    "call",
    [
        lambda: keys.voice_hash(**voice_kwargs(clip_sha256="sha256:" + CLIP)),
        lambda: keys.voice_hash(**voice_kwargs(clip_sha256=CLIP.upper())),
        lambda: keys.voice_hash(**voice_kwargs(transcript="")),
        lambda: keys.render_key(**render_kwargs(voice_hash=CLIP)),
        lambda: keys.render_key(**render_kwargs(seed=-1)),
        lambda: keys.render_key(**render_kwargs(seed=2**31)),
        lambda: keys.render_key(**render_kwargs(seed=True)),
        lambda: keys.delivery_key(raw_sha256="sha256:" + "ab" * 32, profile=DELIVERY_PROFILE, tools=TOOLS),
        lambda: keys.seed(voice_hash=VOICE_HASH, engine_text=ENGINE_TEXT, attempt=-1),
        lambda: keys.analysis_key(analysis_inputs(measurement_key="c2fb0d03")),
        lambda: keys.analysis_key(analysis_inputs(cue_spans=((0, -1),))),
    ],
)
def test_keys_refuse_malformed_inputs(call: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda nl: keys.voice_hash(**voice_kwargs(clip_sha256=CLIP + nl)),
        lambda nl: keys.render_key(**render_kwargs(voice_hash=VOICE_HASH + nl)),
        lambda nl: keys.render_key(**render_kwargs(engine_profile_hash=ENGINE_PROFILE_HASH + nl)),
        lambda nl: keys.measurement_key(**measurement_kwargs(corpus_version=CORPUS_VERSION + nl)),
        lambda nl: keys.delivery_key(raw_sha256="ab" * 32 + nl, profile=DELIVERY_PROFILE, tools=TOOLS),
        lambda nl: keys.analysis_key(analysis_inputs(delivery_sha256="cd" * 32 + nl)),
        lambda nl: keys.analysis_key(analysis_inputs(measurement_key=ANALYSIS_MEASUREMENT_KEY + nl)),
        lambda nl: keys.seed(voice_hash=VOICE_HASH + nl, engine_text=ENGINE_TEXT, attempt=0),
        lambda nl: keys.take_id(DELIVERY_KEY + nl),
        lambda nl: ulid.decode("01ARYZ6S41TSV4RRFFQ69G5FAV" + nl),
    ],
)
def test_a_trailing_newline_is_refused(call: Any) -> None:
    # ``$`` matches before a final newline; every check is a whole match, so a trailing newline is refused.
    call("")  # the well-formed value passes
    with pytest.raises((TypeError, ValueError)):
        call("\n")


def test_every_pattern_is_a_whole_match() -> None:
    for pattern, value in (
        (keys.HEX64_PATTERN, "ab" * 32),
        (keys.KEY_PATTERN, VOICE_HASH),
        (keys.CORPUS_VERSION_PATTERN, CORPUS_VERSION),
        (ulid.ULID_PATTERN, "01ARYZ6S41TSV4RRFFQ69G5FAV"),
    ):
        assert pattern.fullmatch(value) and not pattern.fullmatch(value + "\n")
    assert keys.KEY_PATTERN.pattern == names.ID_PATTERNS["voice_hash"]
    assert ulid.ULID_PATTERN.pattern == names.ID_PATTERNS["design_id"]
    assert is_ulid("01ARYZ6S41TSV4RRFFQ69G5FAV") and not is_ulid("01ARYZ6S41TSV4RRFFQ69G5FAV\n")


def test_hint_aliases_must_be_a_sequence_not_a_string_s10_2() -> None:
    # A bare string would be hashed as its characters.
    with pytest.raises(TypeError, match="asr_aliases"):
        keys.analysis_key(analysis_inputs(hints_qa=(("Ossavine", "Osavine", None),)))


def test_repeated_hint_aliases_do_not_change_the_key_s10_2() -> None:
    repeated = analysis_inputs(hints_qa=(("Ossavine", ("Osavine", "Ossa vine", "Osavine"), None),))
    assert keys.analysis_key(repeated) == ANALYSIS_KEY


# ---------------------------------------------------------------- the KeyBuilder


def test_keys_implements_the_keybuilder_contract() -> None:
    builder: KeyBuilder = keys.Keys()
    assert isinstance(builder, KeyBuilder)
    assert builder.voice_hash(**voice_kwargs()) == VOICE_HASH
    assert builder.render_key(**render_kwargs()) == RENDER_KEY
    assert builder.delivery_key(raw_sha256="ab" * 32, profile=DELIVERY_PROFILE, tools=TOOLS) == DELIVERY_KEY
    assert builder.measurement_key(**measurement_kwargs()) == MEASUREMENT_KEY
    assert builder.analysis_key(analysis_inputs()) == ANALYSIS_KEY
    assert builder.seed(voice_hash=VOICE_HASH, engine_text=ENGINE_TEXT, attempt=7) == SEEDS[7]
    assert builder.canonical_json({"b": 1, "a": [True, None]}) == b'{"a":[true,null],"b":1}'


def test_job_and_design_ids_are_ulids_s5() -> None:
    builder = keys.Keys()
    job_ids = [builder.new_job_id() for _ in range(50)]
    design_ids = [builder.new_design_id() for _ in range(50)]
    assert all(j.startswith(names.JOB_ID_PREFIX) and is_ulid(j[len(names.JOB_ID_PREFIX) :]) for j in job_ids)
    assert all(is_ulid(d) for d in design_ids)
    assert job_ids == sorted(job_ids) and len(set(job_ids)) == 50
    assert design_ids == sorted(design_ids) and len(set(design_ids)) == 50


def test_the_qa_profile_moves_no_render_key_seed_or_measurement_dc19() -> None:
    """DC-19 moved the QA profile to default.v4 (contracts 1.6.7). Only the analysis key names it: every voice
    hash, render key, seed, delivery key and measurement key stays as it was, so a voice measured under
    default.v3 stays measured and every cached render and take is reused; only QA runs again."""
    for fn in (keys.voice_hash, keys.render_key, keys.seed, keys.delivery_key, keys.measurement_key):
        assert not any("qa" in p for p in inspect.signature(fn).parameters), fn.__name__
    assert keys.render_key(**render_kwargs()) == RENDER_KEY
    assert keys.measurement_key(**measurement_kwargs()) == MEASUREMENT_KEY
    assert {a: keys.seed(voice_hash=VOICE_HASH, engine_text=ENGINE_TEXT, attempt=a) for a in SEEDS} == SEEDS
    old = keys.analysis_key(analysis_inputs(qa_profile="default.v3"))
    assert keys.analysis_key(analysis_inputs(qa_profile=names.QA_PROFILE)) != old
