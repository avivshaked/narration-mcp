"""Keys, ids and seeds (design sections 5, 6, 10.2 and 10.3; plan.md WP12).

Every key is ``"sha256:"`` + the 64 hex digits of the sha256 of the RFC 8785 canonical JSON of one object.
Each object is **built explicitly here**, member by member, from the inputs section 10.2 names, and carries
a ``schema`` member (``narration.contracts.names``), so no change to a record or to its JSON form can move
a key. The keys come from the request and the service's pins only, never from an earlier request; the
segment id is in no key and not in the seed.

**These values are forever once released**: a changed byte changes every take id. ``tests/keys`` pins
them with golden values derived from hand-written canonical bytes.

Canonicalisation this module adds, beyond RFC 8785:

- ``voice_hash``: the transcript is put in Unicode NFC (section 10.2).
- ``analysis_key``: the hints' QA inputs are a set, so they are sorted (by term, then aliases, then
  ``align_as``), and each hint's ``asr_aliases`` are a set too: sorted, with repeats dropped. The order a
  request lists them in, or a repeated alias, does not change the key.

Nothing else is normalised: text is hashed exactly as the text pipeline (WP10) produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Final

import rfc8785

from narration.config import MeasurementConfig
from narration.contracts import names
from narration.contracts.interfaces import AnalysisKeyInputs, DeliveryProfile
from narration.contracts.models import DeliveryTools

from .ulid import UlidGenerator, is_ulid

__all__ = [
    "CORPUS_VERSION_PATTERN",
    "HEX64_PATTERN",
    "KEY_PATTERN",
    "SEED_MASK",
    "Keys",
    "analysis_id",
    "analysis_key",
    "analysis_key_object",
    "canonical_json",
    "delivery_key",
    "delivery_key_object",
    "hash_key",
    "is_ulid",
    "key_hex",
    "measurement_key",
    "measurement_key_object",
    "render_id",
    "render_key",
    "render_key_object",
    "seed",
    "seed_message",
    "take_id",
    "voice_hash",
    "voice_hash_object",
]

HEX64_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
"""A file's sha256 as the service reports it: 64 lower-case hex digits, no prefix."""
KEY_PATTERN: Final = re.compile(names.ID_PATTERNS["voice_hash"])
"""A key or a voice / engine profile hash: ``sha256:`` + 64 lower-case hex digits."""
SEED_MASK: Final = 0x7FFFFFFF
_MAX_ATTEMPT: Final = 1 << 31


# ---------------------------------------------------------------- canonical JSON and hashing


def canonical_json(value: Any) -> bytes:
    """RFC 8785 canonical JSON of a plain JSON value (dicts or mappings with string keys, lists or tuples,
    strings, integers within ±2**53, finite floats, booleans, None).

    Records (dataclasses) are refused on purpose: a key is hashed over an object built member by member,
    never over a record's serialised form, whose rules may change.
    """
    return rfc8785.dumps(_plain(value))


def hash_key(obj: Mapping[str, Any]) -> str:
    """``"sha256:"`` + the hex sha256 of ``canonical_json(obj)``; ``obj`` must carry a ``schema`` member."""
    if not isinstance(obj.get("schema"), str) or not obj["schema"]:
        raise ValueError("a hashed object must carry a 'schema' member")
    return names.HASH_PREFIX + hashlib.sha256(canonical_json(obj)).hexdigest()


def key_hex(key: str) -> str:
    """The 64 hex digits of a key (``sha256:`` + 64 hex)."""
    _check_key(key, "key")
    return key[len(names.HASH_PREFIX) :]


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        raise TypeError("canonical_json takes plain JSON values; build the object explicitly, not from a record")
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"object keys must be strings, got {type(k).__name__}")
            out[k] = _plain(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"not a JSON value: {type(value).__name__}")


# ---------------------------------------------------------------- input checks


def _check_hex64(value: str, what: str) -> None:
    if not isinstance(value, str) or not HEX64_PATTERN.fullmatch(value):
        raise ValueError(f"{what} must be 64 lower-case hex digits (a file's sha256), got {value!r}")


def _check_key(value: str, what: str) -> None:
    if not isinstance(value, str) or not KEY_PATTERN.fullmatch(value):
        raise ValueError(f"{what} must be 'sha256:' + 64 lower-case hex digits, got {value!r}")


def _check_text(value: str, what: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{what} must not be empty")


def _check_int(value: int, what: str, *, low: int = 0, high: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an integer")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{what} = {value} is outside {low}..{high if high is not None else ''}")


# ---------------------------------------------------------------- voice_hash (section 10.2)


def voice_hash_object(
    *, model: str, clip_sha256: str, transcript: str, language: str, x_vector_only_mode: bool
) -> dict[str, Any]:
    """The object ``voice_hash`` hashes: {schema, model, clip_sha256, transcript (NFC), language,
    x_vector_only_mode}. The clip's path is not in it: a voice is its bytes and its words.

    ``model`` must be ``names.MODEL_QWEN_BASE``, the cloning model's repo id, and ``language`` must be
    ``names.LANGUAGE``. Anything else (a revision, an engine profile id or hash, another spelling) is
    refused rather than hashed: it would give the same voice a second hash, and a voice's hash must survive
    a re-pin, because measurements are kept per voice (sections 10.1, 15). The transcript is taken as
    received and put in NFC here.
    """
    _check_text(model, "model")
    if model != names.MODEL_QWEN_BASE:
        raise ValueError(
            f"model must be the cloning model's repo id {names.MODEL_QWEN_BASE!r} (not a revision or an engine "
            f"profile), got {model!r}"
        )
    _check_hex64(clip_sha256, "clip_sha256")
    _check_text(transcript, "transcript")
    _check_text(language, "language")
    if language != names.LANGUAGE:
        raise ValueError(f"language must be {names.LANGUAGE!r}, got {language!r}")
    if not isinstance(x_vector_only_mode, bool):
        raise TypeError("x_vector_only_mode must be a boolean")
    return {
        "schema": names.VOICE_SCHEMA,
        "model": model,
        "clip_sha256": clip_sha256,
        "transcript": unicodedata.normalize("NFC", transcript),
        "language": language,
        "x_vector_only_mode": x_vector_only_mode,
    }


def voice_hash(*, model: str, clip_sha256: str, transcript: str, language: str, x_vector_only_mode: bool) -> str:
    """``voice_hash`` (section 10.2), computed from what a request sends on every request."""
    return hash_key(
        voice_hash_object(
            model=model,
            clip_sha256=clip_sha256,
            transcript=transcript,
            language=language,
            x_vector_only_mode=x_vector_only_mode,
        )
    )


# ---------------------------------------------------------------- measurement key (section 10.2)


CORPUS_VERSION_PATTERN: Final = re.compile(r"[^@\s]+@" + re.escape(names.HASH_PREFIX) + r"[0-9a-f]{64}")
"""``"<set id>@sha256:<hex>"``: the corpus set and the sha256 of its manifest, so its content is in the key."""


def measurement_key_object(
    *, voice_hash: str, engine_profile_hash: str, corpus_version: str, settings: MeasurementConfig
) -> dict[str, Any]:
    """The object the measurement key hashes: {schema, voice_hash, engine_profile_hash, corpus_version,
    ladder_settings, pace_method}, where ``ladder_settings`` holds every field of ``settings`` except ``corpus``
    (whose content ``corpus_version`` already names), and ``pace_method`` is ``names.PACE_METHOD``, the rule the
    pace model is measured by (added with ``narration.measurement-key/v2``, WP47)."""
    _check_key(voice_hash, "voice_hash")
    _check_key(engine_profile_hash, "engine_profile_hash")
    if not isinstance(corpus_version, str) or not CORPUS_VERSION_PATTERN.fullmatch(corpus_version):
        raise ValueError(f"corpus_version must be '<set id>@sha256:<64 hex>', got {corpus_version!r}")
    return {
        "schema": names.MEASUREMENT_KEY_SCHEMA,
        "voice_hash": voice_hash,
        "engine_profile_hash": engine_profile_hash,
        "corpus_version": corpus_version,
        "ladder_settings": _fields_object(settings, exclude=("corpus",)),
        "pace_method": names.PACE_METHOD,
    }


def measurement_key(
    *, voice_hash: str, engine_profile_hash: str, corpus_version: str, settings: MeasurementConfig
) -> str:
    """The key of a voice's measurement (section 10.2)."""
    return hash_key(
        measurement_key_object(
            voice_hash=voice_hash,
            engine_profile_hash=engine_profile_hash,
            corpus_version=corpus_version,
            settings=settings,
        )
    )


def _fields_object(record: Any, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Every field of a settings dataclass, read attribute by attribute (not through ``serial.to_json``, whose
    presentation rules must never move a key). A field added to the class enters the key, on purpose."""
    if not dataclasses.is_dataclass(record) or isinstance(record, type):
        raise TypeError(f"expected a settings record, got {type(record).__name__}")
    return {f.name: _plain(getattr(record, f.name)) for f in dataclasses.fields(record) if f.name not in exclude}


# ---------------------------------------------------------------- render key (section 10.2)


def render_key_object(*, engine_profile_hash: str, voice_hash: str, engine_text: str, seed: int) -> dict[str, Any]:
    """The object ``render_key`` hashes: {schema: "narration.render/v1", engine_profile_hash, voice_hash,
    engine_text, seed}. The segment id is not in it."""
    _check_key(engine_profile_hash, "engine_profile_hash")
    _check_key(voice_hash, "voice_hash")
    _check_text(engine_text, "engine_text")
    _check_int(seed, "seed", high=SEED_MASK)
    return {
        "schema": names.RENDER_SCHEMA,
        "engine_profile_hash": engine_profile_hash,
        "voice_hash": voice_hash,
        "engine_text": engine_text,
        "seed": seed,
    }


def render_key(*, engine_profile_hash: str, voice_hash: str, engine_text: str, seed: int) -> str:
    """``render_key`` (section 10.2): what names a raw render."""
    return hash_key(
        render_key_object(
            engine_profile_hash=engine_profile_hash, voice_hash=voice_hash, engine_text=engine_text, seed=seed
        )
    )


# ---------------------------------------------------------------- delivery key (section 10.2)


def delivery_key_object(*, raw_sha256: str, profile: DeliveryProfile, tools: DeliveryTools) -> dict[str, Any]:
    """The object ``delivery_key`` hashes: {schema, raw_sha256, delivery_profile (every field of ``profile``:
    trim rule, target LUFS, true-peak ceiling, sample rate, subtype, fades), tools {resampler,
    loudness_meter, post}, stretch: null}. ``tools.post`` is the post-processing rules' version
    (``names.POST_RULES``, design revision 5.3 section 10.2), kept with the tools as ``DeliveryTools`` and
    App. B's take.json keep it."""
    _check_hex64(raw_sha256, "raw_sha256")
    _check_text(tools.resampler, "tools.resampler")
    _check_text(tools.loudness_meter, "tools.loudness_meter")
    _check_text(tools.post, "tools.post")
    return {
        "schema": names.DELIVERY_KEY_SCHEMA,
        "raw_sha256": raw_sha256,
        "delivery_profile": _fields_object(profile),
        "tools": {"resampler": tools.resampler, "loudness_meter": tools.loudness_meter, "post": tools.post},
        "stretch": None,
    }


def delivery_key(*, raw_sha256: str, profile: DeliveryProfile, tools: DeliveryTools) -> str:
    """``delivery_key`` (section 10.2): what names a take (``take_id``)."""
    return hash_key(delivery_key_object(raw_sha256=raw_sha256, profile=profile, tools=tools))


# ---------------------------------------------------------------- analysis key (section 10.2)


def _hints_qa_object(hints_qa: Sequence[tuple[str, tuple[str, ...], str | None]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, hint in enumerate(hints_qa):
        term, aliases, align_as = hint
        _check_text(term, f"hints_qa[{i}].term")
        if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Sequence):
            # A bare string would be read as a list of its characters.
            raise TypeError(f"hints_qa[{i}].asr_aliases must be a sequence of strings, not {type(aliases).__name__}")
        for alias in aliases:
            _check_text(alias, f"hints_qa[{i}].asr_aliases[]", allow_empty=True)
        if align_as is not None:
            _check_text(align_as, f"hints_qa[{i}].align_as", allow_empty=True)
        # The aliases are a set: order and repeats do not change the key.
        out.append({"term": term, "asr_aliases": sorted(set(aliases)), "align_as": align_as})
    out.sort(key=lambda h: (h["term"], h["asr_aliases"], h["align_as"] is not None, h["align_as"] or ""))
    return out


def analysis_key_object(inputs: AnalysisKeyInputs) -> dict[str, Any]:
    """The object ``analysis_key`` hashes: {schema, delivery_sha256, spoken_text, cue_spans, exact_spans,
    hints_qa [{term, asr_aliases, align_as}], qa_profile, text_checks_version, number_reader, asr_model,
    sv_model, aligner_method_id, measurement_key}."""
    _check_hex64(inputs.delivery_sha256, "delivery_sha256")
    _check_text(inputs.spoken_text, "spoken_text")
    for i, span in enumerate(inputs.cue_spans):
        if len(span) != 2:
            raise ValueError(f"cue_spans[{i}] must be (start, end)")
        for v in span:
            _check_int(v, f"cue_spans[{i}]")
    for i, span in enumerate(inputs.exact_spans):
        if len(span) != 3:
            raise ValueError(f"exact_spans[{i}] must be (cue, first word, end word)")
        for v in span:
            _check_int(v, f"exact_spans[{i}]")
    for name in ("qa_profile", "text_checks_version", "number_reader", "asr_model", "sv_model", "aligner_method_id"):
        _check_text(getattr(inputs, name), name)
    if inputs.measurement_key is not None:
        _check_key(inputs.measurement_key, "measurement_key")
    return {
        "schema": names.ANALYSIS_KEY_SCHEMA,
        "delivery_sha256": inputs.delivery_sha256,
        "spoken_text": inputs.spoken_text,
        "cue_spans": [[start, end] for start, end in inputs.cue_spans],
        "exact_spans": [[cue, first, end] for cue, first, end in inputs.exact_spans],
        "hints_qa": _hints_qa_object(inputs.hints_qa),
        "qa_profile": inputs.qa_profile,
        "text_checks_version": inputs.text_checks_version,
        "number_reader": inputs.number_reader,
        "asr_model": inputs.asr_model,
        "sv_model": inputs.sv_model,
        "aligner_method_id": inputs.aligner_method_id,
        "measurement_key": inputs.measurement_key,
    }


def analysis_key(inputs: AnalysisKeyInputs) -> str:
    """``analysis_key`` (section 10.2): what names an analysis (``analysis_id``)."""
    return hash_key(analysis_key_object(inputs))


# ---------------------------------------------------------------- seed (section 10.3)


def seed_message(*, voice_hash: str, engine_text: str, attempt: int) -> bytes:
    """The bytes the seed hashes, exactly as ``interfaces.KeyBuilder`` pins them::

        parts = ["narration-seed/v1", voice_hash, sha256_hex(engine_text as UTF-8), str(attempt)]
        message = b"\\x00".join(p.encode("utf-8") for p in parts)

    ``voice_hash`` is the full reported form (``sha256:`` + 64 hex); ``attempt`` is written in decimal.
    """
    _check_key(voice_hash, "voice_hash")
    _check_text(engine_text, "engine_text")
    _check_int(attempt, "attempt", high=_MAX_ATTEMPT)
    parts = [
        names.SEED_SCHEME,
        voice_hash,
        hashlib.sha256(engine_text.encode("utf-8")).hexdigest(),
        str(attempt),
    ]
    return b"\x00".join(p.encode("utf-8") for p in parts)


def seed(*, voice_hash: str, engine_text: str, attempt: int) -> int:
    """The seed of section 10.3: the first four digest bytes, big-endian, masked to 31 bits. It depends on
    the voice, the engine text and the attempt only; the segment id is never an input."""
    digest = hashlib.sha256(seed_message(voice_hash=voice_hash, engine_text=engine_text, attempt=attempt)).digest()
    return int.from_bytes(digest[0:4], "big") & SEED_MASK


# ---------------------------------------------------------------- ids (sections 5, 6)


def render_id(render_key: str) -> str:
    """``rn_`` + the first 16 hex digits of the render key."""
    return names.RENDER_ID_PREFIX + key_hex(render_key)[: names.ID_HEX_CHARS]


def take_id(delivery_key: str) -> str:
    """``tk_`` + the first 16 hex digits of the delivery key."""
    return names.TAKE_ID_PREFIX + key_hex(delivery_key)[: names.ID_HEX_CHARS]


def analysis_id(analysis_key: str) -> str:
    """``an_`` + the first 16 hex digits of the analysis key."""
    return names.ANALYSIS_ID_PREFIX + key_hex(analysis_key)[: names.ID_HEX_CHARS]


# ---------------------------------------------------------------- the KeyBuilder


class Keys:
    """The service's ``KeyBuilder`` (``narration.contracts.interfaces``): keys, ids, seeds and ULIDs.

    Stateless apart from its ULID generator, which keeps the job and design ids it makes strictly
    increasing.
    """

    def __init__(self, ulids: UlidGenerator | None = None) -> None:
        self._ulids = ulids or UlidGenerator()

    def canonical_json(self, value: Any) -> bytes:
        """RFC 8785 canonical JSON of ``value``."""
        return canonical_json(value)

    def voice_hash(
        self, *, model: str, clip_sha256: str, transcript: str, language: str, x_vector_only_mode: bool
    ) -> str:
        """``voice_hash`` (section 10.2)."""
        return voice_hash(
            model=model,
            clip_sha256=clip_sha256,
            transcript=transcript,
            language=language,
            x_vector_only_mode=x_vector_only_mode,
        )

    def measurement_key(
        self, *, voice_hash: str, engine_profile_hash: str, corpus_version: str, settings: MeasurementConfig
    ) -> str:
        """The measurement key (section 10.2)."""
        return measurement_key(
            voice_hash=voice_hash,
            engine_profile_hash=engine_profile_hash,
            corpus_version=corpus_version,
            settings=settings,
        )

    def render_key(self, *, engine_profile_hash: str, voice_hash: str, engine_text: str, seed: int) -> str:
        """``render_key`` (section 10.2)."""
        return render_key(
            engine_profile_hash=engine_profile_hash, voice_hash=voice_hash, engine_text=engine_text, seed=seed
        )

    def delivery_key(self, *, raw_sha256: str, profile: DeliveryProfile, tools: DeliveryTools) -> str:
        """``delivery_key`` (section 10.2)."""
        return delivery_key(raw_sha256=raw_sha256, profile=profile, tools=tools)

    def analysis_key(self, inputs: AnalysisKeyInputs) -> str:
        """``analysis_key`` (section 10.2)."""
        return analysis_key(inputs)

    def seed(self, *, voice_hash: str, engine_text: str, attempt: int) -> int:
        """The seed of section 10.3."""
        return seed(voice_hash=voice_hash, engine_text=engine_text, attempt=attempt)

    def render_id(self, render_key: str) -> str:
        return render_id(render_key)

    def take_id(self, delivery_key: str) -> str:
        return take_id(delivery_key)

    def analysis_id(self, analysis_key: str) -> str:
        return analysis_id(analysis_key)

    def new_job_id(self) -> str:
        """``job_`` + a new ULID."""
        return names.JOB_ID_PREFIX + self._ulids.new()

    def new_design_id(self) -> str:
        """A new ULID."""
        return self._ulids.new()
