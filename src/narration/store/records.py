"""Records as the store keeps them: JSON sidecars and rows, with paths relative to the store root.

A record's file paths are stored relative to the store root (``renders/77/rn_…/raw.wav``), so a store can
be moved; the store hands records out with absolute paths. These helpers apply one path mapping to every
path field a record has.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

from narration.contracts.models import (
    Candidate,
    EngineProfile,
    ProfilePictures,
    ProfileRecord,
    RenderRecord,
    TakeRecord,
)
from narration.contracts.serial import to_json

PathMap = Callable[[str], str]


def sidecar_bytes(record: Any) -> bytes:
    """A record's sidecar: pretty JSON in UTF-8 (``ensure_ascii=False``), ending with a newline."""
    return (json.dumps(to_json(record), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def row_json(record: Any) -> str:
    """A record as compact JSON for a ``record`` column."""
    return json.dumps(to_json(record), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def map_render(record: RenderRecord, f: PathMap) -> RenderRecord:
    return dataclasses.replace(record, raw=dataclasses.replace(record.raw, path=f(record.raw.path)))


def map_take(record: TakeRecord, f: PathMap) -> TakeRecord:
    return dataclasses.replace(record, delivery=dataclasses.replace(record.delivery, path=f(record.delivery.path)))


def map_profile(record: ProfileRecord, f: PathMap) -> ProfileRecord:
    pictures = ProfilePictures(spectrogram=f(record.pictures.spectrogram), pitch=f(record.pictures.pitch))
    return dataclasses.replace(record, pictures=pictures)


def map_candidate(record: Candidate, f: PathMap) -> Candidate:
    profile = map_profile(record.profile, f) if record.profile is not None else None
    return dataclasses.replace(record, clip=dataclasses.replace(record.clip, path=f(record.clip.path)), profile=profile)


def map_engine_profile(record: EngineProfile, f: PathMap) -> EngineProfile:
    if record.canary is None:
        return record
    clip = dataclasses.replace(record.canary.clip, path=f(record.canary.clip.path))
    return dataclasses.replace(record, canary=dataclasses.replace(record.canary, clip=clip))
