"""The voice clip a worker clones from (design section 17, App. A): a working copy inside the store.

A caller's clip is copied into the store before a worker sees it, to ``scratch/voices/<sha256>.wav`` (App. A's
``ref_wav``). The front-end copies it at submit (WP36); if the copy is gone by the time the job runs (scratch
is a working area), the engine copies it again from the request's path, and only if the file there still has
the sha256 the request sent. Anything else is ``VOICE_FILE_MISMATCH``: the clip changed or moved, and the
caller must send it again. The copy is written to a temporary name and renamed, so a worker never reads a
half-written clip. It is a working copy, not a record of the caller's voice (section 0.2).
"""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Final

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Store

from .plan import VoiceSpec

VOICES_DIR: Final = "voices"
_CHUNK: Final = 1 << 20

PathCheck = Callable[[str], Path]
"""Checks a caller's path by section 17.3 (``Platform.check_readable_path``) and returns the file to read."""


def clip_path(store: Store, clip_sha256: str) -> Path:
    """Where the working copy of a clip lives: ``scratch/voices/<sha256>.wav``."""
    return store.scratch_path(VOICES_DIR, f"{clip_sha256}.wav")


def require_synthetic(
    store: Store, allow_sha256: Collection[str], clip_sha256: str, *, field: str = "voice.sha256"
) -> None:
    """Section 17.4: a clip is cloned only if the service designed it (its provenance list) or the owner
    allowed it (``[voices] allow_sha256``). Raises ``NarrationError`` (``VOICE_NOT_SYNTHETIC``) otherwise.

    Every path that clones a clip or measures a voice for cloning checks it, wherever the request came from:
    the front-end at submit, and the job engine again before a worker sees the clip.
    """
    if store.is_provenance(clip_sha256) or clip_sha256 in allow_sha256:
        return
    raise NarrationError(
        codes.VOICE_NOT_SYNTHETIC,
        f"the clip {clip_sha256} is neither one this service designed nor one the owner allowed",
        field=field,
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage_clip(store: Store, voice: VoiceSpec, *, check_path: PathCheck | None = None) -> Path:
    """The working copy of the request's clip, copied from ``voice.path`` if it is not there (see the module
    docstring). Raises ``NarrationError``: ``VOICE_FILE_MISMATCH`` when the file cannot be read or its sha256 is
    not the one sent, and whatever ``check_path`` raises (``PATH_NOT_ALLOWED``)."""
    target = clip_path(store, voice.sha256)
    if target.is_file() and _sha256(target) == voice.sha256:
        return target
    source = check_path(voice.path) if check_path is not None else Path(voice.path)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as src, tmp.open("wb") as dst:
            while chunk := src.read(_CHUNK):
                digest.update(chunk)
                dst.write(chunk)
        if digest.hexdigest() != voice.sha256:
            raise NarrationError(
                codes.VOICE_FILE_MISMATCH,
                f"the clip at {voice.path} no longer has the sha256 the request sent",
                field="voice.sha256",
                details={"expected": voice.sha256, "actual": digest.hexdigest()},
            )
        os.replace(tmp, target)
    except OSError as exc:
        raise NarrationError(
            codes.VOICE_FILE_MISMATCH,
            f"the clip at {voice.path} cannot be read any more ({exc.strerror or exc})",
            field="voice.path",
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)
    return target


__all__ = ["VOICES_DIR", "PathCheck", "clip_path", "require_synthetic", "stage_clip"]
