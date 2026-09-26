"""The voice clip a worker clones from (design section 17, App. A): a working copy inside the store.

A caller's clip is copied into the store before a worker sees it, to ``scratch/voices/<sha256>.wav`` (App. A's
``ref_wav``). The front-end copies it at submit (WP36); if the copy is gone by the time the job runs (scratch
is a working area), the engine copies it again from the request's path, and only if the file there still has
the sha256 the request sent. Anything else is ``VOICE_FILE_MISMATCH``: the clip changed or moved, and the
caller must send it again. The caller's path is read only through section 17.3's check, and at most 20 MB
of it. The copy is written to a temporary name and renamed, so a worker never reads a half-written clip. It
is a working copy, not a record of the caller's voice (section 0.2).
"""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Final

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Store

from .admission import STORE_FULL_RETRY_S
from .plan import VoiceSpec

VOICES_DIR: Final = "voices"
MAX_CLIP_BYTES: Final = 20 * 1024 * 1024
"""Section 17.3: a voice clip is at most 20 MB (MiB, as the service counts memory)."""
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


def stage_clip(store: Store, voice: VoiceSpec, *, check_path: PathCheck | None) -> Path:
    """The working copy of the request's clip, copied from ``voice.path`` if it is not there (see the module
    docstring).

    A caller's file is read only through ``check_path`` (section 17.3, ``Platform.check_readable_path``):
    with none wired, nothing is read (``INTERNAL``: the service was built without its path check). Raises
    ``NarrationError``: whatever ``check_path`` raises (``PATH_NOT_ALLOWED``); ``UNSUPPORTED_AUDIO`` for a
    file over ``MAX_CLIP_BYTES``; ``VOICE_FILE_MISMATCH`` when the file cannot be read or its sha256 is not
    the one sent; ``STORE_FULL`` when the store's disk fills during the copy. Nothing is left half copied.
    """
    target = clip_path(store, voice.sha256)
    if target.is_file() and _sha256(target) == voice.sha256:
        return target
    if check_path is None:
        raise NarrationError(
            codes.INTERNAL,
            "the job engine was built without the path check of section 17.3, so it read no clip",
            retryable=False,
            hint="This is a bug in the service; nothing was read or rendered. Report it.",
        )
    source = check_path(voice.path)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    try:
        digest = _copy(source, tmp, voice)
        if digest != voice.sha256:
            raise NarrationError(
                codes.VOICE_FILE_MISMATCH,
                f"the clip at {voice.path} no longer has the sha256 the request sent",
                field="voice.sha256",
                details={"expected": voice.sha256, "actual": digest},
            )
        try:
            os.replace(tmp, target)
        except OSError as exc:
            _raise_if_full(exc)
            raise
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _copy(source: Path, tmp: Path, voice: VoiceSpec) -> str:
    """Copy ``source`` to ``tmp``, at most ``MAX_CLIP_BYTES``, and return its sha256. A read that fails is the
    caller's file (``VOICE_FILE_MISMATCH``); a write that fails is the store's (``_raise_if_full``)."""
    digest, copied = hashlib.sha256(), 0
    try:
        size = source.stat().st_size
        src = source.open("rb")
    except OSError as exc:
        raise _unreadable(voice, exc) from exc
    with src:
        if size > MAX_CLIP_BYTES:
            raise _too_large(size)
        try:
            dst = tmp.open("wb")
        except OSError as exc:
            _raise_if_full(exc)
            raise
        with dst:
            while True:
                try:
                    chunk = src.read(_CHUNK)
                except OSError as exc:
                    raise _unreadable(voice, exc) from exc
                if not chunk:
                    return digest.hexdigest()
                copied += len(chunk)
                if copied > MAX_CLIP_BYTES:  # it grew since it was checked
                    raise _too_large(copied)
                digest.update(chunk)
                try:
                    dst.write(chunk)
                except OSError as exc:
                    _raise_if_full(exc)
                    raise


def _unreadable(voice: VoiceSpec, exc: OSError) -> NarrationError:
    return NarrationError(
        codes.VOICE_FILE_MISMATCH,
        f"the clip at {voice.path} cannot be read any more ({exc.strerror or exc})",
        field="voice.path",
    )


def _too_large(size: int) -> NarrationError:
    return NarrationError(
        codes.UNSUPPORTED_AUDIO,
        f"the clip is {size} bytes; a voice clip is at most {MAX_CLIP_BYTES} bytes (20 MB)",
        field="voice.path",
        details={"bytes": size, "max_bytes": MAX_CLIP_BYTES},
    )


def _raise_if_full(exc: OSError) -> None:
    """A write into the store failed: ``STORE_FULL`` when its disk is full. Any other failure is raised as it
    is (a broken store, which the runner reports as ``INTERNAL``)."""
    if exc.errno == errno.ENOSPC:
        raise NarrationError(
            codes.STORE_FULL,
            "the store's disk filled while the voice clip was copied into it",
            retry_after_s=STORE_FULL_RETRY_S,
        ) from exc


__all__ = ["MAX_CLIP_BYTES", "VOICES_DIR", "PathCheck", "clip_path", "require_synthetic", "stage_clip"]
