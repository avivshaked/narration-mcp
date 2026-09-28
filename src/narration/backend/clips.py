"""A caller's voice clip, checked and copied into the store before any worker sees it (design section 17).

A voice is three things the caller sends: a clip path, the clip's sha256 and its transcript (section 7.2).
The front-end admits a clip in this order, each failure a tool error naming the field:

1. **Synthetic voices only** (section 17.4): the sha256 sent must be on the service's provenance list or in
   ``[voices] allow_sha256``, else ``VOICE_NOT_SYNTHETIC``. Nothing is read for a clip that could never be
   cloned (``narration.jobs.voice.require_synthetic``, which the job engine checks again).
2. **The path** (section 17.3, ``Platform.check_readable_path``): absolute, on a local drive, resolving to a
   regular file; else ``PATH_NOT_ALLOWED``.
3. **The file**: at most 20 MB (``MAX_CLIP_BYTES``), else ``UNSUPPORTED_AUDIO``; a WAV of at most ``[limits]
   max_clip_seconds``, else ``UNSUPPORTED_AUDIO``; and its sha256 must be the one sent, else
   ``VOICE_FILE_MISMATCH``, whose ``details.actual`` is the file's sha256. The WAV check comes first, so that
   hash is only ever given for a WAV: a caller cannot learn the sha256 of any other file by naming it
   (``docs/security-review.md``, S1).
4. **The copy** (``keep``): the bytes that were hashed are the bytes copied, to ``scratch/voices/<sha256>.wav``
   (the working copy the engine clones from, App. A's ``ref_wav``), under a temporary name then renamed. A
   file changed after the check is therefore never used. A copy already there with the right sha256 is kept
   as it is (a worker may be reading it). A full disk is ``STORE_FULL``, retryable.

The copy is a working copy, not a record of the caller's voice (section 0.2): the service keeps no voice.
"""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import os
import secrets
from collections.abc import Collection
from pathlib import Path
from typing import BinaryIO, Final

import soundfile

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Platform, Store
from narration.jobs.admission import STORE_FULL_RETRY_S
from narration.jobs.voice import MAX_CLIP_BYTES, clip_path, require_synthetic

_CHUNK: Final = 1 << 20
WAV_FORMATS: Final = frozenset({"WAV", "WAVEX"})
"""The container formats soundfile reports for a WAV file (a plain RIFF WAV, or WAVE_FORMAT_EXTENSIBLE)."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ClipRef:
    """A clip by location, as a request sends it: a voice's clip, or ``profile_voice``'s audio."""

    path: str
    sha256: str


def refield(exc: NarrationError, field: str) -> NarrationError:
    """The same error, naming ``field`` (the platform's path errors name no field)."""
    if exc.field == field:
        return exc
    return NarrationError(
        exc.code,
        exc.message,
        field=field,
        hint=exc.hint,
        details=exc.details,
        retryable=exc.retryable,
        retry_after_s=exc.retry_after_s,
    )


def check_synthetic(store: Store, allow_sha256: Collection[str], clip: ClipRef, *, field: str) -> None:
    """Section 17.4: ``VOICE_NOT_SYNTHETIC`` unless the service designed the clip or the operator allowed it."""
    require_synthetic(store, allow_sha256, clip.sha256, field=f"{field}.sha256")


def admit_clip(
    store: Store,
    platform: Platform,
    clip: ClipRef,
    *,
    max_seconds: float,
    keep: bool,
    field: str = "voice",
) -> Path:
    """Check the caller's clip (section 17.3) and, with ``keep``, copy it into the store (see the module
    docstring). Returns the working copy's path (where it is, or would be, with ``keep`` False).

    Raises ``NarrationError``: ``PATH_NOT_ALLOWED`` (``<field>.path``), ``UNSUPPORTED_AUDIO``,
    ``VOICE_FILE_MISMATCH`` (``<field>.sha256``), ``STORE_FULL``.
    """
    try:
        source = platform.check_readable_path(clip.path)
    except NarrationError as exc:
        raise refield(exc, f"{field}.path") from exc
    target = clip_path(store, clip.sha256)
    if not keep:
        digest = _hash_file(source, clip, field)
        _check_wav(source, max_seconds, field)
        _check_digest(digest, clip, field)
        return target
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    try:
        digest = _copy(source, tmp, clip, field)
        _check_wav(tmp, max_seconds, field)
        _check_digest(digest, clip, field)
        if not (target.is_file() and _sha256(target) == clip.sha256):
            try:
                os.replace(tmp, target)
            except OSError as exc:
                _raise_if_full(exc)
                raise
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _open(source: Path, clip: ClipRef, field: str) -> tuple[BinaryIO, int]:
    try:
        size = source.stat().st_size
        handle = source.open("rb")
    except OSError as exc:
        raise _unreadable(clip, field, exc) from exc
    if size > MAX_CLIP_BYTES:
        handle.close()
        raise _too_large(size, field)
    return handle, size


def _hash_file(source: Path, clip: ClipRef, field: str) -> str:
    handle, _ = _open(source, clip, field)
    digest, read = hashlib.sha256(), 0
    with handle:
        while True:
            try:
                chunk = handle.read(_CHUNK)
            except OSError as exc:
                raise _unreadable(clip, field, exc) from exc
            if not chunk:
                return digest.hexdigest()
            read += len(chunk)
            if read > MAX_CLIP_BYTES:
                raise _too_large(read, field)
            digest.update(chunk)


def _copy(source: Path, tmp: Path, clip: ClipRef, field: str) -> str:
    """Copy at most ``MAX_CLIP_BYTES`` of ``source`` to ``tmp`` and return the sha256 of what was copied."""
    handle, _ = _open(source, clip, field)
    digest, copied = hashlib.sha256(), 0
    with handle:
        try:
            out = tmp.open("wb")
        except OSError as exc:
            _raise_if_full(exc)
            raise
        with out:
            while True:
                try:
                    chunk = handle.read(_CHUNK)
                except OSError as exc:
                    raise _unreadable(clip, field, exc) from exc
                if not chunk:
                    return digest.hexdigest()
                copied += len(chunk)
                if copied > MAX_CLIP_BYTES:  # it grew since it was checked
                    raise _too_large(copied, field)
                digest.update(chunk)
                try:
                    out.write(chunk)
                except OSError as exc:
                    _raise_if_full(exc)
                    raise


def _check_digest(digest: str, clip: ClipRef, field: str) -> None:
    if digest != clip.sha256:
        raise NarrationError(
            codes.VOICE_FILE_MISMATCH,
            f"the file at {field}.path does not have the sha256 the request sent",
            field=f"{field}.sha256",
            details={"expected": clip.sha256, "actual": digest},
        )


def _check_wav(path: Path, max_seconds: float, field: str) -> None:
    """A WAV the service reads (soundfile), no longer than ``max_seconds``; else ``UNSUPPORTED_AUDIO``."""
    try:
        info = soundfile.info(str(path))
    except (RuntimeError, OSError, ValueError) as exc:  # soundfile.LibsndfileError is a RuntimeError
        raise NarrationError(
            codes.UNSUPPORTED_AUDIO,
            f"{field}.path is not an audio file the service reads",
            field=f"{field}.path",
            details={"reason": type(exc).__name__},
        ) from exc
    if info.format not in WAV_FORMATS:
        raise NarrationError(
            codes.UNSUPPORTED_AUDIO,
            f"{field}.path is a {info.format} file, not a WAV",
            field=f"{field}.path",
            details={"format": info.format},
        )
    seconds = info.frames / info.samplerate if info.samplerate else 0.0
    if seconds > max_seconds:
        raise NarrationError(
            codes.UNSUPPORTED_AUDIO,
            f"the clip is {seconds:.1f} s long; a voice clip is at most {max_seconds:g} s",
            field=f"{field}.path",
            details={"seconds": round(seconds, 3), "max_seconds": max_seconds},
        )


def _unreadable(clip: ClipRef, field: str, exc: OSError) -> NarrationError:
    return NarrationError(
        codes.VOICE_FILE_MISMATCH,
        f"the file at {field}.path cannot be read ({exc.strerror or type(exc).__name__})",
        field=f"{field}.path",
        hint="Check the path, and send the sha256 of the file that is there.",
    )


def _too_large(size: int, field: str) -> NarrationError:
    return NarrationError(
        codes.UNSUPPORTED_AUDIO,
        f"the file is {size} bytes; a voice clip is at most {MAX_CLIP_BYTES} bytes (20 MB)",
        field=f"{field}.path",
        details={"bytes": size, "max_bytes": MAX_CLIP_BYTES},
    )


def _raise_if_full(exc: OSError) -> None:
    if exc.errno == errno.ENOSPC:
        raise NarrationError(
            codes.STORE_FULL,
            "the store's disk filled while the clip was copied into it",
            retry_after_s=STORE_FULL_RETRY_S,
        ) from exc


__all__ = ["WAV_FORMATS", "ClipRef", "admit_clip", "check_synthetic", "refield"]
