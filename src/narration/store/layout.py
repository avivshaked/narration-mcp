r"""Where things live in the store (design section 15), and write confinement (section 17.2).

Every path is built here, by the store, from ids that are checked against a strict grammar first; no path
a caller sends is ever written to. A built path is then *confined*: its ``realpath`` (which follows
symlinks and junctions) must stay under the store root's ``realpath``, and the platform's
``check_store_path`` (``narration.platform``, WP19) refuses Windows reserved names and reparse points.
Both ``realpath``s come from ``narration.platform.real_path``, which drops a ``\\?\`` prefix Windows can
leave on a file being replaced. The OS-specific rules live only in the platform package; this module is
portable.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Final

from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Platform
from narration.keys import HEX64_PATTERN, KEY_PATTERN
from narration.platform import real_path

# ---------------------------------------------------------------- file and folder names (section 15)
DB_NAME: Final = "narration.sqlite"
PROVENANCE_NAME: Final = "provenance.jsonl"
RENDERS: Final = "renders"
TAKES: Final = "takes"
ANALYSES: Final = "analyses"
MEASUREMENTS: Final = "measurements"
PROFILES: Final = "profiles"
DESIGNS: Final = "designs"
JOBS: Final = "jobs"
ENGINES: Final = "engines"
ALIGNMENT: Final = "alignment"
SCRATCH: Final = "scratch"
LOGS: Final = "logs"
RUN: Final = "run"

RAW_WAV: Final = "raw.wav"
RENDER_JSON: Final = "render.json"
DELIVERY_WAV: Final = "delivery.wav"
TAKE_JSON: Final = "take.json"
MEASUREMENT_JSON: Final = "measurement.json"
CLIP_WAV: Final = "clip.wav"
CANDIDATE_JSON: Final = "candidate.json"
CANDIDATE_PROFILE_DIR: Final = "profile"
PROFILE_JSON: Final = "profile.json"
SPECTROGRAM_PNG: Final = "spectrogram.png"
PITCH_PNG: Final = "pitch.png"
JOB_JSON: Final = "job.json"
DAEMON_JSON: Final = "daemon.json"
CANARY_WAV: Final = "canary.wav"

CACHE_TREES: Final = (RENDERS, TAKES, MEASUREMENTS, PROFILES, DESIGNS, JOBS)
"""The folders ``gc`` may collect from (after retention). ``engines``, ``alignment`` and
``provenance.jsonl`` are the service's own records and are never collected."""

TEMP_PREFIXES: Final = (".tmp-", ".staging-", ".trash-")
"""Names the store gives work in progress. Nothing with these prefixes is ever a published file."""

# ---------------------------------------------------------------- id grammar
# The ids a caller sends back take their shapes from ``names.ID_PATTERNS``, which the front end and the
# tool schemas use too. Every pattern here is unanchored and used with ``fullmatch`` only: ``$`` (or a
# ``match``) would let an id with a trailing newline through to a file name.
RENDER_ID_PATTERN: Final = re.compile(names.ID_PATTERNS["render_id"])
TAKE_ID_PATTERN: Final = re.compile(names.ID_PATTERNS["take_id"])
ANALYSIS_ID_PATTERN: Final = re.compile(names.ID_PATTERNS["analysis_id"])
JOB_ID_PATTERN: Final = re.compile(names.ID_PATTERNS["job_id"])
DESIGN_ID_PATTERN: Final = re.compile(names.ID_PATTERNS["design_id"])
PROFILE_ID_PATTERN: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}(?<!\.)")
"""An engine profile id such as ``qwen3-base-1.7b.p1`` (section 7.2's Id, without a trailing dot, which
Windows would drop from a file name)."""
COMPONENT_MAX: Final = 128
"""The longest path component the store builds. Ids are far shorter; an alignment method id can be longer,
and ``method_file_stem`` shortens it to fit."""
_COMPONENT: Final = re.compile(r"[A-Za-z0-9_%+-][A-Za-z0-9._%+-]{0," + str(COMPONENT_MAX - 1) + r"}(?<!\.)")
"""One path component the store itself made: no separators, no drive, not ``.`` or ``..``, no trailing dot."""
ALIGNMENT_SUFFIX: Final = ".json"
METHOD_STEM_MAX: Final = COMPONENT_MAX - len(ALIGNMENT_SUFFIX)
"""The longest stem ``method_file_stem`` returns, so that ``<stem>.json`` is one valid component."""
_METHOD_HASH_MARK: Final = "%h"
"""Separates a shortened stem's readable prefix from its hash. An encoded stem never contains it: every
``%`` there starts an upper-case hex escape (``%XX``), and ``h`` or ``H`` is not a hex digit."""
_METHOD_HASH_HEX: Final = 24
_METHOD_SAFE: Final = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._+-")


class InvalidIdError(NarrationError):
    """An id that does not match its grammar (``INVALID_ARGUMENT``); no path is built from it."""

    def __init__(self, what: str, value: object) -> None:
        super().__init__(
            codes.INVALID_ARGUMENT,
            f"{what} {value!r} is not a valid id",
            hint="Send the id exactly as the service returned it.",
        )


class StorePathError(NarrationError):
    """A store path that would leave the store root, or that the platform refuses (section 17.2)."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            codes.PATH_NOT_ALLOWED,
            message,
            hint="The store root must be a plain local folder with no links or junctions inside it; "
            "ask the operator to check it (narration-admin verify).",
            details=details,
        )


def _check(pattern: re.Pattern[str], value: str, what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise InvalidIdError(what, value)
    return value


def shard(ident: str) -> str:
    """The two-character shard folder (``<ab>``) of a content id or hash: its first two hex digits."""
    hex_part = ident.split("_", 1)[1] if "_" in ident else ident
    return hex_part[:2]


def method_file_stem(method_id: str) -> str:
    """A file name stem for an alignment method id, which may contain ``/``, ``@``, ``{`` or capitals.

    Lower-case letters, digits and ``._+-`` are kept, except a leading ``.``; every other byte of its
    UTF-8 form is written ``%XX``. That form is reversible, so two ids never collide, even on a
    case-insensitive file system.

    A stem longer than ``METHOD_STEM_MAX`` (a method id with a long model name, revision and snap
    parameters) is shortened: its first characters, cut so that no ``%XX`` is split, then ``%h`` and the
    first 24 hex digits of the sha256 of the id. The result is stable and still names the method to a
    reader. A short stem never contains ``%h``, so a shortened stem never equals a short one.
    """
    if not isinstance(method_id, str) or not method_id:
        raise InvalidIdError("method_id", method_id)
    out: list[str] = []
    for i, char in enumerate(method_id):
        if char in _METHOD_SAFE and not (i == 0 and char == "."):
            out.append(char)
        else:
            out.extend(f"%{b:02X}" for b in char.encode("utf-8"))
    stem = "".join(out)
    if len(stem) <= METHOD_STEM_MAX:
        return stem
    digest = hashlib.sha256(method_id.encode("utf-8")).hexdigest()[:_METHOD_HASH_HEX]
    cut = METHOD_STEM_MAX - len(_METHOD_HASH_MARK) - _METHOD_HASH_HEX
    percent = stem.rfind("%", max(0, cut - 2), cut)
    if percent != -1:
        cut = percent  # never keep half of a %XX escape
    return stem[:cut] + _METHOD_HASH_MARK + digest


def is_under(path: PurePath, root: PurePath) -> bool:
    """Whether ``path`` is ``root`` or inside it (both already resolved). The comparison is on the text of
    the names, so a path with a ``..`` in it (``D:\\root\\..\\..\\x`` has ``D:\\root`` among its parents) is
    never inside."""
    if ".." in path.parts or ".." in root.parts:
        return False
    return path == root or root in path.parents


class StoreLayout:
    """Builds and confines every path of one store (section 15).

    ``root`` is made absolute (not resolved: an operator may reach the store through a mapped drive) and
    created if missing. Confinement compares ``realpath``s, so a link or junction that leads out of the
    root is refused wherever it sits.
    """

    def __init__(self, root: Path, platform: Platform) -> None:
        self._root = Path(os.path.abspath(root))
        self._root.mkdir(parents=True, exist_ok=True)
        self._real_root = Path(real_path(self._root))
        self._platform = platform

    @property
    def root(self) -> Path:
        return self._root

    # ---- confinement
    def confine(self, path: Path) -> Path:
        """Return ``path`` if it is safe to write: under the root after resolving links, and accepted by
        the platform. Raises ``StorePathError`` otherwise.

        Only the platform's ``PATH_NOT_ALLOWED`` refusal becomes a ``StorePathError`` (keeping its
        ``details``). Any other error from the platform is raised unchanged: on an OS v1 does not support,
        ``UnsupportedPlatform`` stays ``DAEMON_UNAVAILABLE``.
        """
        real = Path(real_path(path))
        if not is_under(real, self._real_root):
            raise StorePathError(f"{path} resolves to {real}, outside the store root {self._real_root}")
        try:
            self._platform.check_store_path(path, self._root)
        except NarrationError as exc:
            if exc.code != codes.PATH_NOT_ALLOWED:
                raise
            raise StorePathError(f"{path} is not allowed in the store: {exc.message}", details=exc.details) from exc
        return path

    def _build(self, *parts: str) -> Path:
        for part in parts:
            if not isinstance(part, str) or not _COMPONENT.fullmatch(part):
                raise StorePathError(f"refusing to build a store path with the component {part!r}")
        return self.confine(self._root.joinpath(*parts))

    def rel(self, path: Path) -> str:
        """``path`` relative to the root, with ``/`` separators: the form stored in sidecars and rows, so
        that a store can be moved."""
        return Path(os.path.abspath(path)).relative_to(self._root).as_posix()

    def abs(self, rel: str) -> Path:
        """The confined absolute path of a stored relative path."""
        parts = PurePosixPath(rel).parts
        if not parts or PurePosixPath(rel).is_absolute():
            raise StorePathError(f"not a relative store path: {rel!r}")
        return self._build(*parts)

    # ---- the layout of section 15
    @property
    def db_path(self) -> Path:
        return self._build(DB_NAME)

    @property
    def provenance_path(self) -> Path:
        return self._build(PROVENANCE_NAME)

    def render_dir(self, render_id: str) -> Path:
        _check(RENDER_ID_PATTERN, render_id, "render_id")
        return self._build(RENDERS, shard(render_id), render_id)

    def take_dir(self, take_id: str) -> Path:
        _check(TAKE_ID_PATTERN, take_id, "take_id")
        return self._build(TAKES, shard(take_id), take_id)

    def analysis_path(self, take_id: str, analysis_id: str) -> Path:
        _check(TAKE_ID_PATTERN, take_id, "take_id")
        _check(ANALYSIS_ID_PATTERN, analysis_id, "analysis_id")
        return self._build(TAKES, shard(take_id), take_id, ANALYSES, f"{analysis_id}.json")

    def measurement_dir(self, voice_hash: str, engine_profile_id: str) -> Path:
        """``measurements/<64 hex of voice_hash>/<engine_profile_id>/``: the ``sha256:`` prefix is left
        out of the folder name (a colon is not allowed in a Windows file name)."""
        _check(KEY_PATTERN, voice_hash, "voice_hash")
        _check(PROFILE_ID_PATTERN, engine_profile_id, "engine_profile_id")
        return self._build(MEASUREMENTS, voice_hash[len(names.HASH_PREFIX) :], engine_profile_id)

    def measurements_root(self, voice_hash: str) -> Path:
        _check(KEY_PATTERN, voice_hash, "voice_hash")
        return self._build(MEASUREMENTS, voice_hash[len(names.HASH_PREFIX) :])

    def design_dir(self, design_id: str, index: int) -> Path:
        _check(DESIGN_ID_PATTERN, design_id, "design_id")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 1000:
            raise InvalidIdError("candidate index", index)
        return self._build(DESIGNS, design_id, str(index))

    def design_root(self, design_id: str) -> Path:
        _check(DESIGN_ID_PATTERN, design_id, "design_id")
        return self._build(DESIGNS, design_id)

    def profile_dir(self, audio_sha256: str) -> Path:
        _check(HEX64_PATTERN, audio_sha256, "audio_sha256")
        return self._build(PROFILES, shard(audio_sha256), audio_sha256)

    def job_dir(self, job_id: str) -> Path:
        _check(JOB_ID_PATTERN, job_id, "job_id")
        return self._build(JOBS, job_id)

    def engine_path(self, engine_profile_id: str) -> Path:
        _check(PROFILE_ID_PATTERN, engine_profile_id, "engine_profile_id")
        return self._build(ENGINES, f"{engine_profile_id}.json")

    def canary_clip_path(self, engine_profile_id: str) -> Path:
        """``engines/<engine_profile_id>/canary.wav`` (DC-3): immutable and never collected."""
        _check(PROFILE_ID_PATTERN, engine_profile_id, "engine_profile_id")
        return self._build(ENGINES, engine_profile_id, CANARY_WAV)

    def alignment_path(self, method_id: str) -> Path:
        return self._build(ALIGNMENT, method_file_stem(method_id) + ALIGNMENT_SUFFIX)

    def daemon_json_path(self) -> Path:
        return self._build(RUN, DAEMON_JSON)

    def logs_dir(self) -> Path:
        return self._build(LOGS)

    def scratch_path(self, *parts: str) -> Path:
        """A path under ``scratch/`` for worker I/O; its parent folder is created."""
        if not parts:
            raise StorePathError("scratch_path needs at least one component")
        path = self._build(SCRATCH, *parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return self.confine(path)

    def tree(self, name: str) -> Path:
        """One of the top-level folders (``renders``, ``takes``, …)."""
        return self._build(name)
