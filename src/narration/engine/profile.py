"""Engine profiles: what they pin, how they are built and hashed, and which one is in use (design sections 6,
10.1 and 10.2; plan.md WP32).

An engine profile pins everything that can change a render's audio, so its ``hash`` can go into every render
key (section 10.2) and a caller can require it (``expect_engine_profile``). Two profiles are pinned: the Base
clone path (``qwen3-base-1.7b.pN``) and VoiceDesign (``qwen3-design-1.7b.pN``). A profile is built from:

- **the service's pins** (``models``): the model's repo and 40-hex revision;
- **the snapshot** under ``[server] models_root``: every file's sha256 (``weights``), and the ten sampling
  values of its ``generation_config.json``, whose ``max_new_tokens`` is the ceiling (DC-4). The pinned
  snapshots set all ten; a snapshot that leaves one out is refused rather than filled from qwen-tts's
  fallbacks, so no audio-changing value is ever a library default (section 10.1);
- **the worker project's ``uv.lock``**: its sha256, and the locked version of each package in
  ``QWEN_PACKAGES``, the packages whose versions can change Qwen's audio;
- **the configuration**: ``[engines.qwen3_base]`` or ``[engines.qwen3_design]`` (``non_streaming_mode`` and the
  per-call cap rule ``max_new_tokens_per_char`` and ``max_new_tokens_floor``, DC-4), and the pinned
  ``CUBLAS_WORKSPACE_CONFIG`` from ``[workers] env``;
- **the design's switches**: bf16 with ``sdpa`` attention, TF32 off, cuDNN deterministic with benchmark off,
  deterministic algorithms warn-only (section 10.1; ADR 0002), the capabilities, the licence and the VRAM the
  group needs (spike h: about 5.8 GB for a 30 s render).

**The hash** (``profile_hash``) is ``sha256:`` over RFC 8785 canonical JSON of every field except the five the
contract leaves out: ``hash``, ``snapshot_dir`` (a local path), ``observed`` (the GPU, driver, CUDA and cuDNN
the pin ran on), ``tier`` (the repeat test's outcome on this machine) and ``canary`` (made on this machine,
DC-3). The id is hashed, so a re-pin is a new id with a new hash. The object is built member by member
(``hashed_object``), never from the record's serialised form.

**The profile in use** is the store's (``current_profile``): ``narration-admin engine pin`` records it and
``engine repin`` replaces it. The configuration's ``[engines.*]`` keys take effect only through a pin, so a
render always runs as its profile says.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from narration import keys
from narration.config import Config
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Store
from narration.contracts.models import Determinism, EngineProfile, EngineRef
from narration.contracts.names import EngineKind
from narration.contracts.worker import HelloReply
from narration.workers.launch import WORKER_PROJECT_DIRS, worker_project

from .models import QWEN_BASE, QWEN_DESIGN, PinnedModel

# ======================================================================== the service's pins for Qwen

FAMILIES: Final[dict[EngineKind, str]] = {"base": "qwen3-base-1.7b", "design": "qwen3-design-1.7b"}
"""The id of a profile is its family, ``.p`` and its pin number: ``qwen3-base-1.7b.p1`` (section 6)."""
MODELS: Final[dict[EngineKind, PinnedModel]] = {"base": QWEN_BASE, "design": QWEN_DESIGN}
QWEN_PACKAGES: Final = (
    "qwen-tts",
    "transformers",
    "accelerate",
    "tokenizers",
    "torch",
    "torchaudio",
    "librosa",
    "soxr",
    "numpy",
    "soundfile",
)
"""The Qwen worker's packages whose versions can change its audio. Their locked versions are pinned in the
profile, and the worker's ``hello`` must report the same (``drift``). The worker reports each of them in its
fingerprint (``narration_qwen3tts.worker.Qwen3Handler.fingerprint_packages``; a test keeps the two agreed)."""
QWEN_ROLE: Final = "qwen3"
DTYPE: Final = "bfloat16"
ATTN_IMPLEMENTATION: Final = "sdpa"
DEFAULT_CUBLAS_WORKSPACE_CONFIG: Final = ":4096:8"
CUBLAS_ENV: Final = "CUBLAS_WORKSPACE_CONFIG"
QWEN_VRAM_MB: Final = 6000
"""The VRAM a Qwen group needs, in MB (section 4 item 2): the 5.3 GB peak of a 30 s render plus about 450 MB
of CUDA context, rounded up (KNOW, spike h: ``spikes/h-i-qwen-load``). An 8192-token render would need about
1.6 GB more (BELIEVE, not measured); the per-call cap (DC-4) makes one rare."""
CAPABILITIES: Final[dict[EngineKind, dict[str, Any]]] = {
    "base": {"controls": {"pace": False, "context": False, "instruct": False}, "ops": ["prepare_voice", "synthesize"]},
    "design": {"controls": {"pace": False, "context": False, "instruct": False}, "ops": ["design"]},
}
"""What each profile can do: Base has no instruction control (section 3.3); VoiceDesign designs voices."""
SAMPLING_KEYS: Final = (
    "do_sample",
    "top_k",
    "top_p",
    "temperature",
    "repetition_penalty",
    "subtalker_dosample",
    "subtalker_top_k",
    "subtalker_top_p",
    "subtalker_temperature",
    "max_new_tokens",
)
"""The ten sampling values qwen-tts 0.1.1 merges (``Qwen3TTSModel._merge_generate_kwargs``), each passed
explicitly (section 10.1). The worker refuses a ``load`` without any of them."""
GENERATION_CONFIG: Final = "generation_config.json"
UV_LOCK: Final = "uv.lock"
PROFILE_ID: Final = re.compile(r"(?P<family>[a-z0-9][a-z0-9.-]*)\.p(?P<n>[1-9][0-9]*)")
UNHASHED: Final = frozenset({"hash", "snapshot_dir", "observed", "tier", "canary"})
"""The ``EngineProfile`` fields its hash leaves out (the contract's docstring; the store keeps the same list)."""
INSTALL_HINT: Final = "Ask the operator to run narration-admin install, then narration-admin engine pin."
_CHUNK: Final = 8 << 20


class EngineSetupError(NarrationError):
    """What a profile is built from is missing or malformed: ``BACKEND_NOT_INSTALLED``, with what to do."""

    def __init__(self, message: str, *, hint: str = INSTALL_HINT, details: dict[str, Any] | None = None) -> None:
        super().__init__(codes.BACKEND_NOT_INSTALLED, message, hint=hint, details=details, retryable=False)


# ======================================================================== the parts a profile is built from


class FileHashes:
    """File sha256s, remembered by (size, modification time) so an unchanged file is not read again.

    A snapshot is about 4 GB, and reading it takes seconds, so the daemon keeps one of these for its lifetime:
    the first check after it starts reads every file, later ones only a file whose size or time changed (a
    replaced or rewritten file). Hashing is exact, never sampled.
    """

    def __init__(self) -> None:
        self._known: dict[str, tuple[int, int, str]] = {}

    def sha256(self, path: Path) -> str:
        """The file's sha256 (64 hex), read again only if its size or modification time changed."""
        stat = path.stat()
        key = os.path.normcase(os.path.abspath(path))
        known = self._known.get(key)
        if known is not None and known[0] == stat.st_size and known[1] == stat.st_mtime_ns:
            return known[2]
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(_CHUNK):
                digest.update(chunk)
        sha = digest.hexdigest()
        after = path.stat()
        if after.st_size == stat.st_size and after.st_mtime_ns == stat.st_mtime_ns:  # not changed while read
            self._known[key] = (stat.st_size, stat.st_mtime_ns, sha)
        return sha


def snapshot_files(snapshot: Path) -> list[str]:
    """Every file of a snapshot the profile pins, as sorted ``/``-separated paths relative to it: every regular
    file except hidden ones (a leading dot), which no install writes."""
    out: list[str] = []
    for path in snapshot.rglob("*"):
        rel = path.relative_to(snapshot)
        if any(part.startswith(".") for part in rel.parts) or not path.is_file():
            continue
        out.append(rel.as_posix())
    return sorted(out)


def hash_snapshot(snapshot: Path, hashes: FileHashes | None = None) -> dict[str, str]:
    """``weights``: every file of the snapshot (``snapshot_files``) and its sha256."""
    if not snapshot.is_dir():
        raise EngineSetupError(f"no model snapshot at {snapshot}", details={"snapshot_dir": str(snapshot)})
    hashes = hashes if hashes is not None else FileHashes()
    return {rel: hashes.sha256(snapshot / rel) for rel in snapshot_files(snapshot)}


def read_generation(snapshot: Path) -> dict[str, Any]:
    """The ten sampling values of the snapshot's ``generation_config.json``, all required (see the module
    docstring), each checked as the worker checks it."""
    path = snapshot / GENERATION_CONFIG
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EngineSetupError(f"cannot read {path}: {exc}", details={"file": str(path)}) from exc
    if not isinstance(data, dict):
        raise EngineSetupError(f"{path} does not hold a JSON object", details={"file": str(path)})
    missing = [k for k in SAMPLING_KEYS if k not in data]
    if missing:
        raise EngineSetupError(
            f"{path} leaves out {', '.join(missing)}; every sampling value must be pinned (design section 10.1)",
            details={"file": str(path), "missing": missing},
        )
    values = {k: data[k] for k in SAMPLING_KEYS}
    problem = _sampling_problem(values)
    if problem is not None:
        raise EngineSetupError(f"{path}: {problem}", details={"file": str(path)})
    return values


def _sampling_problem(values: Mapping[str, Any]) -> str | None:
    for key in ("do_sample", "subtalker_dosample"):
        if not isinstance(values[key], bool):
            return f"{key} must be true or false"
    for key in ("top_k", "subtalker_top_k", "max_new_tokens"):
        least = names.MIN_MAX_NEW_TOKENS if key == "max_new_tokens" else 1
        if isinstance(values[key], bool) or not isinstance(values[key], int) or values[key] < least:
            return f"{key} must be an integer of at least {least}"
    for key in ("top_p", "subtalker_top_p", "temperature", "repetition_penalty", "subtalker_temperature"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
            return f"{key} must be a finite number above 0"
        if key.endswith("top_p") and value > 1:
            return f"{key} must be at most 1"
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class LockFacts:
    """A worker project's ``uv.lock``: its sha256, and the locked version of each package, by name."""

    sha256: str
    versions: dict[str, str]


def read_lock(project: Path) -> LockFacts:
    """The worker project's ``uv.lock`` (``BACKEND_NOT_INSTALLED`` when it is missing or unreadable)."""
    path = project / UV_LOCK
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EngineSetupError(
            f"cannot read the worker's lock file {path}: {exc}", details={"file": str(path)}
        ) from exc
    versions: dict[str, str] = {}
    for package in data.get("package", []):
        if (
            isinstance(package, dict)
            and isinstance(package.get("name"), str)
            and isinstance(package.get("version"), str)
        ):
            versions.setdefault(package["name"], package["version"])
    return LockFacts(sha256=hashlib.sha256(raw).hexdigest(), versions=versions)


def lock_sha256(project: Path) -> str:
    """The sha256 of the worker project's ``uv.lock`` as it is now."""
    return read_lock(project).sha256


def qwen_project(config: Config) -> Path:
    """The Qwen worker's uv project (``[workers.qwen3] project``, else ``<service_root>/workers/qwen3tts``)."""
    project = worker_project(config, QWEN_ROLE)
    if project is None:
        raise EngineSetupError(
            "no project is configured for the qwen3 worker",
            hint="Set [workers.qwen3] project in the configuration, then run narration-admin install.",
        )
    return project


def determinism(config: Config) -> Determinism:
    """The determinism switches section 10.1 pins, with the configured ``CUBLAS_WORKSPACE_CONFIG``."""
    return Determinism(
        attn_implementation=ATTN_IMPLEMENTATION,
        tf32=False,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        deterministic_algorithms="warn_only",
        cublas_workspace_config=config.workers.env.get(CUBLAS_ENV) or DEFAULT_CUBLAS_WORKSPACE_CONFIG,
    )


def settings(config: Config, kind: EngineKind, generation: Mapping[str, Any]) -> dict[str, Any]:
    """The audio-changing settings a profile pins (``pins.qwen_load_payload`` and ``pins.call_cap`` read them)."""
    engine = config.engines.qwen3_base if kind == "base" else config.engines.qwen3_design
    return {
        "non_streaming_mode": engine.non_streaming_mode,
        "generation": dict(generation),
        "max_new_tokens_per_char": engine.max_new_tokens_per_char,
        "max_new_tokens_floor": engine.max_new_tokens_floor,
    }


# ======================================================================== building and hashing


def build_profile(
    config: Config,
    kind: EngineKind,
    *,
    engine_profile_id: str,
    model: PinnedModel | None = None,
    packages: Sequence[str] = QWEN_PACKAGES,
    hashes: FileHashes | None = None,
) -> EngineProfile:
    """The profile this installation would pin now for ``kind``, with its hash (see the module docstring).

    ``model`` defaults to the service's pin; ``packages`` to ``QWEN_PACKAGES``. Raises
    ``EngineSetupError`` (``BACKEND_NOT_INSTALLED``) when the snapshot, its ``generation_config.json``, the
    worker project's lock file or one of the packages in it is missing."""
    model = model if model is not None else MODELS[kind]
    snapshot = model.snapshot_dir(config.server.models_root)
    weights = hash_snapshot(snapshot, hashes)
    generation = read_generation(snapshot)
    project = qwen_project(config)
    lock = read_lock(project)
    missing = [p for p in packages if p not in lock.versions]
    if missing:
        raise EngineSetupError(
            f"the qwen3 worker's lock file ({project / UV_LOCK}) has no {', '.join(missing)}",
            hint="Check [workers.qwen3] project names the service's Qwen worker, then narration-admin install.",
            details={"missing": missing},
        )
    profile = EngineProfile(
        engine_profile_id=engine_profile_id,
        hash="",
        model_repo=model.repo,
        model_revision=model.revision,
        snapshot_dir=str(snapshot),
        weights=weights,
        worker_project=f"workers/{WORKER_PROJECT_DIRS[QWEN_ROLE]}",
        uv_lock_sha256=lock.sha256,
        packages={p: lock.versions[p] for p in packages},
        dtype=DTYPE,
        determinism=determinism(config),
        settings=settings(config, kind, generation),
        capabilities=dict(CAPABILITIES[kind]),
        licence=model.licence,
        vram_need_mb=QWEN_VRAM_MB,
    )
    return with_hash(profile)


def hashed_object(profile: EngineProfile) -> dict[str, Any]:
    """What a profile's hash covers, member by member: every field but ``UNHASHED``."""
    d = profile.determinism
    return {
        "schema": profile.schema,
        "engine_profile_id": profile.engine_profile_id,
        "model_repo": profile.model_repo,
        "model_revision": profile.model_revision,
        "weights": dict(profile.weights),
        "worker_project": profile.worker_project,
        "uv_lock_sha256": profile.uv_lock_sha256,
        "packages": dict(profile.packages),
        "dtype": profile.dtype,
        "determinism": {
            "attn_implementation": d.attn_implementation,
            "tf32": d.tf32,
            "cudnn_deterministic": d.cudnn_deterministic,
            "cudnn_benchmark": d.cudnn_benchmark,
            "deterministic_algorithms": d.deterministic_algorithms,
            "cublas_workspace_config": d.cublas_workspace_config,
        },
        "settings": dict(profile.settings),
        "capabilities": dict(profile.capabilities),
        "licence": profile.licence,
        "vram_need_mb": profile.vram_need_mb,
    }


def profile_hash(profile: EngineProfile) -> str:
    """``sha256:`` + 64 hex over the canonical JSON of ``hashed_object``."""
    return keys.hash_key(hashed_object(profile))


def with_hash(profile: EngineProfile) -> EngineProfile:
    """The profile with its ``hash`` computed."""
    return dataclasses.replace(profile, hash=profile_hash(profile))


def pin_differences(pinned: EngineProfile, built: EngineProfile) -> tuple[str, ...]:
    """The hashed fields in which two profiles differ, the id aside: what a re-pin would change. Empty when
    the installation is still exactly what ``pinned`` pinned."""
    a, b = hashed_object(pinned), hashed_object(built)
    a.pop("engine_profile_id")
    b.pop("engine_profile_id")
    return tuple(sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k)))


def next_profile_id(existing: Sequence[str], kind: EngineKind) -> str:
    """The id of the next pin of ``kind``: its family with the next free number (``qwen3-base-1.7b.p1``,
    then ``.p2`` …)."""
    family = FAMILIES[kind]
    numbers = [
        int(m.group("n")) for ident in existing if (m := PROFILE_ID.fullmatch(ident)) and m.group("family") == family
    ]
    return f"{family}.p{max(numbers, default=0) + 1}"


def observed(hello: HelloReply | None) -> dict[str, Any]:
    """What the worker observed of the machine, recorded with the pin and never hashed: the GPU, the driver,
    CUDA, cuDNN and Python."""
    if hello is None:
        return {}
    fingerprint: Mapping[str, Any] = hello.get("fingerprint") or {}
    return {k: fingerprint[k] for k in ("gpu", "driver", "cuda", "cudnn", "python") if fingerprint.get(k) is not None}


# ======================================================================== the profile in use


def current_profile(store: Store, kind: EngineKind = "base") -> EngineProfile:
    """The engine profile in use for ``kind`` (``base``: every take; ``design``: ``design_voice``).

    This is what the render keys, measurements and ``expect_engine_profile`` use (sections 7.3, 10.1, 10.2).
    Raises ``NarrationError(BACKEND_NOT_INSTALLED)`` before ``narration-admin engine pin``."""
    profile = store.current_engine_profile(kind)
    if profile is None:
        raise EngineSetupError(f"no engine profile is pinned for the {kind} engine ({FAMILIES[kind]})")
    return profile


def current_ref(store: Store, kind: EngineKind = "base") -> EngineRef:
    """The engine profile in use, as results report it: ``{id, hash}``."""
    profile = current_profile(store, kind)
    return EngineRef(id=profile.engine_profile_id, hash=profile.hash)


def require_expected(store: Store, expected: str | None, kind: EngineKind = "base") -> EngineProfile:
    """The profile in use, checked against a request's ``expect_engine_profile`` (sections 7.3, 10.1): a
    different hash is ``ENGINE_CHANGED`` (not retryable), with both hashes in ``details``."""
    profile = current_profile(store, kind)
    if expected is not None and expected != profile.hash:
        raise NarrationError(
            codes.ENGINE_CHANGED,
            f"the request expects engine profile {expected}; the service's is {profile.hash}",
            field="expect_engine_profile",
            details={"expected": expected, "current": profile.hash, "engine_profile_id": profile.engine_profile_id},
        )
    return profile


__all__ = [
    "ATTN_IMPLEMENTATION",
    "CAPABILITIES",
    "DTYPE",
    "FAMILIES",
    "MODELS",
    "QWEN_PACKAGES",
    "QWEN_VRAM_MB",
    "SAMPLING_KEYS",
    "UNHASHED",
    "EngineSetupError",
    "FileHashes",
    "LockFacts",
    "build_profile",
    "current_profile",
    "current_ref",
    "determinism",
    "hash_snapshot",
    "hashed_object",
    "lock_sha256",
    "next_profile_id",
    "observed",
    "pin_differences",
    "profile_hash",
    "qwen_project",
    "read_generation",
    "read_lock",
    "require_expected",
    "settings",
    "snapshot_files",
    "with_hash",
]
