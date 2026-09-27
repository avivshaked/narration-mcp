"""The worker fingerprint check: is the engine that loaded the one the profile pinned? (design sections 6,
10.1 and 14; plan.md WP32)

After every Qwen load, before the canary, the daemon compares what it can observe with the pinned profile:

- **the worker's ``hello``** (Appendix A): the version of every package the profile pins (``packages``) and
  the ``CUBLAS_WORKSPACE_CONFIG`` the worker runs with (section 10.1);
- **the worker project's ``uv.lock``**, by sha256 (``uv_lock_sha256``): a changed lock is drift even before
  the venv is synced to it;
- **the snapshot's files**, by sha256 (``weights``): a changed, missing or added file is drift. Files are
  read again only when their size or modification time changed (``profile.FileHashes``), so only the first
  check in a daemon's life reads the whole snapshot. An unfinished download (``*.partial``, which an
  interrupted install leaves) is drift of its own kind (``download``), whose message says to run the install
  again.

Any difference fails the job with ``ENGINE_DRIFT`` (not retryable) before anything renders; ``details.drift``
lists each difference (``what``, ``name``, ``pinned``, ``found``). The GPU, driver, CUDA and cuDNN are
observed, never pinned: the canary gate (``canary``) is what notices a change there.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import EngineProfile
from narration.contracts.worker import HelloReply

from .profile import CUBLAS_ENV, UV_LOCK, FileHashes, partial_files, snapshot_files

DriftKind = Literal["fingerprint", "package", "env", "uv_lock", "weights", "download"]
MAX_LISTED: Final = 20
"""The most differences an ``ENGINE_DRIFT`` error lists (the count is always given)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Drift:
    """One difference between the pinned profile and what was found. ``None`` means absent."""

    what: DriftKind
    name: str
    pinned: Any
    found: Any

    def as_dict(self) -> dict[str, Any]:
        """The difference as ``ENGINE_DRIFT``'s ``details.drift`` lists it."""
        return {"what": self.what, "name": self.name, "pinned": self.pinned, "found": self.found}


class DriftCheck:
    """Compares a loaded engine with its pinned profile (see the module docstring). One instance serves one
    daemon, and remembers file hashes by size and modification time between checks."""

    def __init__(self, hashes: FileHashes | None = None) -> None:
        self.hashes = hashes if hashes is not None else FileHashes()

    def fingerprint(self, profile: EngineProfile, hello: HelloReply | None) -> list[Drift]:
        """What the worker's ``hello`` shows that differs from the profile: package versions and the cuBLAS pin."""
        if hello is None:
            return [Drift(what="fingerprint", name="hello", pinned="reported", found=None)]
        fingerprint: Mapping[str, Any] = hello.get("fingerprint") or {}
        packages: Mapping[str, Any] = fingerprint.get("packages") or {}
        env: Mapping[str, Any] = fingerprint.get("env") or {}
        out = [
            Drift(what="package", name=name, pinned=version, found=packages.get(name))
            for name, version in sorted(profile.packages.items())
            if packages.get(name) != version
        ]
        cublas = profile.determinism.cublas_workspace_config
        if env.get(CUBLAS_ENV) != cublas:
            out.append(Drift(what="env", name=CUBLAS_ENV, pinned=cublas, found=env.get(CUBLAS_ENV)))
        return out

    def lock(self, profile: EngineProfile, project: Path) -> list[Drift]:
        """The worker project's ``uv.lock`` against ``uv_lock_sha256``."""
        path = project / UV_LOCK
        found = self.hashes.sha256(path) if path.is_file() else None
        if found == profile.uv_lock_sha256:
            return []
        return [Drift(what="uv_lock", name=UV_LOCK, pinned=profile.uv_lock_sha256, found=found)]

    def weights(self, profile: EngineProfile) -> list[Drift]:
        """The snapshot's files against ``weights``: unfinished downloads first, then changed, missing and
        added files."""
        snapshot = Path(profile.snapshot_dir)
        present = set(snapshot_files(snapshot)) if snapshot.is_dir() else set()
        out = [Drift(what="download", name=rel, pinned=None, found="unfinished") for rel in partial_files(snapshot)]
        for rel in sorted(present | set(profile.weights)):
            pinned = profile.weights.get(rel)
            found = self.hashes.sha256(snapshot / rel) if rel in present else None
            if found != pinned:
                out.append(Drift(what="weights", name=rel, pinned=pinned, found=found))
        return out

    def find(self, profile: EngineProfile, hello: HelloReply | None, *, project: Path) -> list[Drift]:
        """Every difference: the fingerprint first, then the lock, then the snapshot's files."""
        return [*self.fingerprint(profile, hello), *self.lock(profile, project), *self.weights(profile)]

    def require_none(self, profile: EngineProfile, hello: HelloReply | None, *, project: Path) -> None:
        """Raise ``ENGINE_DRIFT`` (``drift_error``) if ``find`` finds anything."""
        found = self.find(profile, hello, project=project)
        if found:
            raise drift_error(profile, found)


def drift_error(profile: EngineProfile, found: list[Drift]) -> NarrationError:
    """``ENGINE_DRIFT`` for these differences, naming the first few and how many there are. An unfinished
    download is named as such, with the hint to finish the install."""
    unfinished = [d.name for d in found if d.what == "download"]
    hint: str | None = None
    if unfinished:
        message = (
            f"the model snapshot of engine profile {profile.engine_profile_id} holds an unfinished download "
            f"({', '.join(unfinished[:3])}{', ...' if len(unfinished) > 3 else ''}): an install was interrupted"
        )
        hint = (
            "Ask the operator to run narration-admin install again, which finishes the download or removes it; "
            "nothing was rendered."
        )
    else:
        first = found[0]
        what = f"{first.what} {first.name}" + (f" and {len(found) - 1} more" if len(found) > 1 else "")
        message = f"the loaded engine is not engine profile {profile.engine_profile_id} as pinned: {what} differ(s)"
        hint = (
            f"Ask the operator to restore the installation engine profile {profile.engine_profile_id} pinned "
            "(narration-admin install syncs the workers and the models), or to make a new profile the one in use "
            "(narration-admin engine repin; new render keys); nothing was rendered."
        )
    return NarrationError(
        codes.ENGINE_DRIFT,
        message,
        hint=hint,
        details={
            "engine_profile_id": profile.engine_profile_id,
            "count": len(found),
            "drift": [d.as_dict() for d in found[:MAX_LISTED]],
        },
        retryable=False,
    )


__all__ = ["MAX_LISTED", "Drift", "DriftCheck", "DriftKind", "drift_error"]
