"""The QA worker's model snapshots: the ``load`` request's ``models`` and the checks made before a model loads
(design section 4; Appendix A ``load``; plan.md WP22).

Every model loads offline from a local snapshot directory named by its 40-hex commit revision (section 4). The
checks run in the order the ``qwen3`` worker and the fake run them, with the same codes and ``details.field``
(``narration_worker.fake``), one pass over the references in the request's order per step:

1. each reference is an object whose ``repo``, ``revision`` and ``snapshot_dir`` are strings (``INVALID_REQUEST``,
   ``models.<use>``), and ``snapshot_dir`` is an absolute path (``models.<use>.snapshot_dir``);
2. every snapshot directory exists (``BACKEND_NOT_INSTALLED``, ``models.<use>``), before anything else;
3. each revision is a 40-hex SHA (``models.<use>.revision``) that names its directory
   (``models.<use>.snapshot_dir``);
4. each use is one of ``USES`` (``models.<use>``).

A directory whose files are missing or damaged, or that holds another model (its ``config.json`` names another
architecture), is ``BACKEND_NOT_INSTALLED`` with the hint to install the models again: the directory the pinned
revision names holds the wrong files, which is an install fault, the same rule the aligner follows (WP15).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from .align import INSTALL_HINT, BackendMissing, InvalidRequest, load_error

USES: Final = ("asr", "sv", "aligner")
"""The models a QA ``load`` may name, by use: Whisper, WavLM-SV and the CTC aligner."""
REVISION_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
DEVICE_PATTERN: Final = re.compile(r"cpu|cuda(:\d+)?")


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One ``ModelRef`` of a ``load`` request, checked: its use, repo, revision and directory."""

    use: str
    repo: str
    revision: str
    path: Path

    @property
    def field(self) -> str:
        return f"models.{self.use}"


def parse_models(request: Mapping[str, Any]) -> list[Snapshot]:
    """The ``models`` member of a ``load`` request, in the request's order, after step 1 of the module docstring.

    ``models`` itself must be an object (``INVALID_REQUEST``, ``models``); it may be empty, a load of no model,
    after which only ``f0`` and ``profile`` run.
    """
    if "models" not in request:
        raise InvalidRequest("the QA worker's load needs models {asr?, sv?, aligner?}", {"field": "models"})
    models = request["models"]
    if not isinstance(models, dict):
        raise InvalidRequest("models must be an object naming asr, sv and aligner", {"field": "models"})
    out: list[Snapshot] = []
    for use, ref in cast(dict[str, object], models).items():
        field = f"models.{use}"
        if not isinstance(ref, dict) or not all(
            isinstance(cast(dict[str, object], ref).get(k), str) for k in ("repo", "revision", "snapshot_dir")
        ):
            raise InvalidRequest(f"{field} must have repo, revision and snapshot_dir", {"field": field})
        ref = cast(dict[str, str], ref)
        path = Path(ref["snapshot_dir"])
        if not path.is_absolute():
            raise InvalidRequest(f"{field}.snapshot_dir must be an absolute path", {"field": f"{field}.snapshot_dir"})
        out.append(Snapshot(str(use), ref["repo"], ref["revision"], path))
    return out


def check_present(snapshot: Snapshot) -> None:
    """``BACKEND_NOT_INSTALLED`` when the snapshot directory does not exist (checked before anything else)."""
    if not snapshot.path.is_dir():
        raise BackendMissing(
            f"no snapshot of {snapshot.repo} at {snapshot.path}; {INSTALL_HINT}",
            {"field": snapshot.field, "repo": snapshot.repo, "snapshot_dir": str(snapshot.path)},
        )


def check_named(snapshot: Snapshot) -> None:
    """``INVALID_REQUEST`` unless the revision is a 40-hex SHA that names the directory (section 4)."""
    if REVISION_PATTERN.fullmatch(snapshot.revision) is None:
        field = f"{snapshot.field}.revision"
        raise InvalidRequest(f"{field} must be a 40-hex commit SHA", {"field": field})
    if snapshot.path.name != snapshot.revision:
        raise InvalidRequest(
            "the snapshot folder must be named by its revision (section 4)",
            {"field": f"{snapshot.field}.snapshot_dir", "revision": snapshot.revision, "folder": snapshot.path.name},
        )


def check_use(snapshot: Snapshot) -> None:
    """``INVALID_REQUEST`` for a use other than ``USES``."""
    if snapshot.use not in USES:
        raise InvalidRequest(f"models may name only {', '.join(USES)}, not {snapshot.use!r}", {"field": snapshot.field})


def read_config(path: Path, what: str) -> dict[str, Any]:
    """A snapshot's ``config.json``; ``BackendMissing`` (with the hint) when it is missing, unreadable or not an
    object."""
    config_path = path / "config.json"
    if not config_path.is_file():
        raise BackendMissing(
            f"the {what} model's config.json is missing: {config_path}; {INSTALL_HINT}", {"path": str(config_path)}
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise load_error(
            exc, f"the {what} model's config.json cannot be read: {config_path}", {"path": str(config_path)}
        ) from exc
    if not isinstance(config, dict):
        raise BackendMissing(
            f"the {what} model's config.json is not an object; {INSTALL_HINT}", {"path": str(config_path)}
        )
    return config


def check_architecture(path: Path, architecture: str, model_type: str, what: str) -> dict[str, Any]:
    """The snapshot's ``config.json``, after checking that it is ``model_type`` with ``architecture`` among its
    ``architectures``; ``BackendMissing`` otherwise, so another model's snapshot never loads as this one."""
    config = read_config(path, what)
    architectures = config.get("architectures")
    if (
        config.get("model_type") != model_type
        or not isinstance(architectures, list)
        or architecture not in architectures
    ):
        raise BackendMissing(
            f"the {what} snapshot is not a {architecture} model (model_type {config.get('model_type')!r}, "
            f"architectures {architectures!r}); {INSTALL_HINT}",
            {"path": str(path / "config.json"), "model_type": config.get("model_type"), "architectures": architectures},
        )
    return config
