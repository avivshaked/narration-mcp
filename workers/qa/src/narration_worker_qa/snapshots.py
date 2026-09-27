"""The QA worker's model snapshots: the ``load`` request's ``models`` and the checks made before a model loads
(design section 4; Appendix A ``load``; plan.md WP22).

Every model loads offline from a local snapshot directory named by its 40-hex commit revision (section 4). The
references are checked by the functions the fake uses (``narration_worker.snapshots``), in the same order and
with the same codes and ``details.field`` as the fake and the ``qwen3`` worker: each reference's shape and
absolute ``snapshot_dir``; then every directory's presence, before anything else; then each revision naming its
directory. Then each use must be one of ``USES`` (``INVALID_REQUEST``, ``models.<use>``).

A directory whose files are missing or damaged, or that holds another model (its ``config.json`` names another
architecture), is ``BACKEND_NOT_INSTALLED`` with the hint to install the models again: the directory the pinned
revision names holds the wrong files, which is an install fault, the same rule the aligner follows (WP15).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from narration_worker.snapshots import by_use, check_ref, check_snapshots

from .align import INSTALL_HINT, BackendMissing, InvalidRequest, load_error

USES: Final = ("asr", "sv", "aligner")
"""The models a QA ``load`` may name, by use: Whisper, WavLM-SV and the CTC aligner."""


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
    """The ``models`` member of a ``load`` request, in the request's order, after the shared checks (module
    docstring); ``OpError`` from ``narration_worker.snapshots``, or ``InvalidRequest`` when ``models`` is missing.

    ``models`` may be empty, a load of no model, after which only ``f0`` and ``profile`` run.
    """
    if "models" not in request:
        raise InvalidRequest("the QA worker's load needs models {asr?, sv?, aligner?}", {"field": "models"})
    refs = [(field, check_ref(field, ref)) for field, ref in by_use(request["models"])]
    check_snapshots(refs)
    return [
        Snapshot(field.removeprefix("models."), ref["repo"], ref["revision"], Path(ref["snapshot_dir"]))
        for field, ref in refs
    ]


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
