"""The models root's install record, and checking the files in it (design sections 4, 15 and 17.8).

A pinned model lives in ``<models_root>/models--<org>--<name>/snapshots/<revision>/``, a folder named by the
commit it holds (section 4). ``narration-admin install`` records every file it put there, with the sha256 it
checked against the hash Hugging Face publishes for that revision, in ``<models_root>/manifest.json``::

    {"<repo>@<revision>": {"repo": ..., "revision": ..., "snapshot_dir": "models--<org>--<name>/snapshots/<revision>",
                          "files_sha256": {"<path in the snapshot>": "<sha256>", ...}}, ...}

(``tools/dev_models.py`` writes the same file for a developer's models root.) ``doctor`` and ``verify`` read it
and hash the files again, offline, so a file that changed or went missing since the install is found.

The pins (which revision of which repo) are the engine's (``narration.engine.models``, WP32); until that module
is in this build, ``pinned_revisions`` returns None and the checks say the pins are unknown.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .cli import names_the_module

MANIFEST_NAME: Final = "manifest.json"
CHUNK: Final = 8 * 1024 * 1024
PINS_MODULE: Final = "narration.engine.models"


@dataclass(frozen=True, slots=True, kw_only=True)
class InstalledModel:
    """One model as the install record lists it."""

    repo: str
    revision: str
    snapshot_dir: Path
    files_sha256: Mapping[str, str]

    @property
    def key(self) -> str:
        """``<repo>@<revision>``, the record's key."""
        return f"{self.repo}@{self.revision}"


def safe_relative(path: str) -> str:
    """``path`` if it names a file inside a folder: a relative POSIX path with no empty, ``.`` or ``..`` part,
    and no ``\\`` or ``:`` (which Windows reads as a separator or a drive). ``ValueError`` otherwise."""
    parts = path.split("/")
    if not path or "\\" in path or ":" in path or "\0" in path or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"{path!r} is not a file name inside the model's folder")
    return path


def inside(folder: Path, path: str, where: str) -> Path:
    """``folder / path``, once ``path`` is ``safe_relative`` and the result resolves inside ``folder``;
    ``ValueError`` naming ``where`` otherwise. No file named by a Hub listing or an install record is ever
    written or read outside its folder."""
    try:
        target = folder / safe_relative(path)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    if not target.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f"{where}: {path!r} resolves outside the model's folder")
    return target


def snapshot_dir(models_root: Path, repo: str, revision: str) -> Path:
    """``<models_root>/models--<org>--<name>/snapshots/<revision>`` (section 4)."""
    return models_root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def pinned_revisions() -> dict[str, str] | None:
    """The engine's pins, repo to revision (``narration.engine.models.PINNED``), or None when that module is
    not in this build. An ``ImportError`` means the module is there but cannot be loaded."""
    try:
        module = importlib.import_module(PINS_MODULE)
    except ModuleNotFoundError as exc:
        if names_the_module(exc, PINS_MODULE):
            return None
        raise  # the module is in this build, but a dependency of it is missing
    pinned: Mapping[str, Any] = module.PINNED
    return {str(repo): str(model.revision) for repo, model in pinned.items()}


def read_manifest(models_root: Path) -> dict[str, InstalledModel]:
    """The install record, by ``<repo>@<revision>``; empty when there is none. ``ValueError`` when it is not
    one (the message names what is wrong)."""
    path = models_root / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not an install record (not a JSON object)")
    records: dict[str, InstalledModel] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {key!r} is not a record")
        try:
            files = entry["files_sha256"]
            repo, revision, rel = str(entry["repo"]), str(entry["revision"]), str(entry["snapshot_dir"])
        except KeyError as exc:
            raise ValueError(f"{path}: {key!r} has no {exc.args[0]!r}") from exc
        if not isinstance(files, dict) or not all(isinstance(v, str) for v in files.values()):
            raise ValueError(f"{path}: {key!r} has no files_sha256 table")
        try:
            folder = inside(models_root, rel, f"{path}: {key!r}")
        except ValueError as exc:
            raise ValueError(f"{exc} (snapshot_dir)") from exc
        records[str(key)] = InstalledModel(
            repo=repo,
            revision=revision,
            snapshot_dir=folder,
            files_sha256={str(k): str(v) for k, v in files.items()},
        )
    return records


def write_manifest(models_root: Path, records: Mapping[str, InstalledModel]) -> Path:
    """Write the install record (to a temporary name, then renamed over the old one)."""
    path = models_root / MANIFEST_NAME
    data = {
        key: {
            "repo": model.repo,
            "revision": model.revision,
            "snapshot_dir": model.snapshot_dir.relative_to(models_root).as_posix(),
            "files_sha256": dict(sorted(model.files_sha256.items())),
        }
        for key, model in sorted(records.items())
    }
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def sha256_file(path: Path) -> str:
    """The sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def check_files(model: InstalledModel, *, hash_files: bool = True) -> list[str]:
    """What is wrong with an installed model's files: each recorded file must be there and, with
    ``hash_files``, have its recorded sha256. Empty when all is well."""
    problems: list[str] = []
    for rel, expected in sorted(model.files_sha256.items()):
        try:
            path = inside(model.snapshot_dir, rel, "the install record")
            if not path.is_file():
                problems.append(f"{rel} is missing")
            elif hash_files and sha256_file(path) != expected:
                problems.append(f"{rel} does not match its recorded sha256")
        except ValueError as exc:
            problems.append(str(exc))
        except OSError as exc:
            problems.append(f"{rel} cannot be read ({exc.strerror or exc})")
    return problems


__all__ = [
    "MANIFEST_NAME",
    "InstalledModel",
    "check_files",
    "inside",
    "pinned_revisions",
    "read_manifest",
    "safe_relative",
    "sha256_file",
    "snapshot_dir",
    "write_manifest",
]
