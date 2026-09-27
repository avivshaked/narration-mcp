"""A ``load`` request's snapshot references, checked as every worker checks them (design section 4; Appendix A).

A snapshot reference (``protocol.ModelRef``) is an object with ``repo``, ``revision`` and ``snapshot_dir``: a Qwen
load's ``model``, or a QA load's ``models.<use>``. Every model loads offline from a local snapshot directory named
by its 40-hex commit revision. The checks, one pass over the references per step, in the request's order, each
refusal an ``OpError`` with its code and ``details.field`` (the reference's field, such as ``models.asr``):

1. ``check_ref``: the reference is an object whose three members are strings (``INVALID_REQUEST``, the
   reference's field), and ``snapshot_dir`` is an absolute path (``<field>.snapshot_dir``);
2. ``check_snapshots``: every snapshot directory exists (``BACKEND_NOT_INSTALLED``, the reference's field, with
   ``repo`` and ``snapshot_dir``), before anything else is checked; then each revision is a 40-hex commit SHA
   (``<field>.revision``) that names its directory (``<field>.snapshot_dir``, with ``revision`` and ``folder``).

The fake (``narration_worker.fake``) and the QA worker (``narration_worker_qa``) use these functions; the
``qwen3`` worker makes the same checks of its ``model`` (``narration_qwen3tts.worker``). The contract tests
(``narration_worker.testing.contract``) pin the codes and fields for every role that takes ``models``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from .errors import OpError
from .handler import REVISION_PATTERN

REF_MEMBERS: Final = ("repo", "revision", "snapshot_dir")
"""The members of a snapshot reference, all strings."""
INSTALL_HINT: Final = "install the models (narration-admin install)"
"""What to do about a snapshot directory that is not there."""


def by_use(models: object, field: str = "models") -> list[tuple[str, object]]:
    """A ``models`` member's references as (field, reference), in the request's order: ``models.<use>``.

    ``INVALID_REQUEST`` (``field``) unless ``models`` is an object. The references themselves are not checked
    here (``check_ref``).
    """
    if not isinstance(models, dict):
        raise OpError("INVALID_REQUEST", f"{field} must be an object", {"field": field})
    return [(f"{field}.{use}", ref) for use, ref in cast(dict[str, object], models).items()]


def check_ref(field: str, ref: object) -> dict[str, str]:
    """Step 1 of the module docstring for one reference: the reference, typed, or ``INVALID_REQUEST``."""
    if not isinstance(ref, dict) or not all(isinstance(cast(dict[str, object], ref).get(k), str) for k in REF_MEMBERS):
        raise OpError("INVALID_REQUEST", f"{field} must have repo, revision and snapshot_dir", {"field": field})
    checked = cast(dict[str, str], ref)
    if not Path(checked["snapshot_dir"]).is_absolute():
        name = f"{field}.snapshot_dir"
        raise OpError("INVALID_REQUEST", f"{name} must be an absolute path", {"field": name})
    return checked


def check_snapshots(refs: Sequence[tuple[str, Mapping[str, str]]]) -> None:
    """Step 2 of the module docstring for references that passed ``check_ref``: every directory exists
    (``BACKEND_NOT_INSTALLED``), then each revision names its directory (``check_revision``)."""
    for field, ref in refs:
        if not Path(ref["snapshot_dir"]).is_dir():
            raise OpError(
                "BACKEND_NOT_INSTALLED",
                f"no snapshot of {ref['repo']} at {ref['snapshot_dir']}; {INSTALL_HINT}",
                {"field": field, "repo": ref["repo"], "snapshot_dir": ref["snapshot_dir"]},
            )
    for field, ref in refs:
        check_revision(field, ref)


def check_revision(field: str, ref: Mapping[str, str]) -> None:
    """``INVALID_REQUEST`` unless the reference's revision is a 40-hex commit SHA that names its directory."""
    revision = ref["revision"]
    if REVISION_PATTERN.fullmatch(revision) is None:
        name = f"{field}.revision"
        raise OpError("INVALID_REQUEST", f"{name} must be a 40-hex commit SHA", {"field": name})
    folder = Path(ref["snapshot_dir"]).name
    if folder != revision:
        raise OpError(
            "INVALID_REQUEST",
            "the snapshot folder must be named by its revision (section 4)",
            {"field": f"{field}.snapshot_dir", "revision": revision, "folder": folder},
        )
