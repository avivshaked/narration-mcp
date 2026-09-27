"""The calibration corpus ``measure_voice`` renders (design sections 3.2 and 15; plan.md WP18 and WP33).

The corpus is the service's own text, never a caller's: ``material/calibration/<set id>/``, a
``manifest.json`` that lists every file with its size and sha256, and ``paragraphs.json``, which holds the
calibration paragraphs and one ladder paragraph per rung. ``load_corpus`` reads a set and checks every file
against its manifest before any of it is used, so a measurement never runs on text the manifest does not
name.

**The calibration set** (section 3.2: "the design text plus three corpus paragraphs") is all corpus content:
``paragraphs.json`` carries the design text as its own top-level ``design_text`` paragraph (the lead's ruling:
the text the calibration renders must be covered by the corpus version, and so by the measurement key, never
read from ``[voice_design] design_text``, which is ``design_voice``'s and the canary's). ``load_corpus`` returns
it first among the ``MaterialSet``'s ``paragraphs``, then the calibration paragraphs. A ``draft`` set may lack
it (the drafts before gate H1 do), and then the calibration set is the corpus paragraphs alone; a ``frozen``
set without it is refused.

**The corpus version** (``corpus_version``) is ``"<set id>@sha256:<hex>"``, where the hex is the sha256 of
the manifest's bytes. The manifest lists every file's sha256, so the version names the set's exact content,
and the measurement key covers it (section 10.2). A draft set that gate H1 freezes changes its manifest
(``status``), and so its version: a voice measured on the draft keeps its measurement for generation, and
``measure_voice`` measures it again under the frozen set.

**Where the material is.** v1 runs from a git clone (plan.md section 1.4), so the material sits in the
checkout, beside ``src``: ``default_material_root`` finds it from this package's location. A wheel would not
carry it; an installation without it cannot measure, and ``load_corpus`` says so (``MaterialError``).

This is the corpus half of the contracts' ``MaterialLoader`` (``corpus``), which no other package implements
yet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Final, get_args

import narration
from narration.contracts.errors import MaterialError
from narration.contracts.models import CueIn, ExactSpan, Hint, MaterialParagraph, MaterialSet
from narration.contracts.names import HASH_PREFIX, MaterialStatus

CALIBRATION_KIND: Final = "calibration"
MANIFEST: Final = "manifest.json"
PARAGRAPHS: Final = "paragraphs.json"
DESIGN_TEXT: Final = "design_text"
"""The top-level field of ``paragraphs.json`` that holds the calibration's design text, as a paragraph."""


def default_material_root() -> Path:
    """``<checkout>/material``: the folder beside ``src`` in the git clone the service runs from."""
    return Path(narration.__file__).resolve().parents[2] / "material"


def corpus_version(corpus: MaterialSet) -> str:
    """``"<set id>@sha256:<hex>"``: the set and the sha256 of its manifest (``keys.CORPUS_VERSION_PATTERN``)."""
    return f"{corpus.set_id}@{HASH_PREFIX}{corpus.sha256}"


def load_corpus(set_id: str, root: Path | None = None) -> MaterialSet:
    """Read the calibration corpus ``set_id`` from ``root`` (default: ``default_material_root``), each file
    checked against the manifest's size and sha256. ``paragraphs`` is the calibration set, the design text first
    when the corpus has one (see the module docstring); ``ladder`` the ladder paragraphs; ``hints`` a term-only
    hint for each invented name.

    Raises ``MaterialError`` when the set is missing or malformed, a file does not match its manifest, or the
    manifest names a file outside the set's folder.
    """
    folder = (root if root is not None else default_material_root()) / CALIBRATION_KIND / set_id
    manifest_path = folder / MANIFEST
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise MaterialError(f"the calibration corpus {set_id} is missing: no {MANIFEST} in {folder}") from exc
    manifest = _object(_json(manifest_bytes, manifest_path), manifest_path)
    if manifest.get("set") != set_id:
        raise MaterialError(f"{manifest_path}: its set is {manifest.get('set')!r}, not {set_id!r}")
    if manifest.get("kind") != CALIBRATION_KIND:
        raise MaterialError(f"{manifest_path}: its kind is {manifest.get('kind')!r}, not {CALIBRATION_KIND!r}")
    status: MaterialStatus | None = next((s for s in get_args(MaterialStatus) if s == manifest.get("status")), None)
    if status is None:
        raise MaterialError(f"{manifest_path}: status {manifest.get('status')!r} is not draft or frozen")
    version = manifest.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise MaterialError(f"{manifest_path}: version must be an integer")
    files = _verified_files(folder, manifest, manifest_path)
    if PARAGRAPHS not in files:
        raise MaterialError(f"{manifest_path}: the manifest does not list {PARAGRAPHS}")
    data = _object(_json(files[PARAGRAPHS], folder / PARAGRAPHS), folder / PARAGRAPHS)
    names = data.get("invented_names", [])
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise MaterialError(f"{folder / PARAGRAPHS}: invented_names must be a list of names")
    design = data.get(DESIGN_TEXT)
    if design is None and status != "draft":
        raise MaterialError(f"{folder / PARAGRAPHS}: a {status} calibration corpus needs its {DESIGN_TEXT}")
    calibration = tuple(_paragraph(p, f"calibration[{i}]") for i, p in enumerate(_list(data, "calibration")))
    if design is not None:
        calibration = (_paragraph(design, DESIGN_TEXT), *calibration)
    ids = [p.segment_id for p in calibration]
    if len(set(ids)) != len(ids):
        raise MaterialError(f"{folder / PARAGRAPHS}: two calibration paragraphs share a segment_id")
    return MaterialSet(
        set_id=set_id,
        kind=CALIBRATION_KIND,
        version=version,
        status=status,
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        paragraphs=calibration,
        ladder=tuple(_paragraph(p, f"ladder[{i}]") for i, p in enumerate(_list(data, "ladder"))),
        hints=tuple(Hint(term=n) for n in names),
        invented_names=tuple(names),
    )


def _verified_files(folder: Path, manifest: dict[str, Any], where: Path) -> dict[str, bytes]:
    """Every file the manifest lists, read and checked against its size and sha256, by its path."""
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise MaterialError(f"{where}: the manifest lists no files")
    out: dict[str, bytes] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise MaterialError(f"{where}: a file entry is not an object")
        rel, size, sha = entry.get("path"), entry.get("bytes"), entry.get("sha256")
        if not isinstance(rel, str) or not isinstance(size, int) or not isinstance(sha, str):
            raise MaterialError(f"{where}: a file entry needs path, bytes and sha256")
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts or "\\" in rel or ":" in rel:
            raise MaterialError(f"{where}: {rel!r} is not a path inside the set")
        try:
            content = (folder / pure).read_bytes()
        except OSError as exc:
            raise MaterialError(f"{folder}: {rel} is listed in the manifest but cannot be read") from exc
        if len(content) != size or hashlib.sha256(content).hexdigest() != sha:
            raise MaterialError(f"{folder}: {rel} does not match its manifest (size or sha256)")
        out[rel] = content
    return out


def _paragraph(raw: Any, where: str) -> MaterialParagraph:
    if not isinstance(raw, dict):
        raise MaterialError(f"{PARAGRAPHS}: {where} is not an object")
    segment_id, spoken_chars, target = raw.get("segment_id"), raw.get("spoken_chars"), raw.get("target_spoken_chars")
    if not isinstance(segment_id, str) or not segment_id:
        raise MaterialError(f"{PARAGRAPHS}: {where} has no segment_id")
    if not isinstance(spoken_chars, int) or (target is not None and not isinstance(target, int)):
        raise MaterialError(f"{PARAGRAPHS}: {where} needs integer spoken_chars (and target_spoken_chars)")
    cues_raw = raw.get("cues")
    if not isinstance(cues_raw, list) or not cues_raw:
        raise MaterialError(f"{PARAGRAPHS}: {where} has no cues")
    cues: list[CueIn] = []
    for k, cue in enumerate(cues_raw):
        text = cue.get("text") if isinstance(cue, dict) else None
        if not isinstance(text, str) or not text:
            raise MaterialError(f"{PARAGRAPHS}: {where}.cues[{k}] has no text")
        spans = cue.get("exact") or []
        try:
            exact = tuple(ExactSpan(start=int(s["start"]), end=int(s["end"])) for s in spans)
        except (TypeError, KeyError, ValueError) as exc:
            raise MaterialError(f"{PARAGRAPHS}: {where}.cues[{k}].exact is malformed") from exc
        cues.append(CueIn(text=text, exact=exact))
    return MaterialParagraph(
        segment_id=segment_id, cues=tuple(cues), spoken_chars=spoken_chars, target_spoken_chars=target
    )


def _json(content: bytes, where: Path) -> Any:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise MaterialError(f"{where}: not valid UTF-8 JSON") from exc


def _object(value: Any, where: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MaterialError(f"{where}: not a JSON object")
    return value


def _list(data: dict[str, Any], name: str) -> list[Any]:
    value = data.get(name)
    if not isinstance(value, list) or not value:
        raise MaterialError(f"{PARAGRAPHS}: {name} must be a non-empty list")
    return value


__all__ = [
    "CALIBRATION_KIND",
    "DESIGN_TEXT",
    "MANIFEST",
    "PARAGRAPHS",
    "corpus_version",
    "default_material_root",
    "load_corpus",
]
