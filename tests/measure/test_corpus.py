"""The calibration corpus loader (design sections 3.2 and 15; ``narration.measure.corpus``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from narration import keys
from narration.config import VoiceDesignConfig
from narration.contracts.errors import MaterialError
from narration.contracts.models import Hint
from narration.contracts.names import CORPUS
from narration.measure import corpus_version, load_corpus
from narration.measure.corpus import CALIBRATION_KIND, MANIFEST, PARAGRAPHS, default_material_root

from .support import DESIGN_SEGMENT, corpus_data, write_corpus


def test_the_services_own_corpus_loads_and_matches_its_manifest_s15() -> None:
    corpus = load_corpus(CORPUS)
    manifest = (default_material_root() / CALIBRATION_KIND / CORPUS / MANIFEST).read_bytes()
    assert corpus.set_id == CORPUS and corpus.kind == CALIBRATION_KIND
    assert corpus.sha256 == hashlib.sha256(manifest).hexdigest()
    assert corpus.paragraphs and corpus.ladder
    assert all(p.target_spoken_chars is not None for p in corpus.ladder)
    assert corpus.hints == tuple(Hint(term=n) for n in corpus.invented_names)


def test_the_services_corpus_is_frozen_and_renders_its_design_text_first_s3_2() -> None:
    """Gate H1 froze the corpus with its design text (DC-13, the service's default), which the calibration set
    renders before the three corpus paragraphs."""
    corpus = load_corpus(CORPUS)
    assert corpus.status == "frozen"
    first, *rest = corpus.paragraphs
    assert first.segment_id == DESIGN_SEGMENT
    assert [c.text for c in first.cues] == [VoiceDesignConfig().design_text]
    assert len(rest) == 3


def test_the_corpus_version_names_the_manifests_bytes_s10_2() -> None:
    corpus = load_corpus(CORPUS)
    version = corpus_version(corpus)
    assert version == f"{CORPUS}@sha256:{corpus.sha256}"
    assert keys.CORPUS_VERSION_PATTERN.fullmatch(version)


def test_the_design_text_comes_first_in_the_calibration_set_s3_2(tmp_path: Path) -> None:
    corpus = load_corpus(CORPUS, write_corpus(tmp_path, corpus_data()))
    ids = [p.segment_id for p in corpus.paragraphs]
    assert ids[0] == DESIGN_SEGMENT
    assert ids[1:] == [p["segment_id"] for p in corpus_data()["calibration"]]


def test_a_draft_may_lack_the_design_text_a_frozen_set_may_not_s3_2(tmp_path: Path) -> None:
    bare = corpus_data(design=False)
    assert DESIGN_SEGMENT not in [
        p.segment_id for p in load_corpus(CORPUS, write_corpus(tmp_path / "d", bare)).paragraphs
    ]
    with pytest.raises(MaterialError, match="design_text"):
        load_corpus(CORPUS, write_corpus(tmp_path / "f", bare, status="frozen"))
    frozen = load_corpus(CORPUS, write_corpus(tmp_path / "ok", corpus_data(), status="frozen"))
    assert frozen.status == "frozen"


def test_every_change_to_the_text_changes_the_version_s10_2(tmp_path: Path) -> None:
    data = corpus_data()
    one = load_corpus(CORPUS, write_corpus(tmp_path / "a", data))
    data["ladder"][0]["cues"][0]["text"] += " Then the rain stopped."
    two = load_corpus(CORPUS, write_corpus(tmp_path / "b", data))
    assert corpus_version(one) != corpus_version(two)


def test_a_file_that_does_not_match_its_manifest_is_refused_s15(tmp_path: Path) -> None:
    root = write_corpus(tmp_path, corpus_data())
    path = root / CALIBRATION_KIND / CORPUS / PARAGRAPHS
    path.write_bytes(path.read_bytes().replace(b"Tarnholm", b"Tarnholn"))
    with pytest.raises(MaterialError, match="does not match its manifest"):
        load_corpus(CORPUS, root)


def test_a_missing_set_is_refused_s15(tmp_path: Path) -> None:
    with pytest.raises(MaterialError, match="missing"):
        load_corpus(CORPUS, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("set", "another-set.v1", "its set is"),
        ("kind", "canary", "its kind is"),
        ("status", "retired", "not draft or frozen"),
        ("version", "1", "version must be an integer"),
        ("files", [{"path": "../paragraphs.json", "bytes": 1, "sha256": "0" * 64}], "not a path inside the set"),
        ("files", [], "lists no files"),
    ],
)
def test_a_malformed_manifest_is_refused_s15(tmp_path: Path, field: str, value: object, message: str) -> None:
    root = write_corpus(tmp_path, corpus_data())
    path = root / CALIBRATION_KIND / CORPUS / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MaterialError, match=message):
        load_corpus(CORPUS, root)


def test_two_calibration_paragraphs_with_one_id_are_refused_s15(tmp_path: Path) -> None:
    data = corpus_data()
    data["calibration"][0]["segment_id"] = DESIGN_SEGMENT
    with pytest.raises(MaterialError, match="share a segment_id"):
        load_corpus(CORPUS, write_corpus(tmp_path, data))


def test_a_paragraph_with_no_cues_is_refused_s15(tmp_path: Path) -> None:
    data = corpus_data()
    data["ladder"][1]["cues"] = []
    with pytest.raises(MaterialError, match=r"ladder\[1\] has no cues"):
        load_corpus(CORPUS, write_corpus(tmp_path, data))
