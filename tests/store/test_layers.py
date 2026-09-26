"""The three cache layers: renders, takes, analyses (design sections 10.2, 15; App. B)."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from narration.contracts.interfaces import Store
from narration.store import NarrationStore, StoreError, StoreIntegrityError, StorePathError
from narration.store import files as store_files

from .factories import analysis_record, audio_bytes, render_record, scratch_file, sha, take_record


def publish_render(store: NarrationStore, text: str = "Before dawn, the reef belongs to the shrimp."):
    data = audio_bytes(text)
    return store.put_render(render_record(text), scratch_file(store, "raw.wav", data)), data


def test_the_store_implements_the_store_contract(store: NarrationStore) -> None:
    contract: Store = store
    assert isinstance(contract, Store)


def test_a_render_is_published_whole_and_read_only_s15(store: NarrationStore) -> None:
    record = render_record()
    src = scratch_file(store, "raw.wav", audio_bytes("r"))
    published = store.put_render(record, src)
    folder = store.render_dir(record.render_id)
    assert not src.exists()  # moved into place
    assert published.raw.path == str(folder / "raw.wav")
    assert published.raw.sha256 == sha(audio_bytes("r"))
    assert (folder / "raw.wav").read_bytes() == audio_bytes("r")
    assert store_files.is_readonly(folder / "raw.wav") and store_files.is_readonly(folder / "render.json")
    assert sorted(p.name for p in folder.iterdir()) == ["raw.wav", "render.json"]
    assert store.get_render(record.render_key) == published
    assert store.get_render_by_id(record.render_id) == published


def test_sidecars_keep_paths_relative_to_the_store_root_s15(store: NarrationStore) -> None:
    published, _ = publish_render(store)
    sidecar = json.loads((store.render_dir(published.render_id) / "render.json").read_text(encoding="utf-8"))
    assert sidecar["schema"] == "narration.render/v1"
    assert sidecar["raw"]["path"] == f"renders/{published.render_id[3:5]}/{published.render_id}/raw.wav"
    assert str(store.root) not in json.dumps(sidecar)


def test_the_first_published_render_of_a_key_is_kept_s10_2(store: NarrationStore) -> None:
    first, data = publish_render(store)
    again = scratch_file(store, "again.wav", b"different bytes for the same key")
    assert store.put_render(render_record(), again) == first
    assert not again.exists()  # consumed
    assert Path(first.raw.path).read_bytes() == data


def test_a_stated_hash_that_does_not_match_the_file_is_refused(store: NarrationStore) -> None:
    record = render_record()
    record = dataclasses.replace(record, raw=dataclasses.replace(record.raw, sha256="0" * 64))
    with pytest.raises(StoreIntegrityError):
        store.put_render(record, scratch_file(store, "raw.wav", b"audio"))
    assert store.get_render(record.render_key) is None


def test_an_id_that_does_not_name_its_key_is_refused_s6(store: NarrationStore) -> None:
    record = dataclasses.replace(render_record(), render_id="rn_0000000000000000")
    with pytest.raises(StoreIntegrityError):
        store.put_render(record, scratch_file(store, "raw.wav", b"audio"))


def test_audio_from_outside_the_store_is_refused_s17_2(store: NarrationStore, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.wav"
    outside.write_bytes(b"audio")
    with pytest.raises(StorePathError):
        store.put_render(render_record(), outside)
    assert outside.read_bytes() == b"audio"  # never moved or removed


def test_takes_and_analyses_round_trip_app_b(store: NarrationStore) -> None:
    render, _ = publish_render(store)
    take = store.put_take(take_record(render.raw.sha256, render.render_id), scratch_file(store, "d.wav", b"delivery"))
    assert take.delivery.path == str(store.take_dir(take.take_id) / "delivery.wav")
    assert take.delivery.sha256 == sha(b"delivery")
    assert store.get_take(take.delivery_key) == take == store.get_take_by_id(take.take_id)
    analysis = store.put_analysis(analysis_record(take))
    path = store.analysis_path(take.take_id, analysis.analysis_id)
    assert path.is_file() and store_files.is_readonly(path)
    assert store.get_analysis(analysis.analysis_key) == analysis == store.get_analysis_by_id(analysis.analysis_id)
    second = store.put_analysis(analysis_record(take, qa_profile="default.v4"))
    # Oldest first; the fake clock gives both one time, so the id decides.
    assert store.analyses_of(take.take_id) == tuple(sorted((analysis, second), key=lambda a: a.analysis_id))
    assert store.analyses_of("tk_0000000000000000") == ()


def test_an_analysis_needs_its_take_first(store: NarrationStore) -> None:
    render, _ = publish_render(store)
    take = take_record(render.raw.sha256, render.render_id)
    take = dataclasses.replace(take, delivery=dataclasses.replace(take.delivery, sha256="d" * 64))
    with pytest.raises(StoreError, match="publish the take"):
        store.put_analysis(analysis_record(take))


def test_a_render_whose_file_was_removed_is_a_miss(store: NarrationStore) -> None:
    published, _ = publish_render(store)
    store_files.remove_tree(store.render_dir(published.render_id))
    assert store.get_render(published.render_key) is None
    assert store.claim(published.render_key, "daemon", ttl_s=30)[0] == "claimed"


def test_unknown_keys_and_ids_are_misses(store: NarrationStore) -> None:
    assert store.get_render("sha256:" + "0" * 64) is None
    assert store.get_take_by_id("tk_0000000000000000") is None
    assert store.get_analysis("not a key") is None


def test_nothing_but_published_names_stays_behind_after_a_publish(store: NarrationStore) -> None:
    publish_render(store)
    leftovers = [p for p in store.root.rglob("*") if p.name.startswith((".tmp-", ".staging-", ".trash-"))]
    assert leftovers == []


def test_a_crash_leftover_at_the_final_path_is_replaced(store: NarrationStore) -> None:
    record = render_record()
    folder = store.render_dir(record.render_id)
    folder.mkdir(parents=True)
    (folder / "raw.wav").write_bytes(b"half a file from a crashed publish")
    published = store.put_render(record, scratch_file(store, "raw.wav", b"the real render"))
    assert Path(published.raw.path).read_bytes() == b"the real render"
    assert not any(p.name.startswith(".trash-") for p in folder.parent.iterdir())
    assert os.listdir(folder) == ["raw.wav", "render.json"]
