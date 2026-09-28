"""The three cache layers: renders, takes, analyses (design sections 10.2, 15; App. B)."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from narration.contracts import names
from narration.contracts.interfaces import Store
from narration.contracts.models import MeasurementRecord
from narration.store import NarrationStore, StoreError, StoreIntegrityError, StorePathError
from narration.store import files as store_files

from .conftest import FakeClock
from .factories import (
    analysis_record,
    audio_bytes,
    measurement_record,
    profile_record,
    render_record,
    scratch_file,
    sha,
    take_record,
)
from .txn_fakes import scripted

DAY = 86_400.0


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
    second = store.put_analysis(analysis_record(take, qa_profile="test.another-profile"))
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


# ---------------------------------------------------------------- failed publishes (review fixes)
def _refuse_folder_renames(monkeypatch: pytest.MonkeyPatch, *, times: int | None = None) -> dict[str, int]:
    """Make renaming a ``.staging-`` folder fail as Windows does while a file in it is open: ``times``
    times, or always. The files inside can still be moved."""
    real = os.rename
    state = {"refused": 0}

    def rename(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(src).name.startswith(".staging-") and (times is None or state["refused"] < times):
            state["refused"] += 1
            raise PermissionError(13, "the folder is in use", os.fspath(src))
        real(src, dst)

    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(store_files, "_retry_sleep", lambda attempt: None)
    return state


def _temp_names(store: NarrationStore) -> list[str]:
    return [p.name for p in store.root.rglob("*") if p.name.startswith((".tmp-", ".staging-", ".trash-"))]


def test_a_folder_rename_refused_for_a_moment_is_retried(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _refuse_folder_renames(monkeypatch, times=2)
    published, data = publish_render(store)
    assert state["refused"] == 2
    assert Path(published.raw.path).read_bytes() == data


def test_a_failed_publish_gives_the_audio_back(store: NarrationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    # A render is GPU work: if its folder cannot be put in place, the raw audio goes back where it was.
    _refuse_folder_renames(monkeypatch)
    record = render_record()
    src = scratch_file(store, "raw.wav", audio_bytes("gpu output"))
    with pytest.raises(PermissionError):
        store.put_render(record, src)
    assert src.read_bytes() == audio_bytes("gpu output")
    assert not store_files.is_readonly(src)  # as the caller had it
    assert store.get_render(record.render_key) is None
    assert _temp_names(store) == []


def test_a_failed_replace_keeps_the_published_profile(store: NarrationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = sha(b"clip")
    old = store.put_profile(profile_record(audio, pictures=False), None)
    _refuse_folder_renames(monkeypatch)
    with pytest.raises(PermissionError):
        store.put_profile(profile_record(audio, version="profile-2", pictures=False), None)
    assert store.get_profile(audio, names.PROFILE_VERSION) == old
    assert os.listdir(store.profile_dir(audio)) == ["profile.json"]
    assert _temp_names(store) == []


def test_a_failed_index_write_puts_everything_back(store: NarrationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = sha(b"clip")
    old = store.put_profile(profile_record(audio, pictures=False), None)
    pictures = store.scratch_path("p2", "spectrogram.png").parent
    (pictures / "spectrogram.png").write_bytes(b"spectrogram")
    (pictures / "pitch.png").write_bytes(b"pitch")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the database is full")

    monkeypatch.setattr(store, "_index_files", fail)
    with pytest.raises(RuntimeError):
        store.put_profile(profile_record(audio, version="profile-2"), pictures)
    assert store.get_profile(audio, names.PROFILE_VERSION) == old
    assert os.listdir(store.profile_dir(audio)) == ["profile.json"]
    assert (pictures / "spectrogram.png").read_bytes() == b"spectrogram"
    assert (pictures / "pitch.png").read_bytes() == b"pitch"
    assert _temp_names(store) == []


def test_a_published_file_is_never_consumed_or_moved_s17_2(store: NarrationStore) -> None:
    first, data = publish_render(store)
    published_raw = Path(first.raw.path)
    # The key exists, so the store would discard a duplicate source; a published file is not one.
    assert store.put_render(render_record(), published_raw) == first
    assert published_raw.read_bytes() == data
    with pytest.raises(StorePathError, match="scratch"):
        store.put_render(render_record("Another line."), published_raw)
    assert published_raw.read_bytes() == data
    assert store.get_render(first.render_key) == first


def test_audio_is_moved_in_only_from_scratch_s17_2(store: NarrationStore) -> None:
    stray = store.root / "logs" / "raw.wav"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"audio")
    with pytest.raises(StorePathError, match="scratch"):
        store.put_render(render_record(), stray)
    assert stray.read_bytes() == b"audio"


def test_an_analysis_is_refused_if_its_take_goes_meanwhile(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The take is checked again under the write lock, so no analysis row can outlive its take.
    render, _ = publish_render(store)
    take = store.put_take(take_record(render.raw.sha256, render.render_id), scratch_file(store, "d.wav", b"delivery"))
    real = store_files.write_temp

    def write_then_lose_the_take(path: Path, data: bytes, **kwargs: bool) -> tuple[Path, str, int]:
        out = real(path, data, **kwargs)
        conn = sqlite3.connect(str(store.root / "narration.sqlite"))
        try:
            with conn:
                conn.execute("DELETE FROM takes WHERE take_id = ?", (take.take_id,))
        finally:
            conn.close()
        return out

    monkeypatch.setattr(store_files, "write_temp", write_then_lose_the_take)
    record = analysis_record(take)
    with pytest.raises(StoreError, match="no longer in the store"):
        store.put_analysis(record)
    assert store.get_analysis(record.analysis_key) is None
    assert _temp_names(store) == []


# ---------------------------------------------------------------- follow-ups: after the commit, and a failed COMMIT
def test_a_publish_that_cannot_remove_the_replaced_copy_still_succeeds(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The row is committed, so the publish has succeeded: removing the replaced folder afterwards is best
    # effort, logged if it fails, and gc removes the .trash- name later.
    audio = sha(b"clip")
    store.put_profile(profile_record(audio, pictures=False), None)
    real = store_files.remove_tree

    def refuse_trash(path: Path) -> None:
        if path.name.startswith(".trash-"):
            raise PermissionError(13, "a file in it is open", str(path))
        real(path)

    monkeypatch.setattr(store_files, "remove_tree", refuse_trash)
    with caplog.at_level(logging.WARNING, logger="narration.store.store"):
        newer = store.put_profile(profile_record(audio, version="profile-2", pictures=False), None)
    monkeypatch.undo()
    assert store.get_profile(audio, "profile-2") == newer
    assert "left for gc" in caplog.text and "neither" not in caplog.text
    trash = [p for p in store.profile_dir(audio).parent.iterdir() if p.name.startswith(".trash-")]
    assert len(trash) == 1 and store_files.trash_time(trash[0].name) == int(clock.now)
    store.gc(dry_run=False)
    assert trash[0].is_dir()  # renamed just now: not yet an old leftover
    clock.advance(2 * DAY)
    report = store.gc(dry_run=False)
    assert store.layout.rel(trash[0]) in report["leftovers"] and not trash[0].exists()
    assert store.get_profile(audio, "profile-2") == newer


def test_a_file_publish_that_cannot_remove_the_replaced_file_still_succeeds(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.put_measurement(measurement_record())
    real = store_files.remove_tree

    def refuse_trash(path: Path) -> None:
        if path.name.startswith(".trash-"):
            raise PermissionError(13, "held open by a reader", str(path))
        real(path)

    monkeypatch.setattr(store_files, "remove_tree", refuse_trash)
    newer = store.put_measurement(measurement_record(corpus_hex="c1" * 32))
    assert store.get_measurement(newer.voice_hash, newer.engine_profile.id) == newer


def test_a_failed_commit_gives_the_audio_back(store: NarrationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    # The COMMIT itself can fail after the folder was renamed into place: it goes back, and so does the audio.
    record = render_record()
    src = scratch_file(store, "raw.wav", audio_bytes("gpu output"))
    scripted(monkeypatch, store, fail_commit=True)
    with pytest.raises(sqlite3.OperationalError):
        store.put_render(record, src)
    monkeypatch.undo()
    assert src.read_bytes() == audio_bytes("gpu output") and not store_files.is_readonly(src)
    assert store.get_render(record.render_key) is None
    assert not store.render_dir(record.render_id).exists()
    assert _temp_names(store) == []


def test_a_failed_commit_keeps_the_published_profile(store: NarrationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = sha(b"clip")
    old = store.put_profile(profile_record(audio, pictures=False), None)
    scripted(monkeypatch, store, fail_commit=True)
    with pytest.raises(sqlite3.OperationalError):
        store.put_profile(profile_record(audio, version="profile-2", pictures=False), None)
    monkeypatch.undo()
    assert store.get_profile(audio, names.PROFILE_VERSION) == old
    assert os.listdir(store.profile_dir(audio)) == ["profile.json"]
    assert _temp_names(store) == []


def test_a_failed_file_commit_keeps_the_published_file(store: NarrationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    old = store.put_measurement(measurement_record())
    scripted(monkeypatch, store, fail_commit=True)
    with pytest.raises(sqlite3.OperationalError):
        store.put_measurement(measurement_record(corpus_hex="c1" * 32))
    monkeypatch.undo()
    assert store.get_measurement(old.voice_hash, old.engine_profile.id) == old
    assert store.verify()["mismatched"] == []
    assert _temp_names(store) == []


def test_a_failed_publish_is_undone_before_the_lock_is_released(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Undone inside the transaction: once the ROLLBACK makes the old row visible again, its folder is
    # already back under its own name, so no reader can take the folder for missing and drop the row.
    audio = sha(b"clip")
    store.put_profile(profile_record(audio, pictures=False), None)
    folder = store.profile_dir(audio)
    seen = scripted(monkeypatch, store, look=lambda: os.listdir(folder)).seen

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the database is full")

    monkeypatch.setattr(store, "_index_files", fail)
    with pytest.raises(RuntimeError):
        store.put_profile(profile_record(audio, version="profile-2", pictures=False), None)
    assert seen == [["profile.json"]]


# ---------------------------------------------------------------- follow-ups: the WP12 follow-ups review


def _measurement_json(store: NarrationStore, record: MeasurementRecord) -> Path:
    return store.measurement_dir(record.voice_hash, record.engine_profile.id) / "measurement.json"


def _fail(*args: object, **kwargs: object) -> None:
    raise RuntimeError("the database is full")


def test_a_failed_file_publish_is_undone_before_the_lock_is_released(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A file publish renames the old file aside and the new one in; when its body fails, both renames are
    # undone before the ROLLBACK makes the old row visible again.
    old = store.put_measurement(measurement_record())
    final = _measurement_json(store, old)
    old_bytes = final.read_bytes()
    seen = scripted(monkeypatch, store, look=final.read_bytes).seen
    monkeypatch.setattr(store, "_index_files", _fail)
    with pytest.raises(RuntimeError):
        store.put_measurement(measurement_record(corpus_hex="c1" * 32))
    assert seen == [old_bytes]


def test_a_failed_commit_is_undone_before_the_lock_is_released(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = store.put_measurement(measurement_record())
    final = _measurement_json(store, old)
    old_bytes = final.read_bytes()
    seen = scripted(monkeypatch, store, fail_commit=True, look=final.read_bytes).seen
    with pytest.raises(sqlite3.OperationalError):
        store.put_measurement(measurement_record(corpus_hex="c1" * 32))
    assert seen == [old_bytes]
    monkeypatch.undo()
    assert store.get_measurement(old.voice_hash, old.engine_profile.id) == old


def test_an_undo_after_sqlite_ended_the_transaction_leaves_another_writers_file(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SQLite can end a transaction itself when its COMMIT fails, releasing the lock. The undo then takes the
    # lock again, and puts back only what is still its own: here another writer published in that moment.
    old = store.put_measurement(measurement_record())
    final = _measurement_json(store, old)

    def another_writer() -> None:
        final.rename(final.with_name(".trash-0123456789abcdef-theirs"))
        other = final.with_name(".tmp-0123456789abcdef-other")
        other.write_bytes(b"another writer's measurement")
        other.rename(final)

    connection = scripted(monkeypatch, store, fail_commit=True, sqlite_ends_it=True, meanwhile=another_writer)
    with pytest.raises(sqlite3.OperationalError):
        store.put_measurement(measurement_record(corpus_hex="c1" * 32))
    assert connection.statements[-3:] == ["COMMIT failed", "BEGIN IMMEDIATE", "ROLLBACK"]
    assert final.read_bytes() == b"another writer's measurement"


def _pause_after_renaming_aside(
    monkeypatch: pytest.MonkeyPatch, final: Path
) -> tuple[threading.Event, threading.Event]:
    """Make a publish stop just after it renamed ``final`` aside (under its write lock) until ``go_on`` is
    set; ``aside`` is set when it gets there."""
    aside, go_on = threading.Event(), threading.Event()
    real_rename = store_files.rename_retrying

    def rename(src: Path, dst: Path, *, attempts: int = store_files.REPLACE_ATTEMPTS) -> None:
        real_rename(src, dst, attempts=attempts)
        if Path(src) == final and Path(dst).name.startswith(".trash-"):
            aside.set()
            assert go_on.wait(30)

    monkeypatch.setattr(store_files, "rename_retrying", rename)
    return aside, go_on


def _read_during_the_publish(
    store: NarrationStore,
    monkeypatch: pytest.MonkeyPatch,
    final: Path,
    publish: Callable[[], object],
    read: Callable[[], object],
) -> tuple[list[object], list[object]]:
    """Run ``publish`` in a thread, stop it once it has renamed ``final`` aside, and ``read`` in another
    thread meanwhile. Returns what each returned or raised."""
    aside, go_on = _pause_after_renaming_aside(monkeypatch, final)
    reader_waits = threading.Event()
    real_drop = store._drop

    def drop(*args: Any) -> bool:
        reader_waits.set()  # the reader found the file missing, and now waits for the publisher's lock
        return real_drop(*args)

    monkeypatch.setattr(store, "_drop", drop)
    published: list[object] = []
    read_back: list[object] = []

    def run(target: Callable[[], object], into: list[object]) -> None:
        try:
            into.append(target())
        except Exception as exc:
            into.append(exc)

    publisher = threading.Thread(target=run, args=(publish, published))
    reader = threading.Thread(target=run, args=(read, read_back))
    publisher.start()
    try:
        assert aside.wait(30)
        reader.start()
        assert reader_waits.wait(30)
    finally:
        go_on.set()
        publisher.join(30)
        if reader.ident is not None:  # started
            reader.join(30)
    return published, read_back


def test_a_reader_during_a_replace_gets_the_new_measurement_s15(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The publish renames the old file aside before it renames the new one in. A reader in that moment
    # finds the file missing; it must not report the measurement as gone.
    old = store.put_measurement(measurement_record())
    newer = measurement_record(corpus_hex="c1" * 32)
    published, read_back = _read_during_the_publish(
        store,
        monkeypatch,
        _measurement_json(store, old),
        lambda: store.put_measurement(newer),
        lambda: store.get_measurement(old.voice_hash, old.engine_profile.id),
    )
    assert published == [newer] and read_back == [newer]


def test_a_reader_during_a_failed_replace_gets_the_old_profile_s15(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same moment in a folder publish that then fails: the reader gets the profile that stays.
    audio = sha(b"clip")
    old = store.put_profile(profile_record(audio, pictures=False), None)
    monkeypatch.setattr(store, "_index_files", _fail)
    published, read_back = _read_during_the_publish(
        store,
        monkeypatch,
        store.profile_dir(audio),
        lambda: store.put_profile(profile_record(audio, version="profile-2", pictures=False), None),
        lambda: store.get_profile(audio, names.PROFILE_VERSION),
    )
    assert [type(p) for p in published] == [RuntimeError] and read_back == [old]


def test_a_measurement_published_while_gc_collects_the_one_it_replaces_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # gc renames the old measurement's folder away while a new measurement of the same voice and engine
    # waits to be published: the new one's temporary file must not be in that folder.
    old = store.put_measurement(measurement_record())
    with store._write() as conn:  # last used past measurement_retention_days (365); files stay recent
        conn.execute("UPDATE measurements SET last_used_at = ?", (clock.now - 366 * DAY,))
    real_write_temp = store_files.write_temp
    collected: list[list[str]] = []

    def write_then_gc(path: Path, data: Any, **kwargs: Any) -> tuple[Path, str, int]:
        written = real_write_temp(path, data, **kwargs)
        collected.append(store.gc(dry_run=False)["items"]["measurements"])
        return written

    monkeypatch.setattr(store_files, "write_temp", write_then_gc)
    newer = store.put_measurement(measurement_record(corpus_hex="c1" * 32))
    monkeypatch.undo()
    assert collected == [[old.measurement_key]]
    assert store.get_measurement(newer.voice_hash, newer.engine_profile.id) == newer
    assert store.verify()["mismatched"] == []
