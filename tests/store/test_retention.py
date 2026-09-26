"""Retention, gc and verify (design section 15)."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from narration.contracts.models import ProvenanceEntry, RenderRecord, TakeRecord
from narration.keys import Keys
from narration.store import NarrationStore
from narration.store import files as store_files
from narration.store import store as store_module

from .conftest import FakeClock
from .factories import (
    alignment_benchmark,
    analysis_record,
    audio_bytes,
    candidate,
    engine_profile,
    job_record,
    measurement_record,
    profile_record,
    render_record,
    scratch_file,
    sha,
    take_record,
)
from .txn_fakes import scripted

DAY = 86_400.0
KEYS = Keys()


def populate(store: NarrationStore) -> dict[str, Any]:
    """One of everything the store keeps."""
    render = store.put_render(render_record(), scratch_file(store, "raw.wav", audio_bytes("raw")))
    take = store.put_take(take_record(render.raw.sha256, render.render_id), scratch_file(store, "d.wav", b"delivery"))
    analysis = store.put_analysis(analysis_record(take))
    measurement = store.put_measurement(measurement_record())
    profile = store.put_profile(profile_record(sha(b"clip"), pictures=False), None)
    design_id = KEYS.new_design_id()
    cand = store.put_candidate(candidate(design_id, 0), scratch_file(store, "c.wav", audio_bytes("cand")))
    store.add_provenance(ProvenanceEntry(clip_sha256=cand.clip.sha256, design_id=design_id, date="2026-09-26"))
    finished, _ = store.create_job(job_record(KEYS.new_job_id(), request_sha256="1" * 64))
    store.update_job(finished.job_id, status="completed")
    active, _ = store.create_job(job_record(KEYS.new_job_id(), request_sha256="2" * 64))
    engine = store.put_engine_profile(engine_profile())
    canary = store.put_canary_clip(engine.engine_profile_id, scratch_file(store, "canary.wav", b"canary"))
    benchmark = store.put_alignment_benchmark(alignment_benchmark())
    store.claim("sha256:" + "ee" * 32, "old-holder", ttl_s=60)
    return {
        "render": render,
        "take": take,
        "analysis": analysis,
        "measurement": measurement,
        "profile": profile,
        "design_id": design_id,
        "candidate": cand,
        "finished": finished,
        "active": active,
        "canary": canary,
        "benchmark": benchmark,
    }


def plant_leftovers(store: NarrationStore) -> dict[str, Path]:
    """What a crash leaves: a staged folder, a temporary file, an unindexed published-looking folder."""
    staging = store.root / "renders" / "aa" / ".staging-0123456789abcdef-rn_aaaaaaaaaaaaaaaa"
    staging.mkdir(parents=True)
    (staging / "raw.wav").write_bytes(b"staged")
    tmp = store.root / "engines" / ".tmp-0123456789abcdef-qwen3-base-1.7b.p1.json"
    tmp.write_bytes(b"{")
    orphan = store.root / "takes" / "bb" / "tk_bbbbbbbbbbbbbbbb"
    orphan.mkdir(parents=True)
    (orphan / "delivery.wav").write_bytes(b"orphan")
    old_scratch = store.scratch_path("job_x", "p01_a0.wav")
    old_scratch.write_bytes(b"scratch")
    return {"staging": staging, "tmp": tmp, "orphan": orphan, "scratch": old_scratch.parent}


def snapshot(store: NarrationStore) -> tuple[dict[str, tuple[bytes, bool]], list[str]]:
    tree = {
        p.relative_to(store.root).as_posix(): (p.read_bytes(), store_files.is_readonly(p))
        for p in sorted(store.root.rglob("*"))
        if p.is_file() and not p.name.startswith("narration.sqlite")
    }
    conn = sqlite3.connect(str(store.root / "narration.sqlite"))
    try:
        dump = list(conn.iterdump())
    finally:
        conn.close()
    return tree, dump


def test_gc_dry_run_changes_nothing_s15(store: NarrationStore, clock: FakeClock) -> None:
    populate(store)
    plant_leftovers(store)
    clock.advance(400 * DAY)  # everything is past every retention period
    before = snapshot(store)
    report = store.gc()  # a dry run by default
    assert report["dry_run"] is True
    assert snapshot(store) == before
    # ... yet it reports what a real run would remove.
    assert all(report["items"][kind] for kind in ("renders", "takes", "profiles", "designs", "measurements", "jobs"))
    assert report["expired_leases"] == 1 and report["bytes"] > 0
    assert report["leftovers"] == [
        "engines/.tmp-0123456789abcdef-qwen3-base-1.7b.p1.json",
        "renders/aa/.staging-0123456789abcdef-rn_aaaaaaaaaaaaaaaa",
    ]
    assert report["orphans"] == ["takes/bb/tk_bbbbbbbbbbbbbbbb"]
    assert report["scratch"] == ["scratch/job_x", "scratch/test"]


def test_gc_collects_what_retention_allows_and_nothing_of_the_services_own_s15(
    store: NarrationStore, clock: FakeClock
) -> None:
    items = populate(store)
    leftovers = plant_leftovers(store)
    clock.advance(31 * DAY)  # past retention_days (30), inside measurement_retention_days (365)
    report = store.gc(dry_run=False)
    assert report["items"]["renders"] == [items["render"].render_id]
    assert report["items"]["takes"] == [items["take"].take_id]
    assert report["items"]["jobs"] == [items["finished"].job_id]
    assert report["items"]["measurements"] == []
    assert store.get_render(items["render"].render_key) is None
    assert not store.render_dir(items["render"].render_id).exists()
    assert store.get_take_by_id(items["take"].take_id) is None and store.analyses_of(items["take"].take_id) == ()
    assert store.get_design(items["design_id"]) == () and not store.layout.design_root(items["design_id"]).exists()
    assert store.get_job(items["finished"].job_id) is None
    assert all(not p.exists() for p in leftovers.values())
    # Kept: the measurement (longer retention), the active job, and the service's own records.
    assert store.get_measurement(items["measurement"].voice_hash, items["measurement"].engine_profile.id) is not None
    assert store.get_job(items["active"].job_id) is not None
    assert store.is_provenance(items["candidate"].clip.sha256)
    assert (store.root / "provenance.jsonl").is_file()
    assert store.get_engine_profile("qwen3-base-1.7b.p1") is not None and Path(items["canary"].path).is_file()
    assert store.current_alignment_benchmark() == items["benchmark"]
    assert report["kept"] == {"provenance": 1, "engine_profiles": 1, "alignment_benchmarks": 1}
    verify = store.verify()
    assert verify["missing"] == [] and verify["mismatched"] == []
    # Past measurement_retention_days the measurement goes too, and its voice's folder with it.
    clock.advance(365 * DAY)
    store.gc(dry_run=False)
    assert store.get_measurement(items["measurement"].voice_hash, items["measurement"].engine_profile.id) is None
    assert not (store.root / "measurements" / items["measurement"].voice_hash.removeprefix("sha256:")).exists()
    assert store.get_job(items["active"].job_id) is not None  # an active job is never collected


def test_touch_restarts_the_retention_period_s15(store: NarrationStore, clock: FakeClock) -> None:
    items = populate(store)
    clock.advance(20 * DAY)
    store.touch("render", items["render"].render_id)
    store.touch("design", items["design_id"])
    store.touch("job", items["finished"].job_id)
    clock.advance(20 * DAY)
    report = store.gc(dry_run=False)
    assert items["render"].render_id not in report["items"]["renders"]
    assert report["items"]["designs"] == [] and report["items"]["jobs"] == []
    assert report["items"]["takes"] == [items["take"].take_id]
    assert store.get_render(items["render"].render_key) is not None


def test_a_used_analysis_keeps_its_take_and_an_unused_one_goes_s15(store: NarrationStore, clock: FakeClock) -> None:
    items = populate(store)
    old_analysis = items["analysis"]
    clock.advance(10 * DAY)
    newer = store.put_analysis(analysis_record(items["take"], qa_profile="default.v4"))
    clock.advance(25 * DAY)  # the take and the first analysis are 35 days old, the newer one 25
    report = store.gc(dry_run=False)
    assert report["items"]["takes"] == []
    assert report["items"]["analyses"] == [old_analysis.analysis_id]
    assert store.get_take_by_id(items["take"].take_id) is not None
    assert store.analyses_of(items["take"].take_id) == (newer,)
    assert not store.analysis_path(items["take"].take_id, old_analysis.analysis_id).exists()


def test_touching_an_analysis_uses_its_take(store: NarrationStore, clock: FakeClock) -> None:
    items = populate(store)
    clock.advance(29 * DAY)
    store.touch("analysis", items["analysis"].analysis_id)
    clock.advance(29 * DAY)
    assert store.gc()["items"]["takes"] == []


def test_verify_rehashes_every_immutable_file_s15(store: NarrationStore) -> None:
    items = populate(store)
    clean = store.verify()
    assert clean["checked"] == clean["ok"] > 0
    assert clean["missing"] == clean["mismatched"] == clean["writable"] == []
    assert clean["database"] == "ok"
    raw = Path(items["render"].raw.path)
    store_files.make_writable(raw)
    raw.write_bytes(b"tampered")
    delivery = Path(items["take"].delivery.path)
    store_files.discard(delivery)
    report = store.verify()
    rel_raw = store.layout.rel(raw)
    assert [m["path"] for m in report["mismatched"]] == [rel_raw]
    assert report["writable"] == [rel_raw]
    assert report["missing"] == [store.layout.rel(delivery)]


def test_a_new_profile_version_leaves_no_old_pictures_behind(store: NarrationStore) -> None:
    audio = sha(b"clip")
    folder = store.scratch_path("p1", "spectrogram.png").parent
    (folder / "spectrogram.png").write_bytes(b"old spectrogram")
    (folder / "pitch.png").write_bytes(b"old pitch")
    store.put_profile(profile_record(audio), folder)
    store.put_profile(dataclasses.replace(profile_record(audio, pictures=False), profile_version="profile-2"), None)
    assert sorted(os.listdir(store.profile_dir(audio))) == ["profile.json"]
    assert store.verify()["missing"] == []


# ---------------------------------------------------------------- review fixes: gc under concurrency, its report,
# what live items keep alive, and verify


def test_a_key_published_again_while_gc_runs_is_kept_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # gc renames each victim aside and deletes its rows under the write lock, and after the commit removes
    # only the renamed copies. A take published again in between goes to a fresh folder that gc never touches.
    render = store.put_render(render_record(), scratch_file(store, "raw.wav", audio_bytes("raw")))
    record = take_record(render.raw.sha256, render.render_id)
    first = store.put_take(record, scratch_file(store, "d.wav", b"delivery"))
    clock.advance(31 * DAY)
    real_remove = store._remove
    republished: list[TakeRecord] = []

    def publish_again_then_remove(path: Path, shown_as: Path, errors: list[dict[str, str]]) -> bool:
        if not republished:
            republished.append(store.put_take(record, scratch_file(store, "d2.wav", b"delivery")))
        return real_remove(path, shown_as, errors)

    monkeypatch.setattr(store, "_remove", publish_again_then_remove)
    report = store.gc(dry_run=False)
    assert report["items"]["takes"] == [first.take_id] and report["errors"] == []
    again = store.get_take(record.delivery_key)
    assert again is not None and Path(again.delivery.path).read_bytes() == b"delivery"
    assert store.verify()["missing"] == []


def test_an_orphan_published_meanwhile_is_not_collected_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    render = store.put_render(render_record(), scratch_file(store, "raw.wav", audio_bytes("raw")))
    record = take_record(render.raw.sha256, render.render_id)
    orphan = store.take_dir(record.take_id)
    orphan.mkdir(parents=True)
    (orphan / "delivery.wav").write_bytes(b"half a file from a crashed publish")
    clock.advance(2 * DAY)  # older than the grace period, inside the retention period
    real_leftovers = store._leftovers

    def list_then_publish(cutoff: float) -> tuple[list[Path], list[Path]]:
        found = real_leftovers(cutoff)
        assert orphan in found[1]
        store.put_take(record, scratch_file(store, "d.wav", b"delivery"))  # replaces the orphan
        return found

    monkeypatch.setattr(store, "_leftovers", list_then_publish)
    report = store.gc(dry_run=False)
    assert report["orphans"] == [] and report["errors"] == []
    take = store.get_take(record.delivery_key)
    assert take is not None and Path(take.delivery.path).read_bytes() == b"delivery"


def test_a_dry_run_reports_what_the_real_run_does_s15(store: NarrationStore, clock: FakeClock) -> None:
    # Leftovers and orphans are found before any row is deleted, so a real run does not also list the folders
    # of the items it just collected, and counts nothing twice.
    populate(store)
    plant_leftovers(store)
    clock.advance(31 * DAY)
    dry = store.gc()
    real = store.gc(dry_run=False)
    for field in ("items", "expired_leases", "leftovers", "orphans", "scratch", "bytes", "errors"):
        assert real[field] == dry[field], field
    assert real["orphans"] == ["takes/bb/tk_bbbbbbbbbbbbbbbb"]
    assert store.gc()["bytes"] == 0  # nothing left to collect


def test_gc_lists_what_it_cannot_move_aside_and_goes_on_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = populate(store)
    clock.advance(31 * DAY)
    render_dir = store.render_dir(items["render"].render_id)
    real = store_files.rename_retrying

    def refuse_the_render(src: Path, dst: Path, *, attempts: int = store_files.REPLACE_ATTEMPTS) -> None:
        if Path(src) == render_dir:
            raise PermissionError(13, "a file in it is open", str(src))
        real(src, dst, attempts=attempts)

    monkeypatch.setattr(store_files, "rename_retrying", refuse_the_render)
    report = store.gc(dry_run=False)
    assert report["items"]["renders"] == []
    assert [e["path"] for e in report["errors"]] == [store.layout.rel(render_dir)]
    assert store.get_render(items["render"].render_key) is not None  # its rows stay, for the next run
    assert report["items"]["takes"] == [items["take"].take_id]  # the rest went on


def test_gc_lists_what_it_cannot_remove_after_the_commit_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = populate(store)
    clock.advance(31 * DAY)
    take_id = items["take"].take_id
    real = store_files.remove_tree

    def refuse_the_take(path: Path) -> None:
        if path.name.startswith(".trash-") and path.name.endswith(take_id):
            raise PermissionError(13, "a file in it is open", str(path))
        real(path)

    monkeypatch.setattr(store_files, "remove_tree", refuse_the_take)
    report = store.gc(dry_run=False)
    assert [e["path"] for e in report["errors"]] == [store.layout.rel(store.take_dir(take_id))]
    assert report["items"]["takes"] == [take_id] and store.get_take_by_id(take_id) is None
    assert report["items"]["renders"] == [items["render"].render_id]
    monkeypatch.undo()
    # What was left under a .trash- name is a leftover the next run removes once it is past the grace period.
    clock.advance(2 * DAY)
    later = store.gc(dry_run=False)
    assert [p for p in later["leftovers"] if take_id in p] and later["errors"] == []
    assert not [p for p in store.root.rglob("*") if take_id in p.name]


def test_a_live_measurement_keeps_the_takes_it_names_s15(store: NarrationStore, clock: FakeClock) -> None:
    # Section 15 keeps measurement audio for measurement_retention_days; that audio is the calibration and
    # ladder takes the measurement names, kept with their renders and analyses.
    def publish(text: str, delivery: bytes) -> tuple[RenderRecord, TakeRecord]:
        render = store.put_render(render_record(text), scratch_file(store, "raw.wav", audio_bytes(text)))
        take = store.put_take(take_record(render.raw.sha256, render.render_id), scratch_file(store, "d.wav", delivery))
        return render, take

    render_a, take_a = publish("Calibration paragraph one.", b"a")
    analysis_a = store.put_analysis(analysis_record(take_a))
    render_b, take_b = publish("A ladder rung of eighty characters.", b"b")
    render_c, take_c = publish("Nobody measured this one.", b"c")
    measurement = store.put_measurement(
        measurement_record(calibration_takes=(take_a.take_id,), ladder_takes=(take_b.take_id, None))
    )
    clock.advance(31 * DAY)
    report = store.gc(dry_run=False)
    assert report["items"]["takes"] == [take_c.take_id]
    assert report["items"]["renders"] == [render_c.render_id]
    assert report["items"]["analyses"] == []
    assert store.get_take_by_id(take_a.take_id) == take_a and store.get_take_by_id(take_b.take_id) == take_b
    assert store.get_render(render_a.render_key) == render_a and store.get_render(render_b.render_key) == render_b
    assert store.analyses_of(take_a.take_id) == (analysis_a,)
    # Once the measurement itself expires, its takes go with it.
    clock.advance(365 * DAY)
    report = store.gc(dry_run=False)
    assert report["items"]["measurements"] == [measurement.measurement_key]
    assert report["items"]["takes"] == sorted([take_a.take_id, take_b.take_id])
    assert report["items"]["renders"] == sorted([render_a.render_id, render_b.render_id])


def test_a_live_design_keeps_its_candidates_profiles_s15(store: NarrationStore, clock: FakeClock) -> None:
    clip = audio_bytes("a designed clip")
    profile = profile_record(sha(clip), pictures=False)
    store.put_profile(profile, None)
    clock.advance(20 * DAY)
    design_id = KEYS.new_design_id()
    store.put_candidate(candidate(design_id, 0, profile=profile), scratch_file(store, "c.wav", clip))
    clock.advance(15 * DAY)  # the profile was last used 35 days ago, the design 15
    report = store.gc(dry_run=False)
    assert report["items"]["profiles"] == [] and report["items"]["designs"] == []
    assert store.get_profile(sha(clip), profile.profile_version) is not None


def test_using_a_design_uses_its_profiles_s15(store: NarrationStore, clock: FakeClock) -> None:
    clip = audio_bytes("a designed clip")
    profile = profile_record(sha(clip), pictures=False)
    store.put_profile(profile, None)
    design_id = KEYS.new_design_id()
    store.put_candidate(candidate(design_id, 0, profile=profile), scratch_file(store, "c.wav", clip))
    clock.advance(20 * DAY)
    store.touch("design", design_id)
    conn = sqlite3.connect(str(store.root / "narration.sqlite"))
    try:
        (used,) = conn.execute("SELECT last_used_at FROM profiles WHERE audio_sha256 = ?", (sha(clip),)).fetchone()
    finally:
        conn.close()
    assert used == clock.now


def test_verify_lists_a_path_it_cannot_check_and_goes_on_s15(store: NarrationStore) -> None:
    populate(store)
    conn = sqlite3.connect(str(store.root / "narration.sqlite"))
    try:
        with conn:
            conn.execute(
                "INSERT INTO files (rel_path, sha256, size, owner_kind, owner_id) VALUES (?, ?, 1, 'render', 'x')",
                ("renders/../../outside.wav", "0" * 64),
            )
    finally:
        conn.close()
    report = store.verify()
    assert [e["path"] for e in report["errors"]] == ["renders/../../outside.wav"]
    assert report["ok"] == report["checked"] - 1
    assert report["missing"] == report["mismatched"] == []


def test_a_gc_that_fails_midway_puts_every_folder_back_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = populate(store)
    clock.advance(31 * DAY)
    real = store._delete_rows
    calls = {"n": 0}

    def fail_on_the_second(conn: sqlite3.Connection, kind: str, ident: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the disk went away")
        real(conn, kind, ident)

    monkeypatch.setattr(store, "_delete_rows", fail_on_the_second)
    with pytest.raises(RuntimeError):
        store.gc(dry_run=False)
    monkeypatch.undo()
    # The rows were rolled back, so every folder is back under its own name.
    assert store.get_render(items["render"].render_key) == items["render"]
    assert store.get_take_by_id(items["take"].take_id) == items["take"]
    assert not [p for p in store.root.rglob("*") if p.name.startswith(".trash-")]
    assert store.verify()["missing"] == []


# ---------------------------------------------------------------- follow-ups: gc never holds the write lock for long


def _write_lock_is_free(store: NarrationStore) -> bool:
    """Whether another connection can take the write lock at once, without waiting."""
    conn = sqlite3.connect(str(store.root / "narration.sqlite"), timeout=0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def _old_renders(store: NarrationStore, clock: FakeClock, count: int) -> list[RenderRecord]:
    renders = [
        store.put_render(
            render_record(f"Line {i} of an invented batch."), scratch_file(store, "raw.wav", audio_bytes(str(i)))
        )
        for i in range(count)
    ]
    clock.advance(31 * DAY)
    return renders


def _count_per_transaction(store: NarrationStore, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many items each write transaction of the store collects, in order (0 for other transactions)."""
    per_txn: list[int] = []
    real_write, real_delete = store._write, store._delete_rows

    def counting_write(**kwargs: Any) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        per_txn.append(0)
        return real_write(**kwargs)

    def counting_delete(conn: sqlite3.Connection, kind: str, ident: str) -> None:
        per_txn[-1] += 1
        real_delete(conn, kind, ident)

    monkeypatch.setattr(store, "_write", counting_write)
    monkeypatch.setattr(store, "_delete_rows", counting_delete)
    return per_txn


def _tried_per_transaction(store: NarrationStore, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many items each write transaction of the store tried to rename to trash, in order."""
    per_txn: list[int] = []
    real_write, real_to_trash = store._write, store._to_trash

    def counting_write(**kwargs: Any) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        per_txn.append(0)
        return real_write(**kwargs)

    def counting_to_trash(path: Path, renames: Any, errors: list[dict[str, str]]) -> bool:
        per_txn[-1] += 1
        return real_to_trash(path, renames, errors)

    monkeypatch.setattr(store, "_write", counting_write)
    monkeypatch.setattr(store, "_to_trash", counting_to_trash)
    return per_txn


def _refuse_renames(monkeypatch: pytest.MonkeyPatch, folders: set[Path]) -> None:
    """Every rename of one of ``folders`` fails, as it does while a file in it is open."""
    real_rename = os.rename

    def rename(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(src) in folders:
            raise PermissionError(13, "a file in it is open", os.fspath(src))
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", rename)


@pytest.mark.parametrize(("seconds", "expected"), [(0.0, [1, 1, 1, 1, 1, 1]), (3600.0, [2, 2, 2])])
def test_gc_bounds_a_transaction_by_the_items_it_tries_s15(
    store: NarrationStore,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    seconds: float,
    expected: list[int],
) -> None:
    # Items that fail to rename close a transaction as surely as items collected. The review's experiment:
    # with 2 items and 0 s per transaction, six refused renames all ran in one transaction.
    renders = _old_renders(store, clock, 6)
    _refuse_renames(monkeypatch, {store.render_dir(r.render_id) for r in renders})
    monkeypatch.setattr(store_module, "GC_BATCH_ITEMS", 2)
    monkeypatch.setattr(store_module, "GC_BATCH_SECONDS", seconds)
    per_txn = _tried_per_transaction(store, monkeypatch)
    report = store.gc(dry_run=False)
    assert report["items"]["renders"] == [] and len(report["errors"]) == 6
    assert [n for n in per_txn if n] == expected


@pytest.mark.parametrize(("seconds", "expected"), [(0.0, [1, 1, 1]), (3600.0, [2, 1])])
def test_gc_bounds_an_orphan_transaction_by_the_items_it_tries_s15(
    store: NarrationStore,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    seconds: float,
    expected: list[int],
) -> None:
    # Folders the index does not know (a crash between the rename and the index row), none renameable.
    months_ago = clock.now - 90 * DAY
    orphans = {store.root / "renders" / "ab" / f"rn_{i:016x}" for i in range(3)}
    for folder in orphans:
        folder.mkdir(parents=True)
        os.utime(folder, (months_ago, months_ago))
    _refuse_renames(monkeypatch, orphans)
    monkeypatch.setattr(store_module, "GC_BATCH_ITEMS", 2)
    monkeypatch.setattr(store_module, "GC_BATCH_SECONDS", seconds)
    per_txn = _tried_per_transaction(store, monkeypatch)
    report = store.gc(dry_run=False)
    assert report["orphans"] == [] and len(report["errors"]) == 3
    assert [n for n in per_txn if n] == expected


def test_gc_collects_in_bounded_transactions_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A first gc over thousands of folders must not hold the write lock past the busy timeout every other
    # writer waits for: it commits every GC_BATCH_ITEMS items, and the lock is free while it removes trash.
    renders = _old_renders(store, clock, 5)
    monkeypatch.setattr(store_module, "GC_BATCH_ITEMS", 2)
    monkeypatch.setattr(store_module, "GC_BATCH_SECONDS", 3600.0)  # only the count bounds it here
    per_txn = _count_per_transaction(store, monkeypatch)
    lock_free: list[bool] = []
    real_remove = store._remove

    def remove_and_probe(path: Path, shown_as: Path, errors: list[dict[str, str]]) -> bool:
        lock_free.append(_write_lock_is_free(store))
        return real_remove(path, shown_as, errors)

    monkeypatch.setattr(store, "_remove", remove_and_probe)
    report = store.gc(dry_run=False)
    assert report["items"]["renders"] == sorted(r.render_id for r in renders)
    assert [n for n in per_txn if n] == [2, 2, 1]
    assert len(lock_free) >= 5 and all(lock_free)  # five renders' trash, then the old scratch folder


def test_gc_commits_after_a_few_seconds_whatever_the_count_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _old_renders(store, clock, 3)
    monkeypatch.setattr(store_module, "GC_BATCH_SECONDS", 0.0)
    per_txn = _count_per_transaction(store, monkeypatch)
    report = store.gc(dry_run=False)
    assert len(report["items"]["renders"]) == 3
    assert [n for n in per_txn if n] == [1, 1, 1]  # each transaction still takes one item


def test_gc_tries_a_folder_in_use_once_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under the write lock, a folder in use is not waited for: tried once, listed, and kept for the next run.
    renders = sorted(_old_renders(store, clock, 2), key=lambda r: r.render_id)
    locked = store.render_dir(renders[0].render_id)
    tries: list[str] = []
    real_rename = os.rename

    def rename(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(src) == locked:
            tries.append(os.fspath(src))
            raise PermissionError(13, "a file in it is open", os.fspath(src))
        real_rename(src, dst)

    def no_waiting(attempt: int) -> None:
        raise AssertionError("gc waited for a folder in use")

    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(store_files, "_retry_sleep", no_waiting)
    report = store.gc(dry_run=False)
    assert len(tries) == 1
    assert report["items"]["renders"] == [renders[1].render_id]
    assert [e["path"] for e in report["errors"]] == [store.layout.rel(locked)]
    monkeypatch.undo()
    assert store.get_render(renders[0].render_key) == renders[0]


def test_an_item_used_while_gc_runs_is_kept_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each transaction checks its items against a plan made under its own lock.
    first, second = sorted(_old_renders(store, clock, 2), key=lambda r: r.render_id)
    monkeypatch.setattr(store_module, "GC_BATCH_ITEMS", 1)
    real_remove = store._remove

    def use_the_second_meanwhile(path: Path, shown_as: Path, errors: list[dict[str, str]]) -> bool:
        store.touch("render", second.render_id)
        return real_remove(path, shown_as, errors)

    monkeypatch.setattr(store, "_remove", use_the_second_meanwhile)
    report = store.gc(dry_run=False)
    assert report["items"]["renders"] == [first.render_id]
    monkeypatch.undo()
    assert store.get_render(second.render_key) == second


def test_a_failed_gc_transaction_puts_folders_back_before_it_releases_the_lock_s15(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The renames are undone inside the transaction: once the ROLLBACK makes the rows visible again, their
    # folders are already back under their own names.
    items = populate(store)
    clock.advance(31 * DAY)
    folders = [store.render_dir(items["render"].render_id), store.take_dir(items["take"].take_id)]
    real_delete = store._delete_rows
    calls = {"n": 0}

    def fail_on_the_second(conn: sqlite3.Connection, kind: str, ident: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the disk went away")
        real_delete(conn, kind, ident)

    seen = scripted(monkeypatch, store, look=lambda: [f.is_dir() for f in folders]).seen
    monkeypatch.setattr(store, "_delete_rows", fail_on_the_second)
    with pytest.raises(RuntimeError):
        store.gc(dry_run=False)
    assert seen == [[True, True]]


# ---------------------------------------------------------------- follow-ups: a trash name's age is written in it


def test_a_trash_name_carries_the_moment_of_the_rename_s15(tmp_path: Path) -> None:
    trash = store_files.trash_name(tmp_path / "profile.json", 1_790_000_000.7)
    assert trash.parent == tmp_path
    assert re.fullmatch(r"\.trash-1790000000-[0-9a-f]{16}-profile\.json", trash.name)
    assert store_files.trash_time(trash.name) == 1_790_000_000.0
    # The older layout (.trash-<16 hex token>-<name>) and any other name carry no time.
    assert store_files.trash_time(".trash-0123456789abcdef-profile.json") is None
    assert store_files.trash_time(".trash-1234567890123456-profile.json") is None
    assert store_files.trash_time(".tmp-0123456789abcdef-profile.json") is None
    assert store_files.trash_time("profile.json") is None


def test_gc_judges_a_trash_name_by_its_time_not_its_mtime_s15(store: NarrationStore, clock: FakeClock) -> None:
    # A publisher renames a published folder, which may be months old, to a trash name, and renames it back
    # if its commit fails. gc must not take that folder for an old leftover because of its old mtime.
    months_ago = clock.now - 90 * DAY
    shard = store.root / "profiles" / "ab"
    shard.mkdir(parents=True)

    def trash_folder(name: str, mtime: float) -> Path:
        folder = shard / name
        folder.mkdir()
        (folder / "profile.json").write_bytes(b"{}")
        for path in (folder / "profile.json", folder):
            os.utime(path, (mtime, mtime))
        return folder

    just_made = trash_folder(store_files.trash_name(shard / ("ab" * 32), clock.now).name, months_ago)
    token_and_name = just_made.name.split("-", 2)[2]
    long_made = trash_folder(f".trash-{int(months_ago)}-{token_and_name}", clock.now)  # a recent mtime
    legacy = trash_folder(".trash-0123456789abcdef-" + "ab" * 32, months_ago)  # no time: judged by mtime
    report = store.gc(dry_run=False)
    assert just_made.is_dir()
    assert not long_made.exists() and not legacy.exists()
    assert sorted(report["leftovers"]) == sorted(store.layout.rel(p) for p in (long_made, legacy))
    clock.advance(2 * DAY)  # past the grace period since the rename
    store.gc(dry_run=False)
    assert not just_made.exists()
