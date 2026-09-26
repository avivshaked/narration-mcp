"""Retention, gc and verify (design section 15)."""

from __future__ import annotations

import dataclasses
import os
import sqlite3
from pathlib import Path
from typing import Any

from narration.contracts.models import ProvenanceEntry
from narration.keys import Keys
from narration.store import NarrationStore
from narration.store import files as store_files

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
