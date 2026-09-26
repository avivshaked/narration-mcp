"""Measurements, profiles, designs, provenance, engine profiles, the canary, the alignment benchmark and the
daemon's status and commands (design sections 3, 4.1, 6, 10.1, 11.2, 15, 17.4)."""

from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from narration.config import Config
from narration.contracts import names
from narration.contracts.models import EngineProfile, ProvenanceEntry
from narration.keys import Keys
from narration.store import (
    InvalidIdError,
    NarrationStore,
    NotFoundError,
    StoreError,
    StoreIntegrityError,
)
from narration.store import db as store_db
from narration.store import files as store_files

from .conftest import FakeClock
from .factories import (
    CLIP_SHA,
    METHOD_ID,
    VOICE_HASH,
    alignment_benchmark,
    audio_bytes,
    canary_pin,
    candidate,
    daemon_status,
    engine_profile,
    measurement_record,
    profile_record,
    scratch_file,
    sha,
)
from .standin import StandInPlatform
from .txn_fakes import commit_fails

DESIGN_ID = Keys().new_design_id()


# ---------------------------------------------------------------- measurements (section 3.2)


def test_a_measurement_is_kept_per_voice_and_engine_profile_s3_2(store: NarrationStore) -> None:
    record = measurement_record()
    assert store.put_measurement(record) == record
    folder = store.measurement_dir(VOICE_HASH, names.ENGINE_PROFILE_BASE)
    assert store_files.is_readonly(folder / "measurement.json")
    assert store.get_measurement(VOICE_HASH, names.ENGINE_PROFILE_BASE) == record
    assert store.put_measurement(record) == record  # the same key again is a no-op
    other_engine = measurement_record(engine_id="qwen3-base-1.7b.p2")
    store.put_measurement(other_engine)
    assert store.measurements_of(VOICE_HASH) == (record, other_engine)
    assert store.measurements_of("sha256:" + "0" * 64) == ()


def test_a_measurement_under_a_new_key_replaces_the_old_s10_2(store: NarrationStore) -> None:
    store.put_measurement(measurement_record())
    newer = measurement_record(corpus_hex="c1" * 32)
    assert store.put_measurement(newer) == newer
    assert store.get_measurement(VOICE_HASH, names.ENGINE_PROFILE_BASE) == newer
    assert store.verify()["mismatched"] == []


# ---------------------------------------------------------------- voice profiles (section 3.6)


def test_a_profile_moves_its_pictures_into_place_s3_6(store: NarrationStore) -> None:
    audio = sha(b"some clip")
    pictures = store.scratch_path("profile-job", "spectrogram.png")
    pictures.write_bytes(b"png 1")
    (pictures.parent / "pitch.png").write_bytes(b"png 2")
    published = store.put_profile(profile_record(audio), pictures.parent)
    folder = store.profile_dir(audio)
    assert published.pictures.spectrogram == str(folder / "spectrogram.png")
    assert Path(published.pictures.pitch).read_bytes() == b"png 2"
    assert not pictures.exists()
    assert store.get_profile(audio, names.PROFILE_VERSION) == published
    assert store.get_profile(audio, "profile-0") is None


def test_a_newer_profile_version_replaces_the_old_s3_6(store: NarrationStore) -> None:
    audio = sha(b"clip")
    store.put_profile(profile_record(audio, pictures=False), None)
    newer = store.put_profile(profile_record(audio, version="profile-2", pictures=False), None)
    assert store.get_profile(audio, "profile-2") == newer
    assert store.get_profile(audio, names.PROFILE_VERSION) is None


def test_pictures_without_a_folder_are_refused(store: NarrationStore) -> None:
    with pytest.raises(StoreError, match="pictures_dir"):
        store.put_profile(profile_record(sha(b"clip")), None)


# ---------------------------------------------------------------- designs (section 3.1)


def test_candidates_are_published_with_their_clip_and_read_back_by_design_s3_1(store: NarrationStore) -> None:
    profile_audio = sha(audio_bytes("cand-1"))
    profile = store.put_profile(profile_record(profile_audio, pictures=False), None)
    first = store.put_candidate(candidate(DESIGN_ID, 1, profile), scratch_file(store, "c1.wav", audio_bytes("cand-1")))
    zeroth = store.put_candidate(candidate(DESIGN_ID, 0), scratch_file(store, "c0.wav", audio_bytes("cand-0")))
    folder = store.design_dir(DESIGN_ID, 1)
    assert first.clip.path == str(folder / "clip.wav") and first.clip.sha256 == sha(audio_bytes("cand-1"))
    assert store_files.is_readonly(folder / "clip.wav") and (folder / "candidate.json").is_file()
    assert first.profile == profile
    assert store.get_design(DESIGN_ID) == (zeroth, first)
    assert store.get_design(Keys().new_design_id()) == ()


# ---------------------------------------------------------------- provenance (section 17.4)


def test_provenance_is_append_only_and_indexed_s17_4(store: NarrationStore) -> None:
    entry = ProvenanceEntry(clip_sha256=CLIP_SHA, design_id=DESIGN_ID, date="2026-09-26")
    assert not store.is_provenance(CLIP_SHA)
    store.add_provenance(entry)
    store.add_provenance(entry)  # once only
    other = ProvenanceEntry(clip_sha256="6c" * 32, design_id=DESIGN_ID, date="2026-09-26")
    store.add_provenance(other)
    assert store.is_provenance(CLIP_SHA) and store.is_provenance("6c" * 32)
    lines = (store.root / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["clip_sha256"] for line in lines] == [CLIP_SHA, "6c" * 32]


def test_provenance_survives_a_torn_line_and_a_lost_index_s17_4(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    good = {"clip_sha256": CLIP_SHA, "design_id": DESIGN_ID, "date": "2026-09-26"}
    # A line written before the database knew it, then a line cut short by a crash.
    (root / "provenance.jsonl").write_bytes((json.dumps(good) + "\n" + '{"clip_sha256": "6c6c').encode("utf-8"))
    with NarrationStore(root, StandInPlatform()) as store:
        assert store.is_provenance(CLIP_SHA)
        store.add_provenance(ProvenanceEntry(clip_sha256="7d" * 32, design_id=DESIGN_ID, date="2026-09-27"))
        assert store.is_provenance("7d" * 32)
    lines = (root / "provenance.jsonl").read_bytes().split(b"\n")
    assert json.loads(lines[-2])["clip_sha256"] == "7d" * 32  # on its own line, after the torn one


# ---------------------------------------------------------------- engine profiles and the canary (10.1, DC-3)


def test_engine_profiles_are_pinned_and_current_per_kind_s10_1(store: NarrationStore) -> None:
    assert store.current_engine_profile("base") is None
    base = store.put_engine_profile(engine_profile())
    design = store.put_engine_profile(engine_profile(names.ENGINE_PROFILE_DESIGN, hash_="sha256:" + "d1" * 32))
    store.set_current_engine_profile("base", base.engine_profile_id)
    store.set_current_engine_profile("design", design.engine_profile_id)
    assert store.current_engine_profile("base") == base
    assert store.current_engine_profile("design") == design
    assert store.list_engine_profiles() == (base, design)
    assert (store.root / "engines" / f"{names.ENGINE_PROFILE_BASE}.json").is_file()
    with pytest.raises(NotFoundError):
        store.set_current_engine_profile("base", "qwen3-base-1.7b.p9")
    with pytest.raises(ValueError):
        store.current_engine_profile("other")  # type: ignore[arg-type]


def test_a_pinned_profile_never_changes_its_hash_s10_1(store: NarrationStore) -> None:
    store.put_engine_profile(engine_profile())
    with pytest.raises(StoreIntegrityError, match="new pin"):
        store.put_engine_profile(engine_profile(hash_="sha256:" + "00" * 32))
    assert not [p for p in (store.root / "engines").iterdir() if p.name.startswith(".tmp-")]


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "float16"},
        {"model_revision": "1" * 40},
        {"packages": {"qwen-tts": "0.1.2"}},
        {"settings": {"non_streaming_mode": True}},
        {"vram_need_mb": 8000},
    ],
)
def test_a_pinned_profile_never_changes_a_hashed_field_s10_1(store: NarrationStore, change: dict[str, Any]) -> None:
    # The same id and the same hash string with a different hashed field is still a changed pin.
    pinned = store.put_engine_profile(engine_profile())
    with pytest.raises(StoreIntegrityError, match="new pin"):
        store.put_engine_profile(dataclasses.replace(engine_profile(), **change))
    assert store.get_engine_profile(pinned.engine_profile_id) == pinned


def test_the_fields_the_hash_leaves_out_may_change_after_the_pin_s10_1(store: NarrationStore) -> None:
    store.put_engine_profile(engine_profile())
    clip = store.put_canary_clip(names.ENGINE_PROFILE_BASE, scratch_file(store, "canary.wav", b"canary"))
    later = dataclasses.replace(
        engine_profile(),
        snapshot_dir="<models_root>/elsewhere",
        observed={"gpu": "a GPU", "driver": "1.0"},
        tier="bit_exact",
        canary=canary_pin(clip),
    )
    assert store.put_engine_profile(later) == later


def test_the_canary_clip_is_kept_with_its_profile_dc_3(store: NarrationStore) -> None:
    clip = store.put_canary_clip(names.ENGINE_PROFILE_BASE, scratch_file(store, "canary.wav", b"canary"))
    assert clip.path == str(store.root / "engines" / names.ENGINE_PROFILE_BASE / "canary.wav")
    assert clip.sha256 == sha(b"canary") and store_files.is_readonly(Path(clip.path))
    pinned = store.put_engine_profile(dataclasses.replace(engine_profile(), canary=canary_pin(clip)))
    assert pinned.canary is not None and pinned.canary.clip == clip
    raw = json.loads((store.root / "engines" / f"{names.ENGINE_PROFILE_BASE}.json").read_text(encoding="utf-8"))
    assert raw["canary"]["clip"]["path"] == f"engines/{names.ENGINE_PROFILE_BASE}/canary.wav"
    # The same clip again is fine; another clip for a pinned canary is a new pin.
    assert store.put_canary_clip(names.ENGINE_PROFILE_BASE, scratch_file(store, "c2.wav", b"canary")) == clip
    with pytest.raises(StoreIntegrityError):
        store.put_canary_clip(names.ENGINE_PROFILE_BASE, scratch_file(store, "c3.wav", b"another canary"))
    assert Path(clip.path).read_bytes() == b"canary"


# ---------------------------------------------------------------- the alignment benchmark (section 11.2)


def test_the_configured_aligners_benchmark_is_current_s11_2(store: NarrationStore) -> None:
    # The store fixture is configured with METHOD_ID.
    assert store.current_alignment_benchmark() is None
    small = store.put_alignment_benchmark(alignment_benchmark(p50=0.04))
    assert store.current_alignment_benchmark() == small
    assert (store.root / "alignment" / "ctc-snap%2Fwav2vec2%40abc.json").is_file()
    full = store.put_alignment_benchmark(alignment_benchmark(p50=0.03))
    assert store.get_alignment_benchmark(full.method_id) == full == store.current_alignment_benchmark()
    # Another aligner's benchmark, published later, is not the one in use.
    other = store.put_alignment_benchmark(alignment_benchmark("qwen-forced-aligner@def", p50=0.02))
    assert store.get_alignment_benchmark(other.method_id) == other
    assert store.current_alignment_benchmark() == full


def test_without_a_configured_aligner_no_benchmark_is_current_s11_2(tmp_path: Path) -> None:
    with NarrationStore(tmp_path / "store", StandInPlatform()) as bare:
        bare.put_alignment_benchmark(alignment_benchmark())
        assert bare.current_alignment_benchmark() is None
    config = Config.for_tests(tmp_path / "store")
    with NarrationStore.from_config(config, StandInPlatform(), alignment_method_id=METHOD_ID) as configured:
        assert configured.current_alignment_benchmark() == alignment_benchmark()
    with pytest.raises(InvalidIdError):
        NarrationStore(tmp_path / "store", StandInPlatform(), alignment_method_id="")


# ---------------------------------------------------------------- the daemon's status and commands (4.1, 7.6)


def test_daemon_status_is_run_daemon_json_s15(store: NarrationStore) -> None:
    assert store.get_daemon_status() is None
    store.put_daemon_status(daemon_status())
    assert store.get_daemon_status() == daemon_status()
    assert (store.root / "run" / "daemon.json").is_file()


def test_commands_reach_the_daemon_through_the_store_s4(store: NarrationStore) -> None:
    posted = store.post_command("release_gpu")
    stop = store.post_command("stop")
    assert store.pending_commands() == (posted, stop)
    assert store.wait_for_command(posted.command_id, timeout_s=0.05) is None

    def daemon() -> None:
        store.complete_command(posted.command_id, {"released": True, "holder_before": "qwen", "busy_job": None})

    thread = threading.Thread(target=daemon)
    thread.start()
    done = store.wait_for_command(posted.command_id, timeout_s=10)
    thread.join()
    assert done is not None and done.result == {"released": True, "holder_before": "qwen", "busy_job": None}
    assert store.pending_commands() == (stop,)
    assert store.complete_command(posted.command_id, {"released": False}).result == done.result  # first outcome kept
    with pytest.raises(NotFoundError):
        store.complete_command("01JBYQ7Z3M8V4T2R9K6N5P0W1C", None)
    with pytest.raises(ValueError):
        store.post_command("reboot")  # type: ignore[arg-type]


# ---------------------------------------------------------------- dropping rows whose files are gone


def _profile_row(store: NarrationStore, audio: str) -> tuple[str, str] | None:
    conn = sqlite3.connect(str(store.root / "narration.sqlite"))
    try:
        row = conn.execute("SELECT record, rel_dir FROM profiles WHERE audio_sha256 = ?", (audio,)).fetchone()
    finally:
        conn.close()
    return (str(row[0]), str(row[1])) if row else None


def test_a_row_replaced_meanwhile_is_never_dropped_as_missing(store: NarrationStore) -> None:
    # A reader that found a row's file missing drops the row only if, under the write lock, the row is still
    # the one it read and its file is still missing.
    audio = sha(b"clip")
    store.put_profile(profile_record(audio, pictures=False), None)
    old = _profile_row(store, audio)
    assert old is not None
    newer = store.put_profile(profile_record(audio, version="profile-2", pictures=False), None)
    rel = f"{old[1]}/profile.json"
    store._drop("profile", audio, old[0], rel)  # a stale reader of the old version
    assert store.get_profile(audio, "profile-2") == newer
    current = _profile_row(store, audio)
    assert current is not None
    store._drop("profile", audio, current[0], rel)  # the same row, but its file is there
    assert store.get_profile(audio, "profile-2") == newer
    store_files.discard(store.root / rel)
    store._drop("profile", audio, current[0], rel)  # the same row, its file still missing: dropped
    assert _profile_row(store, audio) is None


# ---------------------------------------------------------------- follow-ups: a canary clip is never destroyed


def test_a_refused_canary_clip_goes_back_to_scratch_dc_3(store: NarrationStore) -> None:
    # The canary is designed on the GPU: a clip the store refuses goes back where the caller had it.
    clip = store.put_canary_clip(names.ENGINE_PROFILE_BASE, scratch_file(store, "canary.wav", b"canary"))
    store.put_engine_profile(dataclasses.replace(engine_profile(), canary=canary_pin(clip)))
    other = scratch_file(store, "c3.wav", b"another canary")
    with pytest.raises(StoreIntegrityError):
        store.put_canary_clip(names.ENGINE_PROFILE_BASE, other)
    assert other.read_bytes() == b"another canary" and not store_files.is_readonly(other)
    assert Path(clip.path).read_bytes() == b"canary"
    assert not [p for p in store.root.rglob("*") if p.name.startswith((".tmp-", ".trash-"))]


def test_a_canary_clip_whose_commit_fails_goes_back_and_the_old_one_stays_dc_3(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = store.put_canary_clip(names.ENGINE_PROFILE_BASE, scratch_file(store, "canary.wav", b"canary"))
    other = scratch_file(store, "c2.wav", b"a newer canary")
    monkeypatch.setattr(store_db, "write_txn", commit_fails)
    with pytest.raises(sqlite3.OperationalError):
        store.put_canary_clip(names.ENGINE_PROFILE_BASE, other)
    monkeypatch.undo()
    assert other.read_bytes() == b"a newer canary" and not store_files.is_readonly(other)
    assert Path(clip.path).read_bytes() == b"canary" and store_files.is_readonly(Path(clip.path))
    assert store.verify()["mismatched"] == []
    assert not [p for p in store.root.rglob("*") if p.name.startswith((".tmp-", ".trash-"))]


def test_a_canary_clip_that_waited_in_scratch_survives_a_gc_during_its_publish_dc_3(
    store: NarrationStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The clip is moved to a .tmp- name before its publish takes the write lock. A gc in between must not
    # take it for an old leftover because it waited in scratch/ for months (a rename keeps the mtime).
    src = scratch_file(store, "canary.wav", b"canary")
    months_ago = clock.now - 90 * 86_400
    os.utime(src, (months_ago, months_ago))
    get_engine_profile = store.get_engine_profile

    def gc_first(engine_profile_id: str) -> EngineProfile | None:  # called between the move and the publish
        store.gc(dry_run=False)
        return get_engine_profile(engine_profile_id)

    monkeypatch.setattr(store, "get_engine_profile", gc_first)
    clip = store.put_canary_clip(names.ENGINE_PROFILE_BASE, src)
    assert Path(clip.path).read_bytes() == b"canary"
