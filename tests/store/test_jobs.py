"""Jobs and the queue (design sections 4, 6, 7.3, 8): idempotent creation, priority then FIFO, atomic
claims and compare-and-set updates."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

import pytest

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import JobSegment
from narration.keys import Keys
from narration.store import InvalidIdError, NarrationStore, NotFoundError
from narration.store import files as store_files

from .conftest import FakeClock
from .factories import job_record
from .store_utils import utc

KEYS = Keys()


def new_id() -> str:
    return KEYS.new_job_id()


def test_create_job_returns_the_active_job_for_the_same_request_s7_3(store: NarrationStore) -> None:
    first, created = store.create_job(job_record(new_id()))
    assert created
    again, created_again = store.create_job(job_record(new_id()))
    assert (again, created_again) == (first, False)
    assert store.get_job(first.job_id) == first
    other, created_other = store.create_job(job_record(new_id(), request_sha256="b" * 64))
    assert created_other and other.job_id != first.job_id


def test_a_retry_with_the_same_key_and_request_returns_the_job_s7_3(store: NarrationStore) -> None:
    first, _ = store.create_job(job_record(new_id(), idempotency_key="retry-1"))
    retried, created = store.create_job(job_record(new_id(), idempotency_key="retry-1"))
    assert (retried, created) == (first, False)


def test_a_reused_idempotency_key_with_another_request_is_refused_dc6(store: NarrationStore) -> None:
    first, _ = store.create_job(job_record(new_id(), idempotency_key="retry-1"))
    with pytest.raises(NarrationError) as caught:
        store.create_job(job_record(new_id(), request_sha256="c" * 64, idempotency_key="retry-1"))
    err = caught.value
    assert (err.code, err.field) == (codes.INVALID_ARGUMENT, "idempotency_key")
    assert err.hint == "Use a new key for a different request."
    assert store.queued_jobs() == (first,)  # nothing was queued for the other request
    # The key is free again once its job has finished, and another tool's job does not hold it.
    other_kind, created = store.create_job(
        job_record(new_id(), kind="analyse", request_sha256="c" * 64, idempotency_key="retry-1")
    )
    assert created
    store.update_job(first.job_id, status="completed", outcome="all_passed")
    _, created = store.create_job(job_record(new_id(), request_sha256="c" * 64, idempotency_key="retry-1"))
    assert created and other_kind.kind == "analyse"


def test_a_finished_job_does_not_absorb_a_new_request_s7_3(store: NarrationStore) -> None:
    first, _ = store.create_job(job_record(new_id(), idempotency_key="k"))
    store.update_job(first.job_id, status="completed", outcome="all_passed")
    second, created = store.create_job(job_record(new_id(), idempotency_key="k"))
    assert created and second.job_id != first.job_id


def _cancelling(store: NarrationStore, job_id: str) -> None:
    assert store.claim_job(job_id, "a-daemon") is not None
    assert store.update_job(job_id, expect_status="running", status="cancelling")


def test_a_job_being_cancelled_does_not_absorb_the_same_request_s7_3(store: NarrationStore) -> None:
    first, _ = store.create_job(job_record(new_id()))
    _cancelling(store, first.job_id)
    second, created = store.create_job(job_record(new_id()))
    assert created and second.job_id != first.job_id
    assert second.status == "queued"


def test_a_job_being_cancelled_no_longer_holds_its_idempotency_key_dc6(store: NarrationStore) -> None:
    first, _ = store.create_job(job_record(new_id(), idempotency_key="retry-1"))
    _cancelling(store, first.job_id)
    other, created = store.create_job(job_record(new_id(), request_sha256="d" * 64, idempotency_key="retry-1"))
    assert created and other.job_id != first.job_id


def test_the_same_request_under_another_tool_is_another_job(store: NarrationStore) -> None:
    store.create_job(job_record(new_id(), kind="generate"))
    _, created = store.create_job(job_record(new_id(), kind="analyse"))
    assert created


def test_a_job_row_writes_job_json_s15(store: NarrationStore) -> None:
    job, _ = store.create_job(job_record(new_id()))
    data = json.loads((store.job_dir(job.job_id) / "job.json").read_text(encoding="utf-8"))
    assert data["job_id"] == job.job_id and data["status"] == "queued"
    store.update_job(job.job_id, status="running", phase="rendering")
    data = json.loads((store.job_dir(job.job_id) / "job.json").read_text(encoding="utf-8"))
    assert (data["status"], data["phase"]) == ("running", "rendering")


def test_a_job_json_that_cannot_be_written_never_fails_the_change(
    store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The row is the record; job.json is a copy for people. A copy that cannot be replaced (held open on
    # Windows, a full disk) is logged, and the job's creation and status changes still happen.
    real = store_files.write_atomic

    def refuse_job_json(path: Any, data: Any, **kw: Any) -> Any:
        if path.name == "job.json":
            raise PermissionError(13, "held open by another program", str(path))
        return real(path, data, **kw)

    monkeypatch.setattr(store_files, "write_atomic", refuse_job_json)
    job, created = store.create_job(job_record(new_id()))
    assert created and store.get_job(job.job_id) == job
    running = store.claim_job(job.job_id, "daemon")
    assert running is not None and running.status == "running"
    done = store.update_job(job.job_id, status="completed", outcome="all_passed")
    assert done is not None and store.get_job(job.job_id) == done
    assert not (store.job_dir(job.job_id) / "job.json").exists()


def test_the_queue_runs_by_priority_then_fifo_s4(store: NarrationStore) -> None:
    batch_1, _ = store.create_job(job_record(new_id(), request_sha256="1" * 64))
    batch_2, _ = store.create_job(job_record(new_id(), request_sha256="2" * 64))
    urgent, _ = store.create_job(job_record(new_id(), request_sha256="3" * 64, priority="interactive"))
    assert [j.job_id for j in store.queued_jobs()] == [urgent.job_id, batch_1.job_id, batch_2.job_id]
    claimed = [store.claim_next_job("daemon") for _ in range(4)]
    assert [j.job_id if j else None for j in claimed] == [urgent.job_id, batch_1.job_id, batch_2.job_id, None]
    assert all(j is not None and j.status == "running" for j in claimed[:3])
    assert [j.status for j in store.queued_jobs()] == ["running"] * 3


def test_claim_job_is_a_compare_and_set_from_queued(store: NarrationStore) -> None:
    job, _ = store.create_job(job_record(new_id()))
    running = store.claim_job(job.job_id, "daemon")
    assert running is not None and running.status == "running"
    assert store.claim_job(job.job_id, "daemon") is None
    assert store.claim_job(new_id(), "daemon") is None


def test_concurrent_claims_start_a_job_once(store: NarrationStore) -> None:
    job, _ = store.create_job(job_record(new_id()))
    results: list[object] = []
    lock = threading.Lock()

    def claim() -> None:
        got = store.claim_next_job("daemon")
        with lock:
            results.append(got)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert [r.job_id for r in results if r is not None] == [job.job_id]  # type: ignore[attr-defined]


def test_update_job_with_expect_status_never_reverts_a_cancel_s7_4(store: NarrationStore, clock: FakeClock) -> None:
    job, _ = store.create_job(job_record(new_id()))
    store.claim_job(job.job_id, "daemon")
    clock.advance(5)
    cancelling = store.update_job(job.job_id, expect_status="running", status="cancelling")
    assert cancelling is not None and cancelling.status == "cancelling"
    assert cancelling.updated_at == utc(clock.now)
    # The daemon's blind progress write, expecting "running", changes nothing.
    assert store.update_job(job.job_id, expect_status="running", phase="scoring") is None
    assert store.get_job(job.job_id) == cancelling


def test_update_job_keeps_the_jobs_identity(store: NarrationStore) -> None:
    job, _ = store.create_job(job_record(new_id()))
    for field in ("kind", "request", "request_sha256", "idempotency_key", "created_at"):
        changes: dict[str, Any] = {field: "x"}
        with pytest.raises(ValueError, match="never change"):
            store.update_job(job.job_id, **changes)
    with pytest.raises(ValueError):
        store.update_job(job.job_id, status="exploded")
    with pytest.raises(NotFoundError):
        store.update_job(new_id(), status="running")


def test_update_job_stores_the_per_segment_items(store: NarrationStore) -> None:
    job, _ = store.create_job(job_record(new_id()))
    items = (JobSegment(segment_id="p01", state="rendering"),)
    store.update_job(job.job_id, items=items, message="Round 0: rendering p01")
    fetched = store.get_job(job.job_id)
    assert fetched is not None and fetched.items == items and fetched.message == "Round 0: rendering p01"


def test_jobs_created_since_counts_across_statuses_for_the_rate_cap(store: NarrationStore, clock: FakeClock) -> None:
    before = utc(clock.now)
    a, _ = store.create_job(job_record(new_id(), request_sha256="1" * 64))
    clock.advance(30)
    store.create_job(job_record(new_id(), request_sha256="2" * 64))
    store.update_job(a.job_id, status="completed")
    assert store.jobs_created_since(before) == 2
    assert store.jobs_created_since(utc(clock.now - 10)) == 1
    assert store.jobs_created_since(utc(clock.now + 1)) == 0


def test_iter_jobs_lists_jobs_newest_first_filtered_and_uses_none_wp48(store: NarrationStore, clock: FakeClock) -> None:
    first, _ = store.create_job(job_record(new_id(), request_sha256="1" * 64))
    store.update_job(first.job_id, status="completed")
    clock.advance(3600)
    second, _ = store.create_job(job_record(new_id(), request_sha256="2" * 64, kind="measure"))
    third, _ = store.create_job(job_record(new_id(), request_sha256="3" * 64))
    store.update_job(third.job_id, status="cancelling")

    def last_used() -> list[tuple[str, float]]:
        conn = sqlite3.connect(str(store.root / "narration.sqlite"))
        try:
            return conn.execute("SELECT job_id, last_used_at FROM jobs ORDER BY job_id").fetchall()
        finally:
            conn.close()

    before = last_used()
    clock.advance(3600)
    listed = list(store.iter_jobs())
    assert [j.job_id for j in listed] == [third.job_id, second.job_id, first.job_id]
    assert [j.status for j in listed] == ["cancelling", "queued", "completed"]
    assert [j.job_id for j in store.iter_jobs(kinds=("generate", "analyse"))] == [third.job_id, first.job_id]
    assert list(store.iter_jobs(kinds=())) == []
    assert [j.job_id for j in store.iter_jobs(inserted_since=clock.now - 7000)] == [third.job_id, second.job_id]
    jobs = store.iter_jobs()
    assert next(jobs).job_id == third.job_id
    jobs.close()  # stopping early closes the cursor
    assert last_used() == before  # a listing is not a use: it never keeps a job from gc


def test_a_malformed_job_id_is_refused(store: NarrationStore) -> None:
    with pytest.raises(InvalidIdError):
        store.create_job(job_record("job_../../x"))
