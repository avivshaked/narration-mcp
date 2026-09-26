"""Claims and leases in one process (design section 4 item 6); across processes see ``test_processes``."""

from __future__ import annotations

import threading
import time

import pytest

from narration.store import InvalidIdError, LeaseLostError, NarrationStore

from .conftest import FakeClock
from .factories import audio_bytes, render_record, scratch_file

KEY = "sha256:" + "ab" * 32


def test_a_free_key_is_claimed_and_then_in_flight_for_others_s4(store: NarrationStore) -> None:
    result, lease = store.claim(KEY, "job-a", ttl_s=30)
    assert result == "claimed" and lease is not None and lease.key == KEY
    assert store.claim(KEY, "job-b", ttl_s=30) == ("in_flight", None)
    lease.release()
    lease.release()  # idempotent
    assert store.claim(KEY, "job-b", ttl_s=30)[0] == "claimed"


def test_a_published_key_exists_at_claim_time_s4(store: NarrationStore) -> None:
    record = render_record()
    store.put_render(record, scratch_file(store, "raw.wav", audio_bytes("x")))
    assert store.claim(record.render_key, "job-a", ttl_s=30) == ("exists", None)


def test_an_expired_lease_can_be_claimed_again_s4(store: NarrationStore, clock: FakeClock) -> None:
    _, stale = store.claim(KEY, "crashed-daemon", ttl_s=10)
    assert stale is not None
    clock.advance(11)
    result, fresh = store.claim(KEY, "new-daemon", ttl_s=10)
    assert result == "claimed" and fresh is not None
    with pytest.raises(LeaseLostError):
        stale.renew(10)
    stale.release()  # the stale holder cannot cut the new holder's lease short
    assert store.claim(KEY, "third", ttl_s=10)[0] == "in_flight"


def test_renewing_keeps_a_lease_alive(store: NarrationStore, clock: FakeClock) -> None:
    _, lease = store.claim(KEY, "job-a", ttl_s=10)
    assert lease is not None
    clock.advance(8)
    lease.renew(10)
    clock.advance(8)
    assert store.claim(KEY, "job-b", ttl_s=10)[0] == "in_flight"


def test_the_same_holder_claiming_again_renews_its_claim(store: NarrationStore) -> None:
    store.claim(KEY, "daemon", ttl_s=10)
    assert store.claim(KEY, "daemon", ttl_s=10)[0] == "claimed"


@pytest.mark.parametrize("key", ["ab" * 32, "sha256:XYZ", "", "sha256:" + "ab" * 31])
def test_claims_need_a_key(store: NarrationStore, key: str) -> None:
    with pytest.raises(InvalidIdError):
        store.claim(key, "job", ttl_s=1)


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan"), True])
def test_claims_need_a_positive_ttl(store: NarrationStore, ttl: float) -> None:
    with pytest.raises(ValueError):
        store.claim(KEY, "job", ttl_s=ttl)


def test_wait_for_returns_when_the_holder_publishes_s4(store: NarrationStore) -> None:
    record = render_record()
    _, lease = store.claim(record.render_key, "producer", ttl_s=30)
    assert lease is not None

    def produce() -> None:
        time.sleep(0.2)
        store.put_render(record, scratch_file(store, "raw.wav", audio_bytes("y")))
        lease.release()

    thread = threading.Thread(target=produce)
    thread.start()
    assert store.claim(record.render_key, "waiter", ttl_s=30)[0] == "in_flight"
    assert store.wait_for(record.render_key, timeout_s=30) is True
    thread.join(timeout=30)
    assert store.get_render(record.render_key) is not None


def test_wait_for_gives_up_when_the_holder_releases_without_a_result_s4(store: NarrationStore) -> None:
    _, lease = store.claim(KEY, "producer", ttl_s=30)
    assert lease is not None
    timer = threading.Timer(0.1, lease.release)
    timer.start()
    try:
        assert store.wait_for(KEY, timeout_s=30) is False
    finally:
        timer.join(timeout=30)
    assert store.claim(KEY, "next", ttl_s=30)[0] == "claimed"


def test_wait_for_times_out(store: NarrationStore) -> None:
    store.claim(KEY, "producer", ttl_s=300)
    started = time.monotonic()
    assert store.wait_for(KEY, timeout_s=0.1) is False
    assert time.monotonic() - started < 5


def test_many_threads_claiming_one_key_get_one_lease(store: NarrationStore) -> None:
    results: list[str] = []
    lock = threading.Lock()

    def claim(i: int) -> None:
        result, _ = store.claim(KEY, f"thread-{i}", ttl_s=30)
        with lock:
            results.append(result)

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert sorted(results) == ["claimed"] + ["in_flight"] * 7
