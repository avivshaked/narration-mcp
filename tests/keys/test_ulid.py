"""ULIDs for job and design ids (design section 5)."""

from __future__ import annotations

import threading

import pytest

from narration.keys import ulid


def test_the_specifications_example_decodes_to_its_time_s5() -> None:
    # The ULID specification's README: ulid(1469918176385) -> 01ARYZ6S41TSV4RRFFQ69G5FAV.
    assert ulid.timestamp_ms("01ARYZ6S41TSV4RRFFQ69G5FAV") == 1469918176385
    value = ulid.decode("01ARYZ6S41TSV4RRFFQ69G5FAV")
    assert ulid.encode(value) == "01ARYZ6S41TSV4RRFFQ69G5FAV"


def test_encode_covers_the_full_128_bits() -> None:
    assert ulid.encode(0) == "0" * 26
    assert ulid.encode((1 << 128) - 1) == "7" + "Z" * 25
    with pytest.raises(ValueError):
        ulid.encode(1 << 128)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "01ARYZ6S41TSV4RRFFQ69G5FA",
        "81ARYZ6S41TSV4RRFFQ69G5FAV",
        "01arYZ6S41TSV4RRFFQ69G5FAV",
        "01ARYZ6S41TSV4RRFFQ69G5FAU",
    ],
)
def test_only_canonical_ulids_are_accepted(bad: str) -> None:
    assert not ulid.is_ulid(bad)
    with pytest.raises(ValueError):
        ulid.decode(bad)


def test_the_alphabet_is_crockfords() -> None:
    assert len(ulid.CROCKFORD) == 32 and not set("ILOU") & set(ulid.CROCKFORD)


def test_a_new_ulid_carries_the_clock_and_80_random_bits() -> None:
    gen = ulid.UlidGenerator(clock_ms=lambda: 1_700_000_000_000, random_bytes=lambda n: b"\x01" * n)
    value = ulid.decode(gen.new())
    assert value >> 80 == 1_700_000_000_000
    assert value & ((1 << 80) - 1) == int.from_bytes(b"\x01" * 10, "big")


def test_ids_increase_within_one_millisecond() -> None:
    gen = ulid.UlidGenerator(clock_ms=lambda: 1_700_000_000_000, random_bytes=lambda n: b"\x00" * n)
    ids = [gen.new() for _ in range(5)]
    assert ids == sorted(ids) and len(set(ids)) == 5
    assert [ulid.decode(i) & 0xFF for i in ids] == [0, 1, 2, 3, 4]


def test_ids_keep_increasing_when_the_clock_steps_back() -> None:
    times = iter([2_000, 2_000, 1_000, 1_500, 3_000])
    gen = ulid.UlidGenerator(clock_ms=lambda: next(times))
    ids = [gen.new() for _ in range(5)]
    assert ids == sorted(ids) and len(set(ids)) == 5
    assert ulid.timestamp_ms(ids[-1]) == 3_000


def test_a_full_random_part_borrows_the_next_millisecond() -> None:
    gen = ulid.UlidGenerator(clock_ms=lambda: 5_000, random_bytes=lambda n: b"\xff" * n)
    first, second = gen.new(), gen.new()
    assert second > first
    assert ulid.timestamp_ms(second) == 5_001


def test_ids_are_unique_and_ordered_across_threads() -> None:
    gen = ulid.UlidGenerator()
    results: list[list[str]] = [[] for _ in range(4)]

    def work(slot: list[str]) -> None:
        slot.extend(gen.new() for _ in range(500))

    threads = [threading.Thread(target=work, args=(slot,)) for slot in results]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    every = [i for slot in results for i in slot]
    assert len(set(every)) == 2000
    assert all(slot == sorted(slot) for slot in results)
