"""The store across processes: one producer per key, and no partial file after a crash (design sections 4
item 6 and 15; plan.md WP12 acceptance).

These start real child processes (``tests.store.children``). Every wait has a timeout, and every child is
killed and reaped when its test ends, whatever happened.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from narration.store import NarrationStore
from narration.store import files as store_files

from .factories import render_record
from .standin import StandInPlatform

CHECKOUT = Path(__file__).resolve().parents[2]
TIMEOUT_S = 60.0


@pytest.fixture
def children() -> Iterator[list[subprocess.Popen[bytes]]]:
    procs: list[subprocess.Popen[bytes]] = []
    try:
        yield procs
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        for proc in procs:
            with contextlib.suppress(subprocess.TimeoutExpired):  # a child that ignores a kill
                proc.communicate(timeout=TIMEOUT_S)


def spawn(procs: list[subprocess.Popen[bytes]], *args: str) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.store.children", *args],
        cwd=CHECKOUT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    procs.append(proc)
    return proc


def wait_for_files(paths: list[Path], procs: list[subprocess.Popen[bytes]]) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not all(p.exists() for p in paths):
        for proc in procs:
            if proc.poll() is not None and proc.returncode != 0:
                _, err = proc.communicate()
                pytest.fail(f"a child failed early:\n{err.decode('utf-8', 'replace')}")
        if time.monotonic() > deadline:
            pytest.fail(f"children did not get ready within {TIMEOUT_S} s")
        time.sleep(0.01)


def test_two_concurrent_claimers_in_separate_processes_produce_a_key_once_s4(
    tmp_path: Path, children: list[subprocess.Popen[bytes]]
) -> None:
    root = tmp_path / "store"
    NarrationStore(root, StandInPlatform()).close()  # create the database before the race
    holders = [f"proc-{i}" for i in range(4)]
    start = tmp_path / "start"
    for holder in holders:
        spawn(
            children,
            "claim",
            str(root),
            holder,
            str(tmp_path / f"{holder}.ready"),
            str(start),
            str(tmp_path / f"{holder}.json"),
        )
    wait_for_files([tmp_path / f"{h}.ready" for h in holders], children)
    start.write_text("go", encoding="utf-8")
    for proc in children:
        _, err = proc.communicate(timeout=TIMEOUT_S)
        assert proc.returncode == 0, err.decode("utf-8", "replace")
    outcomes = [json.loads((tmp_path / f"{h}.json").read_text(encoding="utf-8")) for h in holders]
    producers = [o for o in outcomes if o["result"] == "claimed"]
    assert len(producers) == 1, outcomes
    assert all(o["result"] in ("in_flight", "exists") for o in outcomes if o is not producers[0])
    assert any(o["result"] == "in_flight" for o in outcomes)  # the others waited for it rather than redo it
    assert all(o.get("waited", True) for o in outcomes)
    assert {o["seen_sha256"] for o in outcomes} == {producers[0]["sha256"]}
    record = render_record()
    with NarrationStore(root, StandInPlatform()) as store:
        published = store.get_render(record.render_key)
        assert published is not None and published.raw.sha256 == producers[0]["sha256"]
        assert [p.name for p in store.render_dir(record.render_id).parent.iterdir()] == [record.render_id]
        assert store.verify()["mismatched"] == []


def test_a_crash_mid_write_leaves_no_partial_file_s15(tmp_path: Path, children: list[subprocess.Popen[bytes]]) -> None:
    target = tmp_path / "delivery.wav"
    marker = tmp_path / "half-written"
    proc = spawn(children, "slow-write", str(target), str(marker))
    wait_for_files([marker], children)
    proc.kill()
    proc.communicate(timeout=TIMEOUT_S)
    assert not target.exists()  # the published name never held a partial file
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert len(leftovers) == 1 and leftovers[0].name.endswith("-delivery.wav")


def test_a_crash_mid_publish_leaves_no_partial_render_s15(
    tmp_path: Path, children: list[subprocess.Popen[bytes]]
) -> None:
    root = tmp_path / "store"
    NarrationStore(root, StandInPlatform()).close()
    marker = tmp_path / "staged"
    proc = spawn(children, "hang-publish", str(root), str(marker))
    wait_for_files([marker], children)
    proc.kill()
    proc.communicate(timeout=TIMEOUT_S)
    record = render_record()
    with NarrationStore(root, StandInPlatform()) as store:
        assert not store.render_dir(record.render_id).exists()
        assert store.get_render(record.render_key) is None
        assert store.claim(record.render_key, "after the crash", ttl_s=30)[0] == "claimed"
        staged = [p for p in store.render_dir(record.render_id).parent.iterdir() if p.name.startswith(".staging-")]
        assert len(staged) == 1
        assert (staged[0] / "raw.wav").is_file() and store_files.is_readonly(staged[0] / "raw.wav")
        # gc treats it as a crash leftover once it is older than the grace period.
        report = store.gc(dry_run=True, grace_s=0.0, now=time.time() + 1)
        assert report["leftovers"] == [store.layout.rel(staged[0])]
        store.gc(dry_run=False, grace_s=0.0, now=time.time() + 1)
        assert not staged[0].exists()
