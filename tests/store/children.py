"""Child processes for the store's multi-process tests: ``python -m tests.store.children <command> …``.

Run from the checkout's root. Every wait here has a deadline, so a child never outlives its test by long
even if the parent could not kill it; the parent kills and reaps every child anyway.

Commands:

- ``claim <root> <holder> <ready_file> <start_file> <out_file>``: open the store, say ready, wait for the
  start signal, then claim the render key of ``factories.render_record()``. The claimer that gets it
  produces the render (slowly, so the others see it in flight) and publishes it; the others wait for it.
  The outcome goes to ``out_file`` as JSON.
- ``slow-write <path> <marker>``: write ``path`` atomically, but stop after the first chunk, create
  ``marker``, and sleep until killed.
- ``hang-publish <root> <marker>``: stage a render (audio moved in, sidecar written), then create
  ``marker`` and sleep just before the folder would be renamed into place, until killed.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from narration.store import NarrationStore
from narration.store import files as store_files
from narration.store import store as store_module

from .factories import audio_bytes, render_record, scratch_file
from .standin import StandInPlatform

DEADLINE_S = 60.0


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + DEADLINE_S
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no {path.name} within {DEADLINE_S} s")
        time.sleep(0.002)


def claim(root: str, holder: str, ready_file: str, start_file: str, out_file: str) -> None:
    with NarrationStore(Path(root), StandInPlatform()) as store:
        record = render_record()
        Path(ready_file).write_text("ready", encoding="utf-8")
        _wait_for_file(Path(start_file))
        result, lease = store.claim(record.render_key, holder, ttl_s=30)
        out: dict[str, object] = {"holder": holder, "result": result}
        if result == "claimed":
            assert lease is not None
            time.sleep(1.0)  # producing takes a while, so the other claimers find it in flight
            raw = scratch_file(store, f"{holder}.wav", audio_bytes(f"rendered by {holder}"))
            published = store.put_render(record, raw)
            lease.release()
            out["sha256"] = published.raw.sha256
        elif result == "in_flight":
            out["waited"] = store.wait_for(record.render_key, timeout_s=DEADLINE_S)
        found = store.get_render(record.render_key)
        out["seen_sha256"] = found.raw.sha256 if found else None
    Path(out_file).write_text(json.dumps(out), encoding="utf-8")


def slow_write(path: str, marker: str) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"the first half of a file that must never be published"
        Path(marker).write_text("written", encoding="utf-8")
        time.sleep(DEADLINE_S)
        yield b" and the second half"

    store_files.write_atomic(Path(path), chunks(), readonly=True)


def hang_publish(root: str, marker: str) -> None:
    def hang(*args: object, **kwargs: object) -> bool:
        Path(marker).write_text("staged", encoding="utf-8")
        time.sleep(DEADLINE_S)
        raise TimeoutError("not killed in time")

    store_module.NarrationStore._publish_dir = hang  # type: ignore[method-assign]
    with NarrationStore(Path(root), StandInPlatform()) as store:
        store.put_render(render_record(), scratch_file(store, "raw.wav", audio_bytes("hanging")))


def main(argv: list[str]) -> int:
    command, *args = argv
    if command == "claim":
        claim(*args)
    elif command == "slow-write":
        slow_write(*args)
    elif command == "hang-publish":
        hang_publish(*args)
    else:
        raise SystemExit(f"unknown command {command!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
