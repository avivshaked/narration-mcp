"""The fake worker's record of what it said, keyed by utterance id (see ``audio.py``).

Every ``synthesize`` and ``design`` writes one record to ``<store_root>/scratch/fake/utterances/<id>.json``.
The fake's QA ops read the id back from the audio and look the record up, so they can answer for a file
the fake rendered even after post-processing, and even in another fake process, as long as both use the
same store. A record is written once; writing a different record under an existing id is refused.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

from narration_worker.errors import OpError

RECORD_SCHEMA: Final = "narration.fake-utterance/v1"


def fake_dir(store_root: Path) -> Path:
    """The fake worker's own folder in the store's scratch area."""
    return store_root / "scratch" / "fake"


class Registry:
    """Utterance records in the store."""

    def __init__(self, store_root: Path) -> None:
        self.root = fake_dir(store_root) / "utterances"

    def path(self, utterance_id: int) -> Path:
        return self.root / f"{utterance_id:08x}.json"

    def put(self, utterance_id: int, record: dict[str, Any]) -> None:
        """Write a record (temp name, then rename); an identical record already there is left alone."""
        self.root.mkdir(parents=True, exist_ok=True)
        data = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path = self.path(utterance_id)
        if path.exists():
            if path.read_bytes() == data:
                return
            raise OpError(
                "INTERNAL",
                f"the fake worker's utterance id {utterance_id:08x} is already taken by another utterance",
                {"path": str(path)},
            )
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def get(self, utterance_id: int) -> dict[str, Any] | None:
        """The record for an id, or None."""
        try:
            data = self.path(utterance_id).read_bytes()
        except FileNotFoundError:
            return None
        record = json.loads(data)
        return record if isinstance(record, dict) and record.get("schema") == RECORD_SCHEMA else None
