"""The fake worker's record of what it said, keyed by utterance id (see ``audio.py``).

Every ``synthesize`` and ``design`` writes one record to ``<store_root>/scratch/fake/utterances/<id>.json``.
The fake's QA ops read the id back from the audio and look the record up, so they can answer for a file
the fake rendered even after post-processing, and even in another fake process, as long as both use the
same store.

An utterance's id is the first 32 bits of a hash of what it says, so two utterances can share one (by the
birthday bound, about even odds after 77,000 utterances in one store). A record is written once and never
changed. When another utterance already holds the id, the new one steps to the next id (``+1`` modulo
2**32) until it finds its own record or a free id: the same store history always gives the same ids.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

from narration_worker.errors import OpError

RECORD_SCHEMA: Final = "narration.fake-utterance/v1"
ID_SPACE: Final = 1 << 32
MAX_STEPS: Final = 64
"""How many ids a claim tries before it gives up; a store would need that many clashes in a row."""


def fake_dir(store_root: Path) -> Path:
    """The fake worker's own folder in the store's scratch area."""
    return store_root / "scratch" / "fake"


def _canonical(record: dict[str, Any]) -> bytes:
    """A record's bytes, without its id: what makes two records the same utterance."""
    body = {k: v for k, v in record.items() if k != "id"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")


class Registry:
    """Utterance records in the store."""

    def __init__(self, store_root: Path) -> None:
        self.root = fake_dir(store_root) / "utterances"

    def path(self, utterance_id: int) -> Path:
        return self.root / f"{utterance_id:08x}.json"

    def claim(self, preferred: int, record: dict[str, Any]) -> int:
        """Store ``record`` under ``preferred``, or under the next id that is free; returns the id it holds.

        The record's ``"id"`` is set to that id. If the same utterance is already stored (at ``preferred``
        or at a later id on its way), its id is returned and nothing is written. Raises ``OpError``
        (``INTERNAL``) only after ``MAX_STEPS`` clashes in a row.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        wanted = _canonical(record)
        for step in range(MAX_STEPS):
            candidate = (preferred + step) % ID_SPACE
            if self._publish(candidate, {**record, "id": f"{candidate:08x}"}, wanted):
                record["id"] = f"{candidate:08x}"
                return candidate
        raise OpError(
            "INTERNAL",
            f"the fake worker found {MAX_STEPS} other utterances in a row from id {preferred:08x}",
            {"root": str(self.root)},
        )

    def _publish(self, utterance_id: int, record: dict[str, Any], wanted: bytes) -> bool:
        """Write the record under this id unless another utterance holds it; True when the id is the record's."""
        path = self.path(utterance_id)
        if path.exists():
            return self._holds(path, wanted)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_bytes(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            try:
                os.link(tmp, path)  # never replaces: whoever publishes first holds the id
            except FileExistsError:
                return self._holds(path, wanted)
            except OSError:
                os.replace(tmp, path)  # a file system without hard links; two fakes racing here is improbable
        finally:
            tmp.unlink(missing_ok=True)
        return True

    @staticmethod
    def _holds(path: Path, wanted: bytes) -> bool:
        try:
            existing = json.loads(path.read_bytes())
        except (OSError, ValueError):
            return False
        return isinstance(existing, dict) and _canonical(existing) == wanted

    def get(self, utterance_id: int) -> dict[str, Any] | None:
        """The record for an id, or None."""
        try:
            data = self.path(utterance_id).read_bytes()
        except FileNotFoundError:
            return None
        record = json.loads(data)
        return record if isinstance(record, dict) and record.get("schema") == RECORD_SCHEMA else None
