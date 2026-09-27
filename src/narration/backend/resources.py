"""The ``narration://`` resources (design section 7.7), read from the store.

| URI | What | ttlMs |
|---|---|---|
| ``narration://status`` | ``get_server_status``'s result | 5 000 |
| ``narration://designs/{design_id}`` | the design's candidates: clips, transcripts, profiles | 60 000 |
| ``narration://measurements/{voice_hash}`` | the voice's measurements, one per engine profile | 60 000 |
| ``narration://jobs/{job_id}`` | ``get_job``'s result, with its segments | 2 000 running; 86 400 000 terminal |
| ``narration://jobs/{job_id}/report`` | the job's report (markdown) | as the job |
| ``narration://takes/{take_id}`` | the take, its render and every analysis of it | 86 400 000 |

The front-end resolves the URI and checks its ids before it calls here, and turns ``NOT_FOUND`` into
JSON-RPC -32602. Audio and pictures are reached through paths and ``file:///`` links only, never inlined.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Final

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import ResourceContent
from narration.contracts.names import RESOURCES, TERMINAL_JOB_STATUSES
from narration.contracts.serial import to_json

from . import views

if TYPE_CHECKING:
    from .service import NarrationBackend

JSON: Final = "application/json"
MARKDOWN: Final = "text/markdown"
_TTL: Final = {r.name: (r.ttl_ms, r.ttl_ms_terminal) for r in RESOURCES}
_JOB: Final = re.compile(r"narration://jobs/(?P<job_id>[^/]+)")
_REPORT: Final = re.compile(r"narration://jobs/(?P<job_id>[^/]+)/report")
_TAKE: Final = re.compile(r"narration://takes/(?P<take_id>[^/]+)")
_DESIGN: Final = re.compile(r"narration://designs/(?P<design_id>[^/]+)")
_MEASUREMENTS: Final = re.compile(r"narration://measurements/(?P<voice_hash>[^/]+)")


def _missing(uri: str) -> NarrationError:
    return NarrationError(codes.NOT_FOUND, f"there is no resource {uri}", details={"uri": uri})


def _json(uri: str, value: Any, ttl_ms: int) -> ResourceContent:
    return ResourceContent(uri=uri, mime_type=JSON, text=json.dumps(value, ensure_ascii=False), ttl_ms=ttl_ms)


def _job_ttl(terminal: bool) -> int:
    running, done = _TTL["job"]
    return done if terminal and done is not None else running


def read_resource(backend: NarrationBackend, uri: str) -> ResourceContent:
    """The resource ``uri`` names (canonical, as the front-end resolved it); ``NOT_FOUND`` otherwise."""
    store = backend.store
    if uri == "narration://status":
        return _json(uri, backend.get_server_status_sync({}), _TTL["status"][0])
    if (m := _REPORT.fullmatch(uri)) is not None:
        job = store.get_job(m["job_id"])
        if job is None:
            raise _missing(uri)
        results = backend.get_results_sync({"job_id": job.job_id})
        text = backend.scorer.report_md(results)
        return ResourceContent(
            uri=uri, mime_type=MARKDOWN, text=text, ttl_ms=_job_ttl(job.status in TERMINAL_JOB_STATUSES)
        )
    if (m := _JOB.fullmatch(uri)) is not None:
        job = store.get_job(m["job_id"])
        if job is None:
            raise _missing(uri)
        body = views.job_json(job, store.queued_jobs(), include_segments=True)
        return _json(uri, body, _job_ttl(job.status in TERMINAL_JOB_STATUSES))
    if (m := _TAKE.fullmatch(uri)) is not None:
        take = store.get_take_by_id(m["take_id"])
        if take is None:
            raise _missing(uri)
        render = store.get_render_by_id(take.render_id)
        analyses = store.analyses_of(take.take_id)
        body = {
            "take": to_json(take),
            "render": to_json(render) if render is not None else None,
            "analyses": [to_json(a) for a in analyses],
        }
        return _json(uri, body, _TTL["take"][0])
    if (m := _DESIGN.fullmatch(uri)) is not None:
        candidates = store.get_design(m["design_id"])
        if not candidates:
            raise _missing(uri)
        body = {"design_id": m["design_id"], "candidates": [to_json(c) for c in candidates]}
        return _json(uri, body, _TTL["design"][0])
    if (m := _MEASUREMENTS.fullmatch(uri)) is not None:
        measurements = store.measurements_of(m["voice_hash"])
        if not measurements:
            raise _missing(uri)
        body = {"voice_hash": m["voice_hash"], "measurements": [to_json(x) for x in measurements]}
        return _json(uri, body, _TTL["measurement"][0])
    raise _missing(uri)


__all__ = ["read_resource"]
