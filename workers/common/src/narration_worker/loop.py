"""The worker's request loop (design Appendix A).

It reads one request per line, runs it, and writes one reply per request, in order:

- a line that is not a JSON object with an integer ``id`` gets ``{"id": null, "ok": false, "error":
  {"code": "INVALID_REQUEST", …}}`` (there is no id to echo), and the loop goes on;
- an ``op`` that is missing, not a string, or not one of the role's ops gets ``INVALID_REQUEST`` with the
  request's ``id``;
- ``OpError`` from the handler becomes ``ok: false`` with its code; any other exception becomes ``INTERNAL``
  (or ``GPU_OOM``, see ``WorkerHandler.classify``) with the exception's type in ``details``, and its
  traceback goes to the log on stderr, never to stdout;
- ``shutdown`` calls the handler's ``shutdown()``, replies, and ends the loop (exit code 0); so does the end
  of input, without a reply.

Blank lines are ignored.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import IO, Any

from .errors import OpError
from .framing import FramingError, LineTooLong, decode_message, encode_message, is_request_id, read_line
from .handler import WorkerHandler

log = logging.getLogger(__name__)


def error_reply(request_id: int | None, error: OpError) -> dict[str, Any]:
    """An ``ok: false`` reply."""
    return {"id": request_id, "ok": False, "error": error.to_error()}


def serve(handler: WorkerHandler, reader: IO[bytes], writer: IO[bytes]) -> int:
    """Serve requests from ``reader`` until ``shutdown`` or the end of input; returns the exit code (0)."""
    out = _Writer(writer)
    log.info("%s worker ready (ops: %s)", handler.role, ", ".join(handler.ops))
    while True:
        try:
            line = read_line(reader)
        except LineTooLong as exc:
            if not out.send(error_reply(None, OpError("INVALID_REQUEST", str(exc)))):
                return 0
            continue
        if line is None:
            log.info("end of input: shutting down")
            _shutdown(handler)
            return 0
        if not line.strip():
            continue
        request, early = _parse(line, handler)
        if early is not None:
            if not out.send(early):
                return 0
            continue
        assert request is not None
        rid: int = request["id"]
        op: str = request["op"]
        if op == "shutdown":
            reply = _run_hook(handler, request, out)
            if reply is None:
                _shutdown(handler)
                reply = {"id": rid, "ok": True}
            out.send(reply)
            log.info("shutdown: exiting")
            return 0
        reply = _run_hook(handler, request, out) or _run(handler, request)
        if not out.send(reply):
            return 0


def _parse(line: bytes, handler: WorkerHandler) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The request, or the error reply to send instead."""
    try:
        request = decode_message(line)
    except FramingError as exc:
        return None, error_reply(None, OpError("INVALID_REQUEST", str(exc)))
    rid = request.get("id")
    if not is_request_id(rid):
        return None, error_reply(None, OpError("INVALID_REQUEST", "a request needs an integer id", {"field": "id"}))
    op = request.get("op")
    if not isinstance(op, str):
        return None, error_reply(rid, OpError("INVALID_REQUEST", "a request needs a string op", {"field": "op"}))
    if op not in handler.ops:
        return None, error_reply(
            rid,
            OpError(
                "INVALID_REQUEST",
                f"the {handler.role} worker has no op {op!r}",
                {"field": "op", "op": op, "ops": list(handler.ops)},
            ),
        )
    return request, None


def _run_hook(handler: WorkerHandler, request: dict[str, Any], out: _Writer) -> dict[str, Any] | None:
    """Run ``before_request``; returns an error reply if it raised, else None."""
    try:
        raw = handler.before_request(request["op"], request)
    except OpError as exc:
        return error_reply(request["id"], exc)
    except Exception as exc:
        return error_reply(request["id"], _unexpected(handler, request["op"], exc))
    if raw is not None:
        out.raw(raw)
    return None


def _run(handler: WorkerHandler, request: dict[str, Any]) -> dict[str, Any]:
    rid: int = request["id"]
    op: str = request["op"]
    try:
        result = handler.handle(op, request)
    except OpError as exc:
        log.info("%s (id %s) failed: %s", op, rid, exc)
        return error_reply(rid, exc)
    except Exception as exc:
        return error_reply(rid, _unexpected(handler, op, exc))
    if not isinstance(result, Mapping):
        return error_reply(rid, OpError("INTERNAL", f"{op} returned {type(result).__name__}, not a mapping"))
    return {**{k: v for k, v in result.items() if k not in ("id", "ok", "error")}, "id": rid, "ok": True}


def _unexpected(handler: WorkerHandler, op: str, exc: Exception) -> OpError:
    classified = handler.classify(exc)
    if classified is not None:
        log.warning("%s: %s", op, classified)
        return classified
    log.exception("%s raised %s", op, type(exc).__name__)
    kind = type(exc)
    return OpError(
        "INTERNAL", f"{op} raised {kind.__name__}: {exc}"[:2000], {"type": f"{kind.__module__}.{kind.__name__}"}
    )


def _shutdown(handler: WorkerHandler) -> None:
    try:
        handler.shutdown()
    except Exception:
        log.exception("shutdown raised")


class _Writer:
    """Writes replies to the protocol stream; a reply JSON cannot hold becomes ``INTERNAL``."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream

    def send(self, reply: dict[str, Any]) -> bool:
        """Write one reply; False when the reader has gone (the daemon closed the pipe)."""
        try:
            data = encode_message(reply)
        except (ValueError, TypeError) as exc:
            log.error("reply to id %s is not a protocol message: %s", reply.get("id"), exc)
            data = encode_message(
                error_reply(reply.get("id"), OpError("INTERNAL", f"the reply could not be encoded: {exc}"[:2000]))
            )
        return self.raw(data)

    def raw(self, data: bytes) -> bool:
        try:
            self._stream.write(data)
            self._stream.flush()
        except (BrokenPipeError, OSError) as exc:
            log.warning("the protocol stream is closed (%s): exiting", exc)
            return False
        return True
