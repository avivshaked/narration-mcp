"""The request loop and the handler base (design Appendix A), in-process with in-memory streams."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from narration_worker.errors import OpError
from narration_worker.handler import (
    WorkerContext,
    WorkerHandler,
    require_bool,
    require_int,
    require_number,
    require_one_of,
    require_str,
    require_str_list,
)
from narration_worker.loop import serve
from narration_worker.protocol import PROTOCOL_VERSION, QA_OPS


class OutOfMemoryError(RuntimeError):
    """Stands in for ``torch.OutOfMemoryError`` (the loop recognises it by name and module)."""


OutOfMemoryError.__module__ = "torch.cuda"


class _QaDouble(WorkerHandler):
    """A minimal handler of the qa role, for driving the loop."""

    role = "qa"
    uses_torch = False

    def __init__(self, context: WorkerContext) -> None:
        super().__init__(context)
        self.shutdowns = 0

    def op_load(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return {"load_s": 0.0, "vram_mb": None}

    def op_unload(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def op_transcribe(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raise OpError("NOT_LOADED", "load first")

    def op_embed(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raise ZeroDivisionError("boom")

    def op_f0(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def op_align(self, request: Mapping[str, Any]) -> dict[str, Any]:
        print("a stray print")  # the loop's stream is separate; this goes to pytest's capture
        return {"id": 999, "ok": False, "value": float("nan")}

    def op_profile(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return {"id": 999, "ok": False, "measurements": {}, "echo": request.get("x")}

    def shutdown(self) -> None:
        self.shutdowns += 1


def _run(tmp_path: Path, *lines: bytes) -> tuple[int, list[dict[str, Any]], _QaDouble]:
    handler = _QaDouble(WorkerContext(role="qa", store_root=tmp_path, cpu_threads=2))
    reader = io.BytesIO(b"".join(line + b"\n" for line in lines))
    writer = io.BytesIO()
    code = serve(handler, reader, writer)
    replies = [json.loads(line) for line in writer.getvalue().splitlines()]
    return code, replies, handler


def test_hello_names_role_protocol_ops_and_fingerprint_appA(tmp_path: Path) -> None:
    _, [reply], _ = _run(tmp_path, b'{"id":5,"op":"hello"}')
    assert reply["id"] == 5 and reply["ok"] is True
    assert reply["role"] == "qa"
    assert reply["protocol"] == PROTOCOL_VERSION
    assert reply["capabilities"]["ops"] == list(QA_OPS)
    assert reply["fingerprint"]["cpu_threads"] == 2
    assert "narration-worker" in reply["fingerprint"]["packages"]


def test_every_reply_echoes_its_request_id_in_order_appA(tmp_path: Path) -> None:
    _, replies, _ = _run(tmp_path, b'{"id":3,"op":"load","device":"cpu"}', b'{"id":-1,"op":"unload"}')
    assert [(r["id"], r["ok"]) for r in replies] == [(3, True), (-1, True)]


@pytest.mark.parametrize(
    ("line", "expected_id"),
    [
        (b"not json", None),
        (b"[1]", None),
        (b'{"op":"hello"}', None),
        (b'{"id":"1","op":"hello"}', None),
        (b'{"id":true,"op":"hello"}', None),
        (b'{"id":4}', 4),
        (b'{"id":4,"op":7}', 4),
        (b'{"id":4,"op":"synthesize"}', 4),
        (b'{"id":4,"op":"no_such_op"}', 4),
    ],
)
def test_a_malformed_request_or_unknown_op_is_invalid_request_appA(
    tmp_path: Path, line: bytes, expected_id: int | None
) -> None:
    _, replies, _ = _run(tmp_path, line, b'{"id":9,"op":"unload"}')
    assert replies[0]["id"] == expected_id
    assert replies[0]["ok"] is False
    assert replies[0]["error"]["code"] == "INVALID_REQUEST"
    assert replies[1] == {"id": 9, "ok": True}


def test_blank_lines_are_ignored_appA(tmp_path: Path) -> None:
    _, replies, _ = _run(tmp_path, b"", b"   ", b'{"id":1,"op":"unload"}')
    assert replies == [{"id": 1, "ok": True}]


def test_an_op_error_replies_its_code_appA(tmp_path: Path) -> None:
    _, [reply], _ = _run(tmp_path, b'{"id":1,"op":"transcribe"}')
    assert reply == {"id": 1, "ok": False, "error": {"code": "NOT_LOADED", "message": "load first"}}


def test_an_unexpected_exception_is_internal_with_its_type_and_no_traceback_appA(tmp_path: Path) -> None:
    _, [reply], _ = _run(tmp_path, b'{"id":1,"op":"embed"}')
    assert reply["ok"] is False
    assert reply["error"]["code"] == "INTERNAL"
    assert reply["error"]["details"]["type"] == "builtins.ZeroDivisionError"
    assert "Traceback" not in json.dumps(reply)


def test_cuda_out_of_memory_is_gpu_oom_s4(tmp_path: Path) -> None:
    _, [reply], _ = _run(tmp_path, b'{"id":1,"op":"f0"}')
    assert reply["error"]["code"] == "GPU_OOM"


def test_a_reply_json_cannot_hold_becomes_internal_appA(tmp_path: Path) -> None:
    _, [reply], _ = _run(tmp_path, b'{"id":1,"op":"align"}')
    assert reply["id"] == 1
    assert reply["error"]["code"] == "INTERNAL"


def test_a_handler_cannot_override_id_or_ok_appA(tmp_path: Path) -> None:
    _, [reply], _ = _run(tmp_path, '{"id":1,"op":"profile","x":"é"}'.encode())
    assert reply == {"id": 1, "ok": True, "measurements": {}, "echo": "é"}


def test_shutdown_replies_then_stops_with_exit_code_zero_appA(tmp_path: Path) -> None:
    code, replies, handler = _run(tmp_path, b'{"id":1,"op":"shutdown"}', b'{"id":2,"op":"hello"}')
    assert code == 0
    assert replies == [{"id": 1, "ok": True}]
    assert handler.shutdowns == 1


def test_end_of_input_shuts_down_with_exit_code_zero_appA(tmp_path: Path) -> None:
    code, replies, handler = _run(tmp_path)
    assert (code, replies, handler.shutdowns) == (0, [], 1)


def test_a_handler_serves_only_its_own_role() -> None:
    with pytest.raises(ValueError):
        _QaDouble(WorkerContext(role="fake", store_root=Path.cwd(), cpu_threads=1))
    assert _QaDouble.missing_ops() == []


# ---------------------------------------------------------------------- request members and paths


def test_request_members_are_checked_with_the_field_named_appA() -> None:
    request = {"s": "x", "b": True, "i": 3, "n": 1.5, "l": ["a"], "c": "cpu", "bad": True}
    assert require_str(request, "s") == "x"
    assert require_bool(request, "b") is True
    assert require_int(request, "i", minimum=0) == 3
    assert require_number(request, "n") == 1.5
    assert require_str_list(request, "l") == ["a"]
    assert require_one_of(request, "c", ("cpu", "cuda")) == "cpu"
    for call in (
        lambda: require_int(request, "bad"),
        lambda: require_str(request, "missing"),
        lambda: require_int(request, "i", maximum=2),
        lambda: require_one_of(request, "s", ("cpu",)),
    ):
        with pytest.raises(OpError) as caught:
            call()
        assert caught.value.code == "INVALID_REQUEST"
        assert caught.value.details and "field" in caught.value.details


def test_paths_must_be_absolute_and_inside_the_store_s17_2(tmp_path: Path) -> None:
    store = tmp_path / "store"
    (store / "scratch").mkdir(parents=True)
    handler = _QaDouble(WorkerContext(role="qa", store_root=store, cpu_threads=1))
    inside = store / "scratch" / "a.wav"
    inside.write_bytes(b"RIFF")
    assert handler.input_file({"wav": str(inside)}, "wav") == inside.resolve()
    out = handler.output_file({"out": str(store / "scratch" / "new" / "b.wav")}, "out")
    assert out.parent.is_dir()
    for value, code in (
        ("scratch/a.wav", "INVALID_REQUEST"),
        (str(tmp_path / "outside.wav"), "INVALID_REQUEST"),
        (str(store / "scratch" / ".." / ".." / "outside.wav"), "INVALID_REQUEST"),
        (str(store / "scratch" / "missing.wav"), "UNSUPPORTED_AUDIO"),
    ):
        with pytest.raises(OpError) as caught:
            handler.input_file({"wav": value}, "wav")
        assert caught.value.code == code, value
