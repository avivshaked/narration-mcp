"""The seams between work packages: documented, and consistent with the worker protocol (plan.md WP01)."""

from __future__ import annotations

import inspect
import re
from typing import Protocol, get_args

from narration.contracts import interfaces, names, worker


def _protocols() -> list[type]:
    return [
        obj
        for _, obj in inspect.getmembers(interfaces, inspect.isclass)
        if issubclass(obj, Protocol) and obj is not Protocol and obj.__module__ == interfaces.__name__  # type: ignore[arg-type]
    ]


def test_every_interface_has_a_docstring_naming_its_design_section_or_wp() -> None:
    protocols = _protocols()
    assert {p.__name__ for p in protocols} >= {
        "TextPlanner",
        "KeyBuilder",
        "Store",
        "DeliveryProcessor",
        "QaScorer",
        "AlignerCore",
        "WorkerClient",
        "Backend",
        "Platform",
    }
    for proto in protocols:
        doc = inspect.getdoc(proto) or ""
        assert re.search(r"section|WP\d\d|Appendix", doc), proto.__name__


def test_the_backend_has_one_method_per_tool_s7_1() -> None:
    for tool in names.TOOL_NAMES:
        assert inspect.iscoroutinefunction(getattr(interfaces.Backend, tool)), tool


def test_the_fake_role_implements_every_op_of_both_real_workers_app_a() -> None:
    assert set(worker.OPS_BY_ROLE["fake"]) == set(worker.QWEN3_OPS) | set(worker.QA_OPS)
    for role_ops in worker.OPS_BY_ROLE.values():
        assert set(worker.COMMON_OPS) <= set(role_ops)


def test_worker_protocol_optional_keys_are_optional_at_run_time_appA() -> None:
    import narration_worker.protocol as protocol

    assert protocol.Reply.__required_keys__ == {"id", "ok"}
    assert protocol.Reply.__optional_keys__ == {"error"}
    assert "controls" in protocol.Capabilities.__optional_keys__


def test_worker_protocol_is_one_definition_appA() -> None:
    import narration_worker.protocol as protocol

    assert worker.OPS_BY_ROLE is protocol.OPS_BY_ROLE
    assert names.WorkerRole is protocol.WorkerRole
    assert set(get_args(worker.Op)) == set(worker.FAKE_OPS)
    assert get_args(worker.WorkerErrorCode) == worker.WORKER_ERROR_CODES
    assert set(worker.OPS_BY_ROLE) == set(get_args(names.WorkerRole))


def test_an_align_transcript_names_its_cue_count_and_term_words_s11_2() -> None:
    import dataclasses

    import pytest

    base = {"tokens": ("A",), "token_words": ((0, 0),), "words": ((0, 0, "a"),)}
    with pytest.raises(TypeError):
        interfaces.AlignTranscript(**base)  # pyright: ignore[reportCallIssue]
    transcript = interfaces.AlignTranscript(**base, cue_count=2)
    assert transcript.cue_count == 2 and transcript.term_words == () and transcript.dropped == ()
    assert [f.name for f in dataclasses.fields(interfaces.AlignTranscript)] == [
        "tokens",
        "token_words",
        "words",
        "cue_count",
        "term_words",
        "dropped",
    ]


def test_the_aligner_core_takes_the_workers_error_and_gives_the_guards_facts_s11_2() -> None:
    # Contracts 1.6.2: resolve takes the ALIGNMENT_ERROR reply's details as a keyword, and guard_details
    # gives the facts behind a failed guard, so a caller whose own guard fails can pass them.
    resolve = inspect.signature(interfaces.AlignerCore.resolve)
    error = resolve.parameters["error"]
    assert (error.kind, error.default) == (inspect.Parameter.KEYWORD_ONLY, None)
    assert list(inspect.signature(interfaces.AlignerCore.guard_details).parameters) == [
        "self",
        "transcript",
        "num_frames",
    ]
    assert "*" in (inspect.getdoc(interfaces.AlignTranscript) or "")
