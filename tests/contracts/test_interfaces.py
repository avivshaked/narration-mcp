"""The seams between work packages: documented, and consistent with the worker protocol (plan.md WP01)."""

from __future__ import annotations

import inspect
import re
from typing import Protocol

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
