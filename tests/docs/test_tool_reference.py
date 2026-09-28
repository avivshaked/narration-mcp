"""``docs/tools.md`` is generated from the published tool definitions (``tools/gen_tool_reference.py``,
plan.md WP43). This pins it current: a schema or description change that is not followed by regenerating
the page fails here, with a hint to rerun the generator, rather than shipping a stale reference.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "gen_tool_reference.py"
OUTPUT = ROOT / "docs" / "tools.md"


def load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_tool_reference", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_tool_reference"] = module
    spec.loader.exec_module(module)
    return module


gen = load_generator()


def test_docs_tools_md_is_current() -> None:
    """The checked-in page is exactly what the generator builds now, byte for byte."""
    assert OUTPUT.is_file(), f"{OUTPUT} does not exist; run `uv run python tools/gen_tool_reference.py`."
    current = OUTPUT.read_bytes()
    generated = gen.render().encode("utf-8")
    assert current == generated, (
        f"{OUTPUT} is stale (it no longer matches the published tool definitions). Run "
        "`uv run python tools/gen_tool_reference.py` and commit the result."
    )


def test_docs_tools_md_has_lf_line_endings() -> None:
    """The page is written with ``newline=\"\\n\"``: no ``\\r`` anywhere, even on Windows."""
    assert b"\r" not in OUTPUT.read_bytes()


def test_docs_tools_md_ends_with_one_newline() -> None:
    text = OUTPUT.read_text(encoding="utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_render_is_deterministic() -> None:
    """Two runs in the same process build byte-identical text (no set or hash-order dependence)."""
    assert gen.render() == gen.render()


def test_every_tool_name_appears_as_a_heading() -> None:
    from narration.contracts.names import TOOL_NAMES

    text = gen.render()
    for name in TOOL_NAMES:
        assert f"## `{name}`" in text


def test_every_error_and_flag_code_appears() -> None:
    from narration.contracts import codes

    text = gen.render()
    for code in codes.ERRORS:
        assert f"`{code}`" in text
    for code in codes.FLAGS:
        assert f"`{code}`" in text
