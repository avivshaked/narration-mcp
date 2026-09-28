"""Generate ``docs/tools.md``, the MCP tool reference, from the published tool definitions.

The page is built from the same sources the server publishes from (``narration.mcp.server.build_tools``):
each tool's title and description (``narration.mcp.descriptions``), and its input and output JSON Schema
(``narration.contracts.schemas.TOOLS_BY_NAME``). The retention periods the descriptions state come from
``narration.config.RetentionConfig``'s defaults, the ones ``narration.example.toml`` ships. After every
tool, the error and flag codes a caller may see (``narration.contracts.codes``) are tabulated once, in
their design section 14 order.

Every schema is a plain JSON Schema tree with no ``$ref`` (design section 5), built from nested Python
dicts with no cycles, so it can be flattened into rows by simple recursion: a field's row, then a row for
each property of a nested object, and a row for the item shape of an array of objects (its path ends in
``[]``). A scalar array (of strings, say) gets no further row; its item type is folded into its own type
column.

The output is deterministic: the tools are in the server's own order (``narration.contracts.names.
TOOL_NAMES``), fields are in the schema's own (insertion) order, and the codes are in ``codes.ERRORS`` /
``codes.FLAGS``'s own order — design section 14's. Nothing here depends on set or dict hash order.

Run it with the server's own interpreter (it imports ``narration``), from the repository root::

    uv run python tools/gen_tool_reference.py [--check]

With no arguments it writes ``docs/tools.md`` (a temporary file, then ``os.replace``), LF line endings,
UTF-8, ending in one newline. ``--check`` writes nothing and exits 1 if the checked-in file is stale
(``tests/docs/test_tool_reference.py`` runs this at import time, in the default test suite).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
OUTPUT: Final = ROOT / "docs" / "tools.md"

Schema = Mapping[str, Any]


def _import_narration() -> tuple[Any, ...]:
    """Import what the generator needs from the server's own package, with a clear message if it is not
    on the path (the tool must be run with the server's interpreter: ``uv run python ...``)."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from narration.config import RetentionConfig
        from narration.contracts import codes
        from narration.contracts.names import TOOL_NAMES
        from narration.contracts.schemas import TOOLS_BY_NAME
        from narration.mcp.descriptions import TOOL_TEXTS, tool_description
    except ImportError as exc:  # pragma: no cover - environment problem, not a logic path to test
        raise SystemExit(
            f"{Path(__file__).name}: cannot import narration ({exc}). Run this with the server's own "
            "interpreter, e.g. `uv run python tools/gen_tool_reference.py` from the repository root."
        ) from exc
    return TOOL_NAMES, TOOLS_BY_NAME, TOOL_TEXTS, tool_description, RetentionConfig, codes


# ======================================================================== schema flattening


def _escape(text: str) -> str:
    """A description or meaning, safe as one Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def _scalar_type(name: str, schema: Schema) -> str:
    if name == "array":
        items = schema.get("items")
        inner = type_of(items) if isinstance(items, Mapping) else "any"
        return f"array of {inner}"
    return name


def type_of(schema: Schema) -> str:
    """A short, human type for one schema node: its ``enum``/``const``, its ``type`` (scalar, or an array's
    item type), or the union of its ``anyOf``/``oneOf`` branches."""
    if "const" in schema:
        return f"const {schema['const']!r}"
    if "enum" in schema:
        values = ["null" if v is None else repr(v) for v in schema["enum"]]
        return "enum: " + ", ".join(values)
    kind = schema.get("type")
    if isinstance(kind, str):
        return _scalar_type(kind, schema)
    if isinstance(kind, list):
        return " or ".join(_scalar_type(k, schema) for k in kind)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return " or ".join(type_of(s) for s in schema[key])
    if "prefixItems" in schema:
        return "tuple of (" + ", ".join(type_of(s) for s in schema["prefixItems"]) + ")"
    return "any"


@dataclass(frozen=True, slots=True)
class Row:
    """One field of a flattened schema table."""

    path: str
    type: str
    required: str
    description: str


def flatten(schema: Schema, path: str, required: bool | None, rows: list[Row], *, seen: int = 0) -> None:
    """Append ``path``'s own row, then recurse into an object's properties (dotted paths) and an array of
    objects' item shape (a path ending in ``[]``). ``seen`` guards against a schema that is not the finite
    tree this codebase always builds (never true here; it turns a latent bug into a clear error instead of
    hanging the generator)."""
    if seen > 40:
        raise RecursionError(f"schema nesting over 40 levels at {path!r}: not a finite tree?")
    required_cell = "yes" if required else ("" if required is False else "n/a")
    rows.append(Row(path, type_of(schema), required_cell, _escape(schema.get("description", ""))))
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        required_here = set(schema.get("required", []))
        for name, sub in properties.items():
            flatten(sub, f"{path}.{name}" if path else name, name in required_here, rows, seen=seen + 1)
    items = schema.get("items")
    if isinstance(items, Mapping) and isinstance(items.get("properties"), Mapping):
        flatten(items, f"{path}[]", None, rows, seen=seen + 1)


def field_rows(schema: Schema) -> list[Row]:
    """One row per top-level property of an input or output schema (never a row for the schema's own root),
    each followed by its nested rows (``flatten``)."""
    rows: list[Row] = []
    required_here = set(schema.get("required", []))
    for name, sub in schema.get("properties", {}).items():
        flatten(sub, name, name in required_here, rows)
    return rows


# ======================================================================== rendering


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(cell for cell in row) + " |" for row in rows]
    return lines


def _field_table(rows: Sequence[Row]) -> list[str]:
    if not rows:
        return ["No fields.", ""]
    table = _table(
        ("Field", "Type", "Required", "Description"),
        [(f"`{r.path}`", _escape(r.type), r.required, r.description) for r in rows],
    )
    return [*table, ""]


def render() -> str:
    """The whole page, as text ending in one newline."""
    TOOL_NAMES, TOOLS_BY_NAME, TOOL_TEXTS, tool_description, RetentionConfig, codes = _import_narration()
    retention = RetentionConfig()
    lines: list[str] = [
        "# MCP tool reference",
        "",
        "Generated by `tools/gen_tool_reference.py` from the published tool definitions "
        "(`narration.contracts.schemas`, `narration.mcp.descriptions`); do not edit by hand. Regenerate it "
        "after a schema or description change (`uv run python tools/gen_tool_reference.py`); "
        "`tests/docs/test_tool_reference.py` fails when this file is stale.",
        "",
        "Every tool's description below states the rules every caller must know: the service keeps no "
        "caller state, a segment's length is the caller's decision, and a pronunciation respelling is a "
        "hint, never a guarantee. Retention periods are this build's defaults "
        f"(`[retention] retention_days = {retention.retention_days}`, "
        f"`measurement_retention_days = {retention.measurement_retention_days}`); an operator's "
        "`narration.toml` may set different ones.",
        "",
        "## Tools",
        "",
    ]
    lines += [f"- [`{name}`](#{name})" for name in TOOL_NAMES]
    lines.append("")
    for name in TOOL_NAMES:
        tool = TOOLS_BY_NAME[name]
        title = TOOL_TEXTS[name].title
        lines += [f"## `{name}`", "", f"{title}.", "", tool_description(name, retention), ""]
        lines += ["### Input", ""]
        lines += _field_table(field_rows(tool.input_schema))
        lines += ["### Output", ""]
        lines += _field_table(field_rows(tool.output_schema))
    lines += ["## Error codes", "", "Every tool error (`isError: true`) carries one of these (design section 14)."]
    lines.append("")
    lines += _table(
        ("Code", "Retryable", "Meaning", "Hint"),
        [
            (
                f"`{e.code}`",
                "maybe" if e.retryable is None else ("yes" if e.retryable else "no"),
                _escape(e.meaning),
                _escape(e.hint),
            )
            for e in codes.ERRORS.values()
        ],
    )
    lines += [
        "",
        "## Flag codes",
        "",
        "A result's `flags` (and, in a QA verdict, `qa.flags`) carry these (design section 14). "
        "**Retake** is when a flag triggers an automatic retake: `always` at any severity it can carry, "
        "`at_fail` only at severity `fail`, `never`.",
        "",
    ]
    lines += _table(
        ("Code", "Severities", "Retake", "Meaning"),
        [(f"`{f.code}`", ", ".join(f.severities), f"`{f.retake}`", _escape(f.meaning)) for f in codes.FLAGS.values()],
    )
    lines.append("")
    text = "\n".join(lines)
    return text if text.endswith("\n") else text + "\n"


def write(text: str, path: Path = OUTPUT) -> None:
    """Write ``text`` to ``path`` deterministically: a temporary file beside it, then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if docs/tools.md would change")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else None
        if current == text:
            return 0
        print(f"{OUTPUT} is stale. Run `uv run python tools/gen_tool_reference.py` and commit the result.")
        return 1
    write(text)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
