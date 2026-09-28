"""The service opens no network listener, and only the operator's install step talks to the network (design
section 17 items 1, 7, 8 and 12; security review S0).

The front-end serves on stdio, the daemon and the front-end meet only through the store, and workers speak over
their standard streams. So no module of the service or its workers may import a server or socket module, and
only ``narration.admin.install`` (the model download, section 17.8) may import an HTTP client. The check reads
the source (``ast``), so it needs no worker venv and no model.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ROOT / "src" / "narration",
    ROOT / "workers" / "common" / "src",
    ROOT / "workers" / "qwen3tts" / "src",
    ROOT / "workers" / "qa" / "src",
)

LISTENERS = (
    "socket",
    "socketserver",
    "http.server",
    "xmlrpc.server",
    "asyncio.start_server",
    "uvicorn",
    "starlette",
    "fastapi",
    "gradio",
    "websockets",
    "aiohttp",
    "mcp.server.sse",
    "mcp.server.streamable_http",
    "mcp.server.streamable_http_manager",
)
"""Modules (or prefixes) that serve on or open a socket."""

CLIENTS = ("ssl", "urllib.request", "http.client", "requests", "httpx", "huggingface_hub")
"""Network clients: allowed only in the install step."""
DOWNLOADER = ROOT / "src" / "narration" / "admin" / "install.py"


def _imports(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "asyncio":
            yield f"asyncio.{node.attr}"


def _matches(name: str, modules: tuple[str, ...]) -> bool:
    return any(name == m or name.startswith(m + ".") for m in modules)


def _python_files() -> list[Path]:
    return sorted(p for source in SOURCES for p in source.rglob("*.py"))


def test_the_sources_are_found_s17_1() -> None:
    files = _python_files()
    assert DOWNLOADER in files
    assert any(p.name == "server.py" and p.parent.name == "mcp" for p in files)


def _offenders(modules: tuple[str, ...], *, allowed: Path | None = None) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in _python_files():
        names = sorted({name for name in _imports(path) if _matches(name, modules)})
        if names and path != allowed:
            found[path.relative_to(ROOT).as_posix()] = names
    return found


def test_no_module_opens_a_listener_s17_1() -> None:
    assert _offenders(LISTENERS) == {}, "the service serves on stdio only"


def test_only_the_install_step_uses_the_network_s17_8() -> None:
    assert _offenders(CLIENTS, allowed=DOWNLOADER) == {}, "only narration-admin install downloads"


def test_the_check_sees_an_import_s17_1(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    source = "import socket\nfrom http import server\nimport asyncio\nasyncio.start_server\n"
    planted.write_text(source, encoding="utf-8")
    assert {n for n in _imports(planted) if _matches(n, LISTENERS)} == {"socket", "http.server", "asyncio.start_server"}


def test_the_front_end_serves_on_stdio_only_s17_1() -> None:
    server = ROOT / "src" / "narration" / "mcp" / "server.py"
    names = set(_imports(server))
    assert "mcp.server.stdio.stdio_server" in names
    assert not {n for n in names if n.startswith("mcp.server.") and "http" in n}
