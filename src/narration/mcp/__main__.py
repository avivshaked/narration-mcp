"""``narration-mcp``: the stdio MCP server a client such as Claude Code starts (design sections 4, 5, 16).

::

    narration-mcp [--config <path>]

The configuration is found by the rule the operator CLI uses too (``narration.config.find_config``): the file
``--config`` names; else the one ``NARRATION_CONFIG`` names; else ``narration.toml`` in the service's folder
(``<service_root>``, the checkout this package runs from), as ``narration.example.toml`` describes. With none,
the server exits at once (code 2) with a message that says what to do; so does a configuration that does
not load.

The server serves one client on stdin and stdout, which belong to the protocol: its log goes to
``<store_root>/logs/narration-mcp.log`` (the path an ``INTERNAL`` error names) and, for warnings, to stderr.
It is stateless and one per client; the work runs in the daemon it starts detached on the first submission
(``[daemon] autostart``), which outlives it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

from narration.config import CONFIG_ENV, CONFIG_FILE_NAME, Config, find_config, load_config
from narration.contracts.errors import ConfigError

LOG_NAME: Final = "narration-mcp.log"
LOG_MAX_BYTES: Final = 5 * 1024 * 1024
LOG_BACKUPS: Final = 3
EXIT_USAGE: Final = 2
EXIT_ERROR: Final = 1

log = logging.getLogger("narration.mcp")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="narration-mcp",
        description="Serve narration-mcp's tools to one MCP client on stdio (design section 5).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"the configuration file (default: the file {CONFIG_ENV} names, else {CONFIG_FILE_NAME} in the service's "
        "folder)",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _log_to_file(folder: Path, level: str) -> Path:
    """Log to ``<store_root>/logs/narration-mcp.log`` (rotated), and warnings to stderr. Returns the log's
    path. stdout is the protocol's, so nothing is ever logged there."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOG_NAME
    formatter = logging.Formatter("%(asctime)s pid %(process)d %(name)s %(levelname)s %(message)s")
    to_file = RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
    to_file.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(to_file)
    if sys.stderr is not None:
        to_stderr = logging.StreamHandler(sys.stderr)
        to_stderr.setLevel(logging.WARNING)
        to_stderr.setFormatter(formatter)
        root.addHandler(to_stderr)
    return path


def serve(config: Config, *, log_level: str = "INFO") -> int:
    """Serve one client on stdio with the service's backend over ``config``'s store."""
    from narration_worker.threads import cap_threads_env

    cap_threads_env(config.workers.cpu_threads)  # before numpy is imported below (section 4.1)

    import anyio

    from narration.backend import backend_for
    from narration.platform import get_platform
    from narration.store import NarrationStore

    from .server import build_front_end

    root = config.server.store_root
    try:
        root.mkdir(parents=True, exist_ok=True)
        log_path = _log_to_file(root / "logs", log_level)
    except OSError as exc:
        print(f"narration-mcp: cannot use the store at {root}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    platform = get_platform()
    try:
        store = NarrationStore.from_config(config, platform)
    except Exception as exc:
        log.exception("cannot open the store at %s", root)
        print(f"narration-mcp: cannot open the store at {root}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    try:
        backend = backend_for(config, store, platform)
        front = build_front_end(backend, retention=config.retention, log_path=log_path)
        log.info("serving on stdio (store %s)", root)
        anyio.run(front.run_stdio)
    finally:
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server on stdio; returns the process exit code."""
    args = _parser().parse_args(argv)
    try:
        config = load_config(find_config(args.config))
    except ConfigError as exc:
        print(f"narration-mcp: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return serve(config, log_level=args.log_level)


if __name__ == "__main__":
    sys.exit(main())
