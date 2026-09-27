"""``python -m narration.daemon --store <store_root> --config <path>``: run the daemon (design sections 4, 4.1).

``-m narration.daemon --store <store_root>`` is the daemon's process-identity marker (section 4.1). The
daemon is normally started detached, through ``narration.daemon.start``; run in a terminal it serves in the
foreground until stopped (``narration-admin daemon stop``, or Ctrl+C, which stops it at once).

Start-up order matters:

1. the arguments and the configuration are read with the standard library and ``narration.config`` only;
2. the CPU thread cap (``[workers] cpu_threads``, section 4.1) is put in this process's environment, since
   post-processing runs here and numpy starts its thread pools when it is first imported;
3. only then are the store, the platform and the rest imported (they import numpy).

Exit codes: 0 when the daemon ran and stopped, or another daemon already runs for this store (a second
daemon exits quietly); 1 on an unexpected error (the log at ``<store_root>/logs/daemon.log`` has it); 2 for
bad arguments or configuration.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Final

from narration_worker.threads import cap_threads_env

from narration.config import Config, load_config
from narration.contracts.errors import ConfigError

from .settings import LOG_NAME  # standard library and narration.config only: no numpy yet

BEGAN: Final = time.time()
"""When this module began to run (Unix seconds): the daemon's launch time when no launcher gave one."""
DEFAULT_RUNNER: Final = "narration.jobs.runner:default_runner"
"""The job runner the daemon drives unless ``--runner`` names another: the job engine (WP31).
``--runner narration.daemon.seam:NullRunner`` runs the daemon without it (it never finds work)."""
LOG_MAX_BYTES: Final = 5 * 1024 * 1024
LOG_BACKUPS: Final = 3

_EXIT_ERROR: Final = 1
_EXIT_USAGE: Final = 2
log = logging.getLogger("narration.daemon")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m narration.daemon", description="Run narration-mcp's daemon for one store."
    )
    parser.add_argument("--store", required=True, type=Path, help="the store root (section 15)")
    parser.add_argument("--config", required=True, type=Path, help="the configuration file (section 16)")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument(
        "--launched-at",
        type=float,
        default=None,
        help="when the launcher started this process (Unix seconds; start_detached sets it)",
    )
    dev = parser.add_argument_group("development and tests")
    dev.add_argument("--fake-workers", action="store_true", help="serve every worker group with the fake role")
    dev.add_argument("--runner", default=DEFAULT_RUNNER, help="module:callable that returns the JobRunner")
    dev.add_argument("--idle-unload-s", type=float, default=None, help="override [daemon] idle_unload_s")
    dev.add_argument("--idle-exit-s", type=float, default=None, help="override [daemon] idle_exit_min, in seconds")
    dev.add_argument("--poll-s", type=float, default=None, help="how often to look for commands and work")
    return parser


def launch_time(given: float | None, began: float) -> float:
    """This daemon's launch time: the launcher's ``--launched-at``, or ``began`` without one. A launcher
    cannot have started this process after it began to run, so a later ``given`` is not believed. A value
    that is not a finite number (``nan``, ``inf``, which ``float`` accepts) is logged and ignored."""
    if given is not None and not math.isfinite(given):
        log.warning("--launched-at %r is not a finite time; using when this process began instead", given)
        given = None
    return began if given is None else min(given, began)


def warn_without_safe_path(safe_path: bool) -> bool:
    """Log one warning when this interpreter was started without ``-P`` (``sys.flags.safe_path`` off): then
    the working folder, the store root, is on ``sys.path``, and a module there could be imported in place of
    the service's own (section 17). True if it warned."""
    if safe_path:
        return False
    log.warning(
        "the daemon was started without -P, so its working folder is on sys.path and a module there could be "
        "imported in place of the service's own; start it with narration-admin daemon start, or through "
        "narration.daemon.start.start_detached, which pass -P"
    )
    return True


def _same_folder(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def main(argv: list[str] | None = None) -> int:
    """Run the daemon; returns the process exit code."""
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"narration daemon: {exc}", file=sys.stderr)
        return _EXIT_USAGE
    if not _same_folder(args.store, config.server.store_root):
        print(
            f"narration daemon: --store {args.store} is not the configuration's [server] store_root "
            f"({config.server.store_root})",
            file=sys.stderr,
        )
        return _EXIT_USAGE
    cap_threads_env(config.workers.cpu_threads)  # before anything below imports numpy (section 4.1)
    return _serve(args, config)


def _serve(args: argparse.Namespace, config: Config) -> int:
    import importlib

    from narration.platform import get_platform
    from narration.store import NarrationStore

    from .seam import JobRunner
    from .service import Daemon
    from .settings import DaemonSettings

    try:
        settings = DaemonSettings.from_config(
            config,
            store_root=Path(os.path.abspath(args.store)),
            fake_workers=args.fake_workers,
            idle_unload_s=args.idle_unload_s,
            idle_exit_s=args.idle_exit_s,
            poll_s=args.poll_s,
        )
    except ValueError as exc:
        print(f"narration daemon: {exc}", file=sys.stderr)
        return _EXIT_USAGE
    platform = get_platform()
    # The daemon runs in the store root: this OS's hardening applies to it and to everything it starts, as
    # to its workers (section 17; on Windows, cmd.exe no longer looks for programs in the working folder).
    os.environ.update(platform.hardening_env())
    try:
        store = NarrationStore(settings.store_root, platform, retention=config.retention)
    except Exception as exc:
        print(f"narration daemon: cannot open the store at {settings.store_root}: {exc}", file=sys.stderr)
        return _EXIT_ERROR
    try:
        _log_to_file(store.layout.logs_dir(), args.log_level)
        warn_without_safe_path(bool(sys.flags.safe_path))
        try:
            module_name, _, attr = args.runner.partition(":")
            runner = getattr(importlib.import_module(module_name), attr)()
        except Exception:
            log.exception("cannot load the job runner %r", args.runner)
            return _EXIT_USAGE
        if not isinstance(runner, JobRunner):
            log.error("%r is not a JobRunner (it needs step, has_work and shutdown)", args.runner)
            return _EXIT_USAGE
        try:
            daemon = Daemon(
                settings=settings,
                config=config,
                store=store,
                platform=platform,
                runner=runner,
                launched_at=launch_time(args.launched_at, BEGAN),
            )
            return daemon.run()
        except Exception:
            log.exception("the daemon failed")
            return _EXIT_ERROR
    finally:
        store.close()
        logging.shutdown()


def _log_to_file(folder: Path, level: str) -> None:
    """Log to ``<store_root>/logs/daemon.log`` (rotated), and to stderr when there is one to write to."""
    from logging.handlers import RotatingFileHandler

    folder.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s pid %(process)d %(threadName)s %(name)s %(levelname)s %(message)s")
    handlers: list[logging.Handler] = [
        RotatingFileHandler(folder / LOG_NAME, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
    ]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    root = logging.getLogger()
    root.setLevel(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)


if __name__ == "__main__":
    sys.exit(main())
