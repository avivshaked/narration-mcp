"""``python -m narration_worker --role <role> --store <store_root>``: run one worker (design section 4.1).

The daemon starts each worker venv's Python directly with this command line, which is also the worker's
process-identity marker. Start-up order matters:

1. the CUDA and thread-pool environment is fixed (``CUBLAS_WORKSPACE_CONFIG``, the thread variables)
   before anything imports torch or numpy;
2. stdin and stdout are taken for the protocol, and everything else is pointed at stderr;
3. the role's handler is loaded and the request loop runs.

Exit codes: 0 after ``shutdown`` or at the end of input; 2 when the worker cannot start (a bad argument, a
missing store, no handler for the role), with the reason on stderr.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import get_args

from .determinism import prepare_cuda_env
from .handler import WorkerContext
from .protocol import WorkerRole
from .threads import DEFAULT_CPU_THREADS, cap_threads_env

EXIT_START_FAILED = 2

log = logging.getLogger("narration_worker")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m narration_worker", description="Run one narration-mcp worker.")
    parser.add_argument("--role", required=True, choices=get_args(WorkerRole), help="the worker role to serve")
    parser.add_argument("--store", required=True, type=Path, help="the store root; every file the worker uses is in it")
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS, help="the CPU thread cap (section 4.1)")
    parser.add_argument("--handler", default=None, help="module:Class of the handler, instead of the role's own")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the worker; returns the process exit code."""
    args = _parser().parse_args(argv)
    if args.cpu_threads < 1:
        print("--cpu-threads must be at least 1", file=sys.stderr)
        return EXIT_START_FAILED
    prepare_cuda_env()
    cap_threads_env(args.cpu_threads)

    from .stdio import claim_stdio

    streams = claim_stdio()
    logging.basicConfig(
        stream=sys.stderr,
        level=args.log_level,
        format=f"%(asctime)s {args.role} %(name)s %(levelname)s %(message)s",
    )
    store = args.store
    if not store.is_absolute() or not store.is_dir():
        log.error("--store must be an existing folder given as an absolute path: %s", store)
        return EXIT_START_FAILED

    from .loop import serve
    from .roles import RoleUnavailable, resolve_handler

    try:
        handler_cls = resolve_handler(args.role, args.handler)
        handler = handler_cls(WorkerContext(role=args.role, store_root=store.resolve(), cpu_threads=args.cpu_threads))
    except RoleUnavailable as exc:
        log.error("%s", exc)
        return EXIT_START_FAILED
    except Exception:
        log.exception("the %s handler could not start", args.role)
        return EXIT_START_FAILED
    return serve(handler, streams.reader, streams.writer)


if __name__ == "__main__":
    sys.exit(main())
