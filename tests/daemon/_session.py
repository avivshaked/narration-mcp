"""An MCP session stand-in for the process tests (run as a script: ``python _session.py …``).

It optionally joins a kill-on-close Job Object (as an MCP client's session may run in one), starts the daemon
detached through ``narration.daemon.start``, writes what happened to ``out`` (a temp name, then renamed),
and exits at once, which kills everything left in its job. It touches no other process.

    python _session.py <store_root> <config> <out> <none|job|no-breakaway|nested> [daemon options…]

``job`` allows breakaway and ``no-breakaway`` forbids it. ``nested`` joins a silent-breakaway job like a venv
launcher's, nested in whatever job the test started this process in (the MCP Python SDK's kill-on-close job,
say): the topology the lead saw a daemon die in (spike k).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from narration.contracts.errors import NarrationError
from narration.daemon.start import start_detached

if sys.platform != "win32":
    raise SystemExit("the session stand-in runs on Windows only")


def main(store: str, config: str, out: str, mode: str, *extra: str) -> None:
    from narration.platform._windows import _JobObject  # pyright: ignore[reportPrivateUsage]

    if mode != "none":
        job = _JobObject(kill_on_close=True, allow_breakaway=mode == "job", silent_breakaway=mode == "nested")
        job.add(os.getpid())
    try:
        result: dict[str, object] = {"spawned_pid": start_detached(Path(store), Path(config), extra=list(extra))}
    except NarrationError as exc:
        result = {"code": exc.code, "retry_after_s": exc.retry_after_s, "details": exc.details}
    tmp = Path(out + ".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, out)
    os._exit(0)  # a session that ends: its job, if any, closes and kills what is left in it


if __name__ == "__main__":
    main(*sys.argv[1:])
