"""Spike (k): does the detached daemon leave every Job Object of the client that started it? (section 4.1)

The lead saw a detached daemon die with its MCP client, without a stop, on Windows 11. The client was the MCP
Python SDK's ``stdio_client``, which puts the server in a kill-on-close Job Object; the server was the venv's
``python.exe``, a launcher whose child is the interpreter. This script reproduces that without a GPU or a model
(the real daemon, with the ``NullRunner``, idles), then measures why, in two parts.

**Part 1, end to end.** A *client* starts a *server* stand-in the way a real client would, and the server calls
the real ``narration.daemon.start.start_detached`` for the real daemon. While the client's job still exists,
the client asks Windows whether the daemon's launcher and the daemon itself are in it (``IsProcessInJob`` with
the job's handle). Then the client ends, and the orchestrator checks from outside whether the daemon is still
alive 1.5 s later. Clients:

- ``sdk``: ``mcp.client.stdio.stdio_client`` (mcp 2.2.0), the real thing;
- ``job:<flags>``: a hand-made Job Object with given limit flags, the server assigned to it after the spawn
  (as the SDK and libuv do) or from its first instruction (spawned suspended, assigned, resumed);
- ``node``: a real Node.js client (``node_client.js``, ``child_process.spawn``), which puts its children in
  libuv's global job; Claude Code spawns through libuv too (its binary carries ``uv_spawn``);
- ``none``: a plain ``subprocess.Popen``, no job.

Each runs the server as the venv's ``python.exe`` (a launcher; the interpreter is its child in a job the
launcher makes) or as the base interpreter (``sys._base_executable``, told about the venv with
``__PYVENV_LAUNCHER__``, so no launcher and no extra job).

**Part 2, the mechanism** (``--role nest``). One process puts itself in an outer job, then in an inner job
nested in it, starts a suspended child with or without ``CREATE_BREAKAWAY_FROM_JOB``, and asks Windows which
jobs the child is in. This is the rule that decides part 1, measured in isolation.

Every process this script measures or ends is one it started (``tests/daemon/owned.py`` proves a daemon by
its pid's creation time before anything acts on it). Stores live under ``<repo>/.dev/spike-k/``; the
results (no paths) go to ``results/``.

    uv run python spikes/k-job-escape/job_escape.py --label before-fix
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform as pyplatform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import Any

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO / ".dev" / "spike-k"
RESULTS = HERE / "results"
NODE_CLIENT = HERE / "node_client.js"
DAEMON_OPTIONS = ["--runner", "narration.daemon.seam:NullRunner", "--idle-exit-s", "120", "--poll-s", "0.05"]

KILL = 0x2000
BREAKAWAY = 0x800
SILENT = 0x1000
DIE = 0x400
FLAG_NAMES = {
    KILL: "KILL_ON_JOB_CLOSE",
    BREAKAWAY: "BREAKAWAY_OK",
    SILENT: "SILENT_BREAKAWAY_OK",
    DIE: "DIE_ON_UNHANDLED_EXCEPTION",
}
JOBS: dict[str, int] = {
    "sdk-like": KILL,  # mcp 2.2.0, mcp/os/win32/utilities.py: KILL_ON_JOB_CLOSE only
    "libuv-like": KILL | BREAKAWAY | SILENT | DIE,  # libuv src/win/process.c, uv__init_global_job_handle
    "launcher-like": KILL | SILENT,  # what the venv launcher's job reports (measured: "orchestrator" below)
    "breakaway-ok": KILL | BREAKAWAY,
}
CREATE_SUSPENDED = 0x4
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
DETACHED_PROCESS = 0x8
CREATE_NEW_PROCESS_GROUP = 0x200
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_ACCESS_DENIED = 5

# The end-to-end scenarios: (client, how the server is assigned, the server's interpreter).
SCENARIOS: list[tuple[str, str, str]] = [
    ("sdk", "post", "launcher"),
    ("sdk", "post", "base"),
    ("job:sdk-like", "suspended", "launcher"),
    ("job:sdk-like", "suspended", "base"),
    ("job:sdk-like", "post", "launcher"),
    ("job:libuv-like", "suspended", "launcher"),
    ("job:libuv-like", "suspended", "base"),
    ("node", "post", "launcher"),
    ("node", "post", "base"),
    ("none", "-", "launcher"),
    ("none", "-", "base"),
]

# The nesting cases: (outer job flags, inner job flags, CREATE_BREAKAWAY_FROM_JOB on the child).
NEST_CASES: list[tuple[int | None, int | None, bool]] = [
    (KILL, KILL | SILENT, True),  # the SDK's job around the venv launcher's; the daemon asks to break away
    (KILL, KILL | SILENT, False),
    (KILL, KILL | BREAKAWAY, True),
    (KILL, KILL | BREAKAWAY, False),
    (KILL | BREAKAWAY | SILENT | DIE, KILL | SILENT, True),  # libuv's job around the venv launcher's
    (KILL | BREAKAWAY | SILENT | DIE, KILL | SILENT, False),
    (KILL | BREAKAWAY, KILL | SILENT, True),
    (KILL | BREAKAWAY, KILL | SILENT, False),
    (KILL, None, True),  # one job that forbids breakaway
    (KILL | SILENT, None, True),  # one job like the launcher's
    (KILL | SILENT, None, False),
    (KILL | BREAKAWAY, None, True),
    (None, None, True),  # no job at all
]


# ---------------------------------------------------------------- Job Objects, with the platform's declarations
def _w() -> Any:
    from narration.platform import _windows

    return _windows


def flag_names(flags: int) -> list[str]:
    names = [name for bit, name in FLAG_NAMES.items() if flags & bit]
    rest = flags & ~sum(FLAG_NAMES)
    return [*names, hex(rest)] if rest else names


class PidList(ctypes.Structure):
    _fields_ = (
        ("NumberOfAssignedProcesses", wintypes.DWORD),
        ("NumberOfProcessIdsInList", wintypes.DWORD),
        ("ProcessIdList", ctypes.c_size_t * 256),
    )


def job_facts() -> dict[str, Any]:
    """This process: is it in any job, and what are the innermost job's limit flags and member pids."""
    w = _w()
    in_job = wintypes.BOOL()
    facts: dict[str, Any] = {"pid": os.getpid(), "ppid": os.getppid()}
    if not w._IsProcessInJob(w._GetCurrentProcess(), None, ctypes.byref(in_job)):
        facts["in_any_job"] = None
        return facts
    facts["in_any_job"] = bool(in_job.value)
    if not in_job.value:
        return facts
    limits = w._ExtendedLimitInformation()
    if w._QueryInformationJobObject(None, 9, ctypes.byref(limits), ctypes.sizeof(limits), None):
        facts["innermost_flags"] = flag_names(limits.BasicLimitInformation.LimitFlags)
    pids = PidList()
    if w._QueryInformationJobObject(None, 3, ctypes.byref(pids), ctypes.sizeof(pids), None):
        facts["innermost_pids"] = list(pids.ProcessIdList[: pids.NumberOfProcessIdsInList])
    return facts


class Job:
    """A Job Object with the given limit flags; ``contains(pid)`` asks Windows; ``close`` may kill members."""

    def __init__(self, flags: int) -> None:
        w = _w()
        self.flags = flags
        handle = w._CreateJobObjectW(None, None)
        if not handle:
            raise w._last_error()
        limits = w._ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = flags
        if not w._SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = w._last_error()
            w._CloseHandle(handle)
            raise error
        self.handle = handle

    def add(self, pid: int) -> None:
        w = _w()
        process = w._OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if not process:
            raise w._last_error()
        try:
            if not w._AssignProcessToJobObject(self.handle, process):
                raise w._last_error()
        finally:
            w._CloseHandle(process)

    def contains(self, pid: int | None) -> bool | None:
        return in_job(pid, self.handle)

    def close(self) -> None:
        _w()._CloseHandle(self.handle)


def in_job(pid: int | None, job_handle: int | None) -> bool | None:
    """Whether process ``pid`` (one this script's tree started) is in the job, or in any job when ``job_handle``
    is None; None when the question cannot be asked (the process is gone)."""
    if pid is None:
        return None
    w = _w()
    process = w._OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not process:
        return None
    try:
        flag = wintypes.BOOL()
        if not w._IsProcessInJob(process, job_handle, ctypes.byref(flag)):
            return None
        return bool(flag.value)
    finally:
        w._CloseHandle(process)


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def wait_for(predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def exit_after(seconds: float) -> None:
    """Whatever happens to its parent, a stand-in never outlives this."""
    threading.Timer(seconds, lambda: os._exit(9)).start()


# ---------------------------------------------------------------- the server stand-in
def server(spec: dict[str, Any]) -> None:
    """Act as the front-end: report this process's job facts, start the real daemon detached, wait until it
    serves and prove it ours, write ``server_out``, then serve stdin until the client closes it."""
    exit_after(120)
    from narration.contracts.errors import NarrationError
    from narration.daemon.start import start_detached
    from narration.daemon.sweep import read_status
    from narration.platform import get_platform
    from narration.store import NarrationStore
    from tests.daemon.owned import capture_daemon

    store_root, config = Path(spec["store"]), Path(spec["config"])
    result: dict[str, Any] = {"server": job_facts()}
    try:
        result["spawned_pid"] = start_detached(store_root, config, extra=DAEMON_OPTIONS)
    except NarrationError as exc:
        result["error"] = {"code": exc.code, "retry_after_s": exc.retry_after_s, "details": exc.details}
    else:
        with NarrationStore(store_root, get_platform()) as store:

            def serving() -> bool:
                status = read_status(store)
                if status is not None and status.state == "idle" and status.pid is not None:
                    owned = capture_daemon(status, int(result["spawned_pid"]))
                    result["daemon_pid"] = status.pid
                    result["identity"] = owned.identity() if owned is not None else None
                    return True
                return False

            result["daemon_served"] = wait_for(serving, 45)
    write_json(Path(spec["server_out"]), result)
    sys.stdin.buffer.read()  # an MCP server serves until its client closes stdin
    os._exit(0)


def server_argv(spec: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """The server's command line and environment: the venv launcher, or the base interpreter told about the
    venv (``__PYVENV_LAUNCHER__`` is what the launcher itself sets for its child)."""
    env = dict(os.environ)
    if spec["server_python"] == "base":
        python = sys._base_executable  # pyright: ignore[reportAttributeAccessIssue]
        env["__PYVENV_LAUNCHER__"] = sys.executable
    else:
        python = sys.executable
    return [python, str(Path(__file__).resolve()), "--role", "server", json.dumps(spec)], env


# ---------------------------------------------------------------- the clients
def daemon_checks(said: dict[str, Any], job_handle: int | None) -> dict[str, Any]:
    """Whether the server, the daemon's launcher and the daemon are in the client's job (None each when the
    client made no job: the question does not apply), and whether the daemon's launcher is in any job."""
    server_pid = said.get("server", {}).get("pid")
    in_client_job = (lambda pid: None) if job_handle is None else (lambda pid: in_job(pid, job_handle))
    return {
        "server_in_client_job": in_client_job(server_pid),
        "daemon_launcher_in_client_job": in_client_job(said.get("spawned_pid")),
        "daemon_in_client_job": in_client_job(said.get("daemon_pid")),
        "daemon_launcher_in_any_job": in_job(said.get("spawned_pid"), None),
    }


def client(spec: dict[str, Any]) -> None:
    """A hand-made client: a Job Object with the scenario's flags (or none), the server assigned after the
    spawn or from its first instruction, then the end of the session (the job closes, stdin closes)."""
    exit_after(120)
    kind, assign = spec["client"], spec["assign"]
    result: dict[str, Any] = {"client": job_facts()}
    job = Job(JOBS[kind.partition(":")[2]]) if kind.startswith("job:") else None
    argv, env = server_argv(spec)
    creationflags = CREATE_SUSPENDED if job is not None and assign == "suspended" else 0
    with (Path(spec["folder"]) / "server.stderr").open("wb") as errlog:
        process = subprocess.Popen(
            argv,
            env=env,
            cwd=REPO,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errlog,
            creationflags=creationflags,
        )
    result["spawned_server_pid"] = process.pid
    if job is not None:
        job.add(process.pid)
        result["job_flags"] = flag_names(job.flags)
        if creationflags & CREATE_SUSPENDED:
            psutil.Process(process.pid).resume()
    server_out = Path(spec["server_out"])
    result["server_reported"] = wait_for(server_out.exists, 60)
    if result["server_reported"]:
        said = json.loads(server_out.read_text(encoding="utf-8"))
        result["server_launcher_in_client_job"] = job.contains(process.pid) if job is not None else None
        result.update(daemon_checks(said, job.handle if job is not None else None))
    # The session ends: stdin closes, then the job (if any) closes, which kills whatever is left in it.
    assert process.stdin is not None
    process.stdin.close()
    if job is not None:
        job.close()
    try:
        process.wait(timeout=10)
        result["server_exit_code"] = process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        result["server_exit_code"] = "killed by the client"
    write_json(Path(spec["client_out"]), result)
    os._exit(0)


def sdk_client(spec: dict[str, Any]) -> None:
    """The MCP Python SDK's own stdio client, as the lead's driver used it."""
    import anyio

    exit_after(120)
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.os.win32 import utilities

    argv, env = server_argv(spec)
    result: dict[str, Any] = {"client": job_facts(), "sdk": _version("mcp")}
    server_out = Path(spec["server_out"])

    async def run() -> None:
        params = StdioServerParameters(command=argv[0], args=argv[1:], env=env, cwd=str(REPO))
        with (Path(spec["folder"]) / "server.stderr").open("w", encoding="utf-8") as errlog:
            async with stdio_client(params, errlog=errlog):
                deadline = time.monotonic() + 60
                while not server_out.exists() and time.monotonic() < deadline:
                    await anyio.sleep(0.05)
                result["server_reported"] = server_out.exists()
                jobs = list(utilities._process_jobs.items())  # pyright: ignore[reportPrivateUsage]
                result["sdk_made_a_job"] = len(jobs) == 1
                if result["server_reported"] and jobs:
                    process, handle = jobs[0]
                    job_handle = int(handle)  # pyright: ignore[reportArgumentType]  # a pywin32 PyHANDLE
                    said = json.loads(server_out.read_text(encoding="utf-8"))
                    result["spawned_server_pid"] = process.pid
                    result["server_launcher_in_client_job"] = in_job(process.pid, job_handle)
                    result.update(daemon_checks(said, job_handle))
            # leaving the block: the SDK closes stdin, waits up to 2 s, kills the tree, and closes its job

    anyio.run(run)
    write_json(Path(spec["client_out"]), result)
    os._exit(0)


def node_client_argv(spec: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node is not on PATH")
    argv, env = server_argv(spec)
    return [node, str(NODE_CLIENT), spec["server_out"], spec["client_out"], *argv], env


def _version(package: str) -> str:
    from importlib.metadata import version

    return version(package)


# ---------------------------------------------------------------- part 2: the nesting rule
def nest(spec: dict[str, Any]) -> None:
    """Join an outer job, then an inner one nested in it; start a suspended child with or without
    ``CREATE_BREAKAWAY_FROM_JOB``; record which jobs the child is in; kill it."""
    exit_after(60)
    result: dict[str, Any] = {"self_before": job_facts()}
    outer = Job(spec["outer"]) if spec["outer"] is not None else None
    if outer is not None:
        outer.add(os.getpid())
    inner = Job(spec["inner"]) if spec["inner"] is not None else None
    if inner is not None:
        inner.add(os.getpid())
    result["self_after"] = job_facts()
    flags = CREATE_SUSPENDED | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    if spec["breakaway_flag"]:
        flags |= CREATE_BREAKAWAY_FROM_JOB
    try:
        child = subprocess.Popen(
            [sys._base_executable, "-c", "import time; time.sleep(60)"],  # pyright: ignore[reportAttributeAccessIssue]
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=flags,
        )
    except OSError as exc:
        result["create_process"] = {"winerror": exc.winerror, "strerror": exc.strerror}
    else:
        result["create_process"] = "ok"
        result["child_in_outer"] = outer.contains(child.pid) if outer is not None else None
        result["child_in_inner"] = inner.contains(child.pid) if inner is not None else None
        result["child_in_any_job"] = in_job(child.pid, None)
        child.kill()
        child.wait(timeout=10)
    write_json(Path(spec["out"]), result)
    os._exit(0)  # this process is inside its own kill-on-close jobs; leaving them is not possible


def run_nest_cases() -> list[dict[str, Any]]:
    folder = WORK / "nest"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    env = dict(os.environ, __PYVENV_LAUNCHER__=sys.executable)  # the base interpreter: no launcher, no extra job
    records: list[dict[str, Any]] = []
    for index, (outer, inner, flag) in enumerate(NEST_CASES):
        out = folder / f"case-{index}.json"
        spec = {"outer": outer, "inner": inner, "breakaway_flag": flag, "out": str(out)}
        done = subprocess.run(
            [sys._base_executable, str(Path(__file__).resolve()), "--role", "nest", json.dumps(spec)],  # pyright: ignore[reportAttributeAccessIssue]
            env=env,
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        record: dict[str, Any] = {
            "outer": flag_names(outer) if outer is not None else None,
            "inner": flag_names(inner) if inner is not None else None,
            "CREATE_BREAKAWAY_FROM_JOB": flag,
        }
        if out.exists():
            said = json.loads(out.read_text(encoding="utf-8"))
            record["process_in_any_job_before"] = said["self_before"]["in_any_job"]
            record["process_innermost_after"] = said["self_after"].get("innermost_flags")
            record["create_process"] = said["create_process"]
            for key in ("child_in_outer", "child_in_inner", "child_in_any_job"):
                record[key] = said.get(key)
        else:
            record["failed"] = done.stderr[-500:]
        records.append(record)
    return records


# ---------------------------------------------------------------- the orchestrator
def run_scenario(client_kind: str, assign: str, server_python: str) -> dict[str, Any]:
    from narration.daemon.start import running_daemon
    from narration.platform import get_platform
    from narration.store import NarrationStore
    from tests.daemon.owned import OwnedDaemon, reattach

    name = f"{client_kind}--{assign}--{server_python}".replace(":", "-")
    folder = WORK / name
    shutil.rmtree(folder, ignore_errors=True)
    for role in ("qwen3tts", "qa"):
        (folder / "workers" / role).mkdir(parents=True)
    config = folder / "narration.toml"
    config.write_text("[server]\nstore_root = 'store'\nmodels_root = 'models'\n", encoding="utf-8")
    spec = {
        "client": client_kind,
        "assign": assign,
        "server_python": server_python,
        "folder": str(folder),
        "store": str(folder / "store"),
        "config": str(config),
        "server_out": str(folder / "server.json"),
        "client_out": str(folder / "client.json"),
    }
    record: dict[str, Any] = {"scenario": name, "client": client_kind, "assign": assign, "server_python": server_python}
    if client_kind == "node":
        argv, env = node_client_argv(spec)
    else:
        role = "sdk-client" if client_kind == "sdk" else "client"
        argv, env = [sys.executable, str(Path(__file__).resolve()), "--role", role, json.dumps(spec)], dict(os.environ)
    client_run = subprocess.run(
        argv, env=env, cwd=REPO, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=150, check=False
    )
    record["client_exit_code"] = client_run.returncode
    client_out, server_out = folder / "client.json", folder / "server.json"
    if client_out.exists():
        record["client_said"] = json.loads(client_out.read_text(encoding="utf-8"))
    else:
        record["client_stderr_tail"] = client_run.stderr[-800:]
    owned: OwnedDaemon | None = None
    with NarrationStore(folder / "store", get_platform()) as store:
        if not server_out.exists():
            record["server_reported"] = False
            time.sleep(1.5)
            record["daemon_json_after"] = None if store.get_daemon_status() is None else "present"
            return record
        said = json.loads(server_out.read_text(encoding="utf-8"))
        record["server"] = said["server"]
        record["start_error"] = said.get("error")
        record["daemon_served_before_client_exit"] = said.get("daemon_served")
        identity = said.get("identity")
        record["daemon_identity_proven"] = identity is not None
        owned = reattach(identity) if identity is not None else None
        time.sleep(1.5)  # the client is gone; a closing job has acted by now
        status = store.get_daemon_status()
        record["daemon_json_after"] = None if status is None else status.state
        if owned is None:
            if identity is not None:  # proven while it served; both its processes are gone now
                record["daemon_alive_after_client"] = False
                record["daemon_launcher_alive_after_client"] = False
            record["running_daemon_after"] = running_daemon(store) is not None
            return record
        record["daemon_alive_after_client"] = bool(owned.daemon and owned.daemon.is_running())
        record["daemon_launcher_alive_after_client"] = bool(owned.launcher and owned.launcher.is_running())
        record["running_daemon_after"] = running_daemon(store) is not None
        if record["daemon_launcher_alive_after_client"] and owned.launcher is not None:
            # Measured from outside, for every client (the Node client cannot ask): the daemon's launcher, the
            # process spawn_detached created, is in no job at all once the client is gone.
            record["daemon_launcher_in_any_job_after_client"] = in_job(owned.launcher.pid, None)
        if record["daemon_alive_after_client"]:
            posted = store.post_command("stop")
            answer = store.wait_for_command(posted.command_id, timeout_s=30)
            record["stop_answered"] = answer is not None and answer.result is not None
            record["exited_after_stop"] = wait_for(lambda: not owned.running(), 30)
            final = store.get_daemon_status()
            record["final_state"] = None if final is None else final.state
    if owned is not None and owned.running():
        owned.kill()
    return record


def flags_text(names: list[str] | None) -> str:
    return " + ".join(names) if names else "none"


def summary(records: list[dict[str, Any]], nested: list[dict[str, Any]]) -> str:
    lines = [
        "## End to end",
        "",
        "| Client | Server assigned | Server runs as | Server in a job (innermost flags) | Start | "
        "Daemon launcher in client's job | Daemon in client's job | Daemon launcher in any job, after | "
        "Daemon alive after client | After |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        server = r.get("server") or {}
        said = r.get("client_said") or {}
        in_job_text = f"{server.get('in_any_job')} {server.get('innermost_flags', '')}".strip()
        launcher_after = r.get("daemon_launcher_in_any_job_after_client", "")
        error = r.get("start_error")
        if error:
            start = f"`{error['code']}` ({error['details'].get('reason')})"
        else:
            start = "started" if r.get("daemon_served_before_client_exit") else "no daemon"
        after = ""
        if "final_state" in r:
            after = f"stop → {r['final_state']} (exited: {r['exited_after_stop']})"
        elif r.get("daemon_alive_after_client") is False:
            after = f"daemon.json says `{r.get('daemon_json_after')}`; running_daemon: {r.get('running_daemon_after')}"
        elif error:
            after = f"daemon.json: {r.get('daemon_json_after')}"
        na = "n/a" if r["client"] in ("none", "node") else ""  # no client job, or a client that cannot ask
        launcher_in = said.get("daemon_launcher_in_client_job")
        daemon_in = said.get("daemon_in_client_job")
        lines.append(
            f"| `{r['client']}` | {r['assign']} | {r['server_python']} | {in_job_text} | {start} | "
            f"{na if launcher_in is None else launcher_in} | {na if daemon_in is None else daemon_in} | "
            f"{launcher_after} | {r.get('daemon_alive_after_client', '')} | {after} |"
        )
    lines += [
        "",
        "## The nesting rule (part 2)",
        "",
        "| Outer job | Inner job (nested) | CREATE_BREAKAWAY_FROM_JOB | CreateProcess | Child in outer | "
        "Child in inner | Child in any job |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in nested:
        made = n.get("create_process")
        made_text = "ok" if made == "ok" else (f"error {made['winerror']}" if isinstance(made, dict) else str(made))
        lines.append(
            f"| {flags_text(n['outer'])} | {flags_text(n['inner'])} | "
            f"{n['CREATE_BREAKAWAY_FROM_JOB']} | {made_text} | {n.get('child_in_outer')} | {n.get('child_in_inner')} | "
            f"{n.get('child_in_any_job')} |"
        )
    return "\n".join(lines) + "\n"


def orchestrate(label: str, only: list[str] | None) -> int:
    if sys.platform != "win32":
        print("this spike measures Windows Job Objects; run it on Windows", file=sys.stderr)
        return 2
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    node = shutil.which("node")
    node_version = (
        subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip()
        if node
        else None
    )
    facts: dict[str, Any] = {
        "label": label,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "windows": pyplatform.version(),
        "python": pyplatform.python_version(),
        "mcp": _version("mcp"),
        "psutil": psutil.__version__,
        "node": node_version,
        "orchestrator": job_facts(),  # this process is the venv launcher's child: its innermost job is the launcher's
    }
    print(f"orchestrator: {facts['orchestrator']}")
    nested = run_nest_cases()
    for n in nested:
        print("nest:", json.dumps(n))
    records: list[dict[str, Any]] = []
    for client_kind, assign, server_python in SCENARIOS:
        name = f"{client_kind}--{assign}--{server_python}".replace(":", "-")
        if only and not any(o in name for o in only):
            continue
        if client_kind == "node" and node is None:
            records.append({"scenario": name, "client": client_kind, "skipped": "no node on PATH"})
            continue
        print(f"scenario {name} ...", flush=True)
        record = run_scenario(client_kind, assign, server_python)
        print(
            f"  start: {(record.get('start_error') or {}).get('code', 'ok')}; alive after client: "
            f"{record.get('daemon_alive_after_client')}; final: {record.get('final_state')}"
        )
        records.append(record)
    write_json(RESULTS / f"{label}.json", {**facts, "nested_jobs": nested, "scenarios": records})
    (RESULTS / f"{label}.md").write_text(
        f"# Spike (k), run `{label}`\n\nWindows {facts['windows']}, Python {facts['python']}, mcp {facts['mcp']}, "
        f"node {facts['node']}, run at {facts['run_at']}.\n\nThe orchestrator's own innermost job (the venv "
        f"launcher's): {facts['orchestrator'].get('innermost_flags')}.\n\n" + summary(records, nested),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--role", choices=("server", "client", "sdk-client", "nest"))
    parser.add_argument("spec", nargs="?", help="the role's JSON spec")
    parser.add_argument("--label", default="run", help="the results' file name")
    parser.add_argument("--only", action="append", help="run only scenarios whose name contains this")
    args = parser.parse_args()
    if args.role is None:
        return orchestrate(args.label, args.only)
    spec = json.loads(args.spec)
    {"server": server, "client": client, "sdk-client": sdk_client, "nest": nest}[args.role](spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
