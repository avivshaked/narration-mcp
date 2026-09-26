"""Spike (g): does a daemon started from an MCP session outlive that session? (design sections 4, 4.1)

An MCP client starts a stdio server as its child, and ends it when the session closes; some clients run it
in a Job Object that kills everything in it when the job closes. This script models such a session as a
child process (``--session``) that starts the daemon, waits until it serves, and then exits at once. The
orchestrator (no arguments) runs one scenario per way a session can be set up, then checks from outside:

- is the daemon still alive after its session is gone, and does it still answer a command (``release_gpu``)
  and render a job with its (fake) workers?
- which processes of the daemon's tree have a console (a ``conhost.exe`` child), and does any of them own a
  window (``MainWindowHandle``, read with PowerShell's ``Get-Process`` for those pids only)?
- does ``stop`` through the store end it?

Scenarios:

- ``plain``: the session runs in whatever Job Objects its launcher gives it; ``start_detached``;
- ``uv-run``: the session is started through ``uv run --no-sync python …``, as an MCP client configured with
  ``uv run`` would; ``start_detached``;
- ``job-breakaway-ok``: the session joins a kill-on-close Job Object that allows breakaway; ``start_detached``;
- ``job-no-breakaway``: the same, but the job forbids breakaway: ``start_detached`` must refuse
  (``DAEMON_UNAVAILABLE``) and start nothing;
- ``control-no-breakaway-flag``: the job allows breakaway, but the session starts the daemon without
  ``CREATE_BREAKAWAY_FROM_JOB`` (only ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``): the control that
  shows what breakaway is for;
- ``console-python``: ``plain``, but the daemon runs as the venv's ``python.exe`` rather than ``pythonw.exe``:
  the control for the console finding.

Every process this script measures or stops is one it started: a daemon is accepted only when it was
created after its scenario began and its parent is the launcher its session got back. Stores live under
``<repo>/.dev/spike-g/``. The results (no paths, no audio) go to ``results/``.

    uv run python spikes/g-detached-daemon/spike_g.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform as pyplatform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO / ".dev" / "spike-g"
RESULTS = HERE / "results"
RUNNER = "narration.daemon.testing:FakeWorkerRunner"
DAEMON_OPTIONS = ["--fake-workers", "--runner", RUNNER, "--idle-exit-s", "120", "--poll-s", "0.05"]
SCENARIOS: dict[str, dict[str, str]] = {
    "plain": {"job": "none", "how": "start_detached", "via": "python"},
    "uv-run": {"job": "none", "how": "start_detached", "via": "uv"},
    "job-breakaway-ok": {"job": "breakaway", "how": "start_detached", "via": "python"},
    "job-no-breakaway": {"job": "no-breakaway", "how": "start_detached", "via": "python"},
    "control-no-breakaway-flag": {"job": "breakaway", "how": "no-breakaway-flag", "via": "python"},
    "console-python": {"job": "none", "how": "console-python", "via": "python"},
}


# ---------------------------------------------------------------- the session (a child process)
def session(spec: dict[str, Any]) -> None:
    """Set up like an MCP session, start the daemon, wait until it serves, write what happened, exit."""
    from narration.contracts.errors import NarrationError
    from narration.daemon.start import daemon_argv, start_detached
    from narration.daemon.sweep import read_status
    from narration.platform import get_platform
    from narration.platform._windows import _JobObject, _own_job_breakaway  # pyright: ignore[reportPrivateUsage]
    from narration.store import NarrationStore

    began = time.time()
    store_root, config = Path(spec["store"]), Path(spec["config"])
    result: dict[str, Any] = {"job_before": list(_own_job_breakaway())}
    if spec["job"] != "none":
        job = _JobObject(kill_on_close=True, allow_breakaway=spec["job"] == "breakaway")
        job.add(os.getpid())
    result["job_after"] = list(_own_job_breakaway())
    try:
        if spec["how"] == "start_detached":
            result["spawned_pid"] = start_detached(store_root, config, extra=DAEMON_OPTIONS)
        elif spec["how"] == "console-python":
            store_root.mkdir(parents=True, exist_ok=True)
            argv = daemon_argv(store_root, config, extra=DAEMON_OPTIONS)
            argv[0] = str(Path(sys.executable).with_name("python.exe"))  # not pythonw.exe
            result["spawned_pid"] = get_platform().spawn_detached(argv, cwd=store_root, env=dict(os.environ))
        else:  # no-breakaway-flag: detached, but still in this session's job
            store_root.mkdir(parents=True, exist_ok=True)
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            child = subprocess.Popen(
                daemon_argv(store_root, config, extra=DAEMON_OPTIONS),
                cwd=store_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=flags,
            )
            result["spawned_pid"] = child.pid
    except NarrationError as exc:
        result["error"] = {"code": exc.code, "retry_after_s": exc.retry_after_s, "details": exc.details}
    if "spawned_pid" in result:
        with NarrationStore(store_root, get_platform()) as store:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                status = read_status(store)
                if status is not None and status.state == "idle" and status.pid is not None:
                    result["daemon_pid"] = status.pid
                    break
                time.sleep(0.05)
    result["serving_before_exit_s"] = round(time.time() - began, 2) if "daemon_pid" in result else None
    out = Path(spec["out"])
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, out)
    os._exit(0)  # the session ends: a kill-on-close job it is in closes now


# ---------------------------------------------------------------- the orchestrator
def alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def ours(pid: int, launcher: int, began: float) -> bool:
    """Whether ``pid`` is the process the scenario's session started (or that launcher's interpreter)."""
    try:
        process = psutil.Process(pid)
        return process.create_time() >= began - 1.0 and (pid == launcher or process.ppid() == launcher)
    except psutil.NoSuchProcess:
        return False


def wait_for(predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def window_handles(pids: list[int]) -> dict[int, bool]:
    """Which of these processes (all this script's own) own a window, as PowerShell's Get-Process says."""
    if not pids:
        return {}
    command = (
        f"Get-Process -Id {','.join(map(str, pids))} -ErrorAction SilentlyContinue | "
        'ForEach-Object { "$($_.Id) $([int64]$_.MainWindowHandle)" }'
    )
    done = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    seen: dict[int, bool] = {}
    for line in done.stdout.splitlines():
        pid, _, handle = line.strip().partition(" ")
        if pid.isdigit() and handle.lstrip("-").isdigit():
            seen[int(pid)] = int(handle) != 0
    return seen


def tree(launcher: int, daemon: int) -> list[dict[str, Any]]:
    """The processes under the launcher, by role, with no paths: name, role, parent role, window."""
    roles: dict[int, str] = {launcher: "daemon-launcher"}
    if daemon != launcher:
        roles[daemon] = "daemon"
    facts: dict[int, tuple[str, int, list[str]]] = {}
    try:
        members = [psutil.Process(launcher), *psutil.Process(launcher).children(recursive=True)]
    except psutil.NoSuchProcess:
        return []
    for member in members:
        with contextlib.suppress(psutil.NoSuchProcess):
            facts[member.pid] = (member.name().lower(), member.ppid(), member.cmdline())
    for _ in range(len(facts)):  # parents before children, whatever order psutil listed them in
        for pid, (name, parent, argv) in facts.items():
            if pid in roles or parent not in roles:
                continue
            if name == "conhost.exe":
                roles[pid] = f"console-of-{roles[parent]}"
            elif roles[parent] == "daemon" and "narration_worker" in argv:
                roles[pid] = "worker-launcher"
            elif roles[parent] in ("daemon", "worker-launcher"):
                roles[pid] = "worker"
            else:
                roles[pid] = "other"
    rows: list[dict[str, Any]] = [
        {
            "pid": pid,
            "role": roles.get(pid, "?"),
            "name": name,
            "parent_role": roles.get(parent, "outside"),
            "args": [a for a in argv[1:] if not os.path.isabs(a)][:3],  # no paths
        }
        for pid, (name, parent, argv) in facts.items()
    ]
    windows = window_handles([row["pid"] for row in rows])
    for row in rows:
        row["has_window"] = windows.get(row.pop("pid"))
    return rows


def run_scenario(name: str, setup: dict[str, str]) -> dict[str, Any]:
    from narration.daemon.start import running_daemon
    from narration.platform import get_platform
    from narration.store import NarrationStore

    sys.path.insert(0, str(REPO))
    from tests.daemon.conftest import make_job

    folder = WORK / name
    shutil.rmtree(folder, ignore_errors=True)  # this script's own folder under .dev
    for role in ("qwen3tts", "qa"):
        (folder / "workers" / role).mkdir(parents=True)
    config = folder / "narration.toml"
    config.write_text("[server]\nstore_root = 'store'\nmodels_root = 'models'\n", encoding="utf-8")
    out = folder / "session.json"
    spec = {**setup, "store": str(folder / "store"), "config": str(config), "out": str(out)}
    head = [sys.executable] if setup["via"] == "python" else [shutil.which("uv") or "uv", "run", "--no-sync", "python"]
    record: dict[str, Any] = {"scenario": name, **setup}
    began = time.time()
    session_run = subprocess.run(
        [*head, str(Path(__file__).resolve()), "--session", json.dumps(spec)],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    record["session_exit_code"] = session_run.returncode
    if not out.exists():
        record["session_stderr_tail"] = session_run.stderr[-800:]
        return record
    said = json.loads(out.read_text(encoding="utf-8"))
    record["session_job_before"], record["session_job_after"] = said["job_before"], said["job_after"]
    record["start_error"] = said.get("error")
    record["serving_before_session_exit_s"] = said.get("serving_before_exit_s")
    launcher, daemon = said.get("spawned_pid"), said.get("daemon_pid")
    with NarrationStore(folder / "store", get_platform()) as store:
        if launcher is None:
            time.sleep(1.0)
            record["daemon_json_after"] = None if store.get_daemon_status() is None else "present"
            return record
        record["daemon_served_before_session_exit"] = daemon is not None
        if daemon is None:
            _cleanup(launcher, daemon, began)
            return record
        time.sleep(1.5)  # the session is gone; give a closing job time to act
        record["daemon_alive_after_session"] = alive(daemon) and ours(daemon, launcher, began)
        if not record["daemon_alive_after_session"]:
            record["launcher_alive_after_session"] = alive(launcher) and ours(launcher, launcher, began)
            record["running_daemon_after"] = running_daemon(store) is not None
            _cleanup(launcher, daemon, began)
            return record
        record["daemon_is_launcher_child"] = daemon != launcher
        record["daemon_name"] = psutil.Process(daemon).name().lower()
        asked = time.monotonic()
        answer = store.wait_for_command(store.post_command("release_gpu").command_id, timeout_s=30)
        record["release_gpu_answered_after_session"] = answer is not None and answer.result is not None
        record["release_gpu_answer_s"] = round(time.monotonic() - asked, 3)
        job = make_job(store, "Spoken after the session ended.")
        finished = wait_for(lambda: (j := store.get_job(job.job_id)) is not None and j.status == "completed", 60)
        record["job_completed_after_session"] = finished
        time.sleep(0.3)
        record["tree"] = tree(launcher, daemon)
        posted = store.post_command("stop")
        stopped = store.wait_for_command(posted.command_id, timeout_s=30)
        record["stop_result"] = None if stopped is None else stopped.result
        record["exited_after_stop"] = wait_for(lambda: not alive(daemon) and not alive(launcher), 30)
        final = store.get_daemon_status()
        record["final_state"] = None if final is None else final.state
    _cleanup(launcher, daemon, began)
    return record


def _cleanup(launcher: int, daemon: int | None, began: float) -> None:
    """Kill what is left of this scenario's daemon, only when it is provably the one its session started."""
    for pid in (daemon, launcher):
        if pid is not None and alive(pid) and ours(pid, launcher, began):
            with contextlib.suppress(psutil.NoSuchProcess):
                process = psutil.Process(pid)
                members = [process, *process.children(recursive=True)]
                for member in members:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        member.kill()
                psutil.wait_procs(members, timeout=15)


def summary(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Scenario | Session job (in job, innermost allows breakaway) | Start | Alive after session | "
        "Answers / renders after | Consoles in the tree | Windows | Stop |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if r.get("skipped"):
            lines.append(f"| `{r['scenario']}` | skipped: {r['skipped']} | | | | | | |")
            continue
        start = (r.get("start_error") or {}).get("code") or (
            "started" if r.get("daemon_served_before_session_exit") else "no daemon"
        )
        rows = r.get("tree", [])
        consoles = ", ".join(row["role"] for row in rows if row["name"] == "conhost.exe") or ("none" if rows else "")
        windows = ", ".join(row["role"] for row in rows if row["has_window"]) or ("none" if rows else "")
        after = (
            f"{r.get('release_gpu_answered_after_session')} / {r.get('job_completed_after_session')}"
            if r.get("daemon_alive_after_session")
            else ""
        )
        stop = f"{r.get('final_state')} (exited: {r.get('exited_after_stop')})" if "stop_result" in r else ""
        lines.append(
            f"| `{r['scenario']}` | {r.get('session_job_after')} | {start} | "
            f"{r.get('daemon_alive_after_session', '')} | {after} | {consoles} | {windows} | {stop} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--session", help="run as the session stand-in (JSON spec); used by the orchestrator")
    parser.add_argument("--only", nargs="*", choices=sorted(SCENARIOS), help="run only these scenarios")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("spike (g) measures Windows process mechanics; run it on Windows")
    if args.session:
        session(json.loads(args.session))
        return
    records = []
    for name, setup in SCENARIOS.items():
        if args.only and name not in args.only:
            continue
        if setup["via"] == "uv" and shutil.which("uv") is None:
            records.append({"scenario": name, **setup, "skipped": "uv is not on PATH"})
            continue
        print(f"-- {name}", flush=True)
        records.append(run_scenario(name, setup))
        print(json.dumps(records[-1], indent=1), flush=True)
    RESULTS.mkdir(exist_ok=True)
    facts = {
        "date": time.strftime("%Y-%m-%d"),
        "python": pyplatform.python_version(),
        "windows": pyplatform.version(),
        "psutil": psutil.__version__,
    }
    payload = json.dumps({"environment": facts, "scenarios": records}, indent=1, ensure_ascii=False) + "\n"
    for name, text in (("spike-g.json", payload), ("summary.md", summary(records))):
        tmp = RESULTS / f".{name}.tmp"
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, RESULTS / name)
    print(summary(records))


if __name__ == "__main__":
    main()
