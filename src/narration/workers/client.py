"""The daemon's side of one worker process (design Appendix A; plan.md WP16).

``SubprocessWorkerClient`` implements ``narration.contracts.interfaces.WorkerClient``:

- it starts the worker from an argument list (never a shell), with the environment ``launch.worker_env``
  builds, and exchanges ``hello`` (the role and protocol version must match);
- a worker that cannot start (no handler for its role, a handler that fails to import, a bad store: exit
  code ``EXIT_START_FAILED``; or a venv without ``narration_worker``), a missing venv, or a role or protocol
  mismatch is ``WorkerFailure`` (``BACKEND_NOT_INSTALLED``): the worker env is missing or broken (section
  14), so it is not retryable and the daemon does not respawn the worker;
- a writer thread sends requests, and a reader thread correlates replies with them by id, so requests may
  come from several threads; a request's timeout counts from the call, so a request queued behind a long
  one (or a large one the worker is not reading yet) never waits past its own timeout;
- ``request`` returns the reply, or raises ``WorkerFailure`` for ``ok: false``, ``WorkerTimeout`` when no
  reply comes in time (the worker is then stopped), and ``WorkerCrashed`` when the worker exits or writes
  anything that is not a protocol message (with its exit code and the tail of its stderr);
- a crash is reported as soon as the worker's stdout closes, and, should a grandchild keep that pipe open,
  within a fraction of a second of the process exiting; it is never waited out to the request's timeout;
- ``close`` sends ``shutdown``, waits, and kills the worker if it does not exit in time. It never blocks on
  a pipe a grandchild of the worker still holds open.

OS-specific start-up (below-normal priority, the kill-on-close group) is not done here. The daemon passes
``on_spawn``, called with the new process's pid before ``hello``, and ``creationflags``; both come from
``narration.platform`` (WP19, WP30). There is a moment between the start and ``on_spawn`` when the worker
is outside the group; it is waiting for its first request then, and starts nothing.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Final, cast

from narration_worker.errors import EXIT_START_FAILED
from narration_worker.framing import (
    FramingError,
    LineTooLong,
    decode_message,
    encode_message,
    is_request_id,
    read_line,
)

from narration.contracts.errors import WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.names import WorkerRole
from narration.contracts.worker import PROTOCOL_VERSION, HelloReply

from .launch import INSTALL_HINT, WorkerCommand

log = logging.getLogger(__name__)

SpawnHook = Callable[[int], None]
"""Called with a new worker's pid right after it starts (e.g. add it to a kill-on-close group)."""

POLL_S: Final = 0.25
"""How often a waiting request checks that the worker process is still running."""
DEFAULT_HELLO_TIMEOUT_S: Final = 120.0
DEFAULT_STDERR_TAIL_BYTES: Final = 64 * 1024
RELEASE_JOIN_S: Final = 2.0
"""How long ``close`` waits, in all, for the pipe threads to finish before leaving their streams open."""
_NO_WORKER_PACKAGE: Final = re.compile(r"No module named '?narration_worker'?\s*$", re.MULTILINE)
"""What Python says when the venv has no ``narration_worker`` at all (it then exits with code 1)."""
STDERR_CATCH_UP_S: Final = 1.0
"""How long a crash report waits for the worker's last stderr lines once its stdout has closed."""


class _Pending:
    """One request waiting for its reply."""

    __slots__ = ("error", "event", "reply")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reply: dict[str, Any] | None = None
        self.error: BaseException | None = None

    def resolve(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.event.set()

    def fail(self, error: BaseException) -> None:
        self.error = error
        self.event.set()


class SubprocessWorkerClient:
    """One worker process, spoken to over its stdin and stdout (see the module docstring)."""

    def __init__(
        self,
        command: WorkerCommand,
        *,
        cwd: Path | None = None,
        on_spawn: SpawnHook | None = None,
        creationflags: int = 0,
        hello_timeout_s: float = DEFAULT_HELLO_TIMEOUT_S,
        stderr_tail_bytes: int = DEFAULT_STDERR_TAIL_BYTES,
        exit_grace_s: float = 5.0,
    ) -> None:
        self._command = command
        self._cwd = cwd
        self._on_spawn = on_spawn
        self._creationflags = creationflags
        self._hello_timeout_s = hello_timeout_s
        self._tail_limit = stderr_tail_bytes
        self._exit_grace_s = exit_grace_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._outbox: queue.Queue[bytes | None] = queue.Queue()
        self._pending: dict[int, _Pending] = {}
        self._next_id = 1
        self._failure: tuple[str, int | None] | None = None
        self._closing = False
        self._tail = bytearray()
        self._tail_lock = threading.Lock()
        self._stdin_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._hello: HelloReply | None = None
        self._worker_log = logging.getLogger(f"narration.workers.{command.role}")

    # ------------------------------------------------------------------ the WorkerClient protocol
    @property
    def role(self) -> WorkerRole:
        return self._command.role

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def hello(self) -> HelloReply | None:
        """The worker's ``hello`` reply, once started."""
        return self._hello

    @property
    def exit_code(self) -> int | None:
        """The process's exit code, once it has exited."""
        return self._proc.poll() if self._proc is not None else None

    def start(self) -> HelloReply:
        """Start the worker and exchange ``hello``.

        Raises ``WorkerFailure`` (``BACKEND_NOT_INSTALLED``) when the process cannot start, exits with
        ``EXIT_START_FAILED``, or speaks another protocol or role; ``WorkerCrashed`` when it exits otherwise
        during start-up; ``WorkerTimeout`` when ``hello`` gets no reply in time. On any failure the process
        is stopped.
        """
        if self._proc is not None:
            raise RuntimeError(f"the {self.role} worker client was already started")
        try:
            self._proc = subprocess.Popen(
                list(self._command.argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(self._command.env),
                cwd=self._cwd,
                close_fds=True,
                creationflags=self._creationflags,
            )
        except OSError as exc:
            raise WorkerFailure(
                "BACKEND_NOT_INSTALLED",
                f"the {self.role} worker could not be started ({exc}): {INSTALL_HINT}",
                {"role": self.role, "python": self._command.argv[0]},
            ) from exc
        proc = self._proc
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        name = f"{self.role}-{proc.pid}"
        self._stdin_thread = threading.Thread(
            target=self._pump_stdin, args=(proc.stdin,), name=f"{name}-stdin", daemon=True
        )
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout, args=(proc.stdout,), name=f"{name}-stdout", daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, args=(proc.stderr,), name=f"{name}-stderr", daemon=True
        )
        for thread in (self._stdin_thread, self._stdout_thread, self._stderr_thread):
            thread.start()
        try:
            if self._on_spawn is not None:
                self._on_spawn(proc.pid)
            reply = self.request("hello", {}, timeout_s=self._hello_timeout_s)
        except WorkerCrashed as exc:
            self._stop(f"the {self.role} worker was stopped because it failed to start")
            self._release()
            if not _start_failed(exc):
                raise
            last = next((line for line in reversed(exc.stderr_tail.splitlines()) if line.strip()), "")
            raise WorkerFailure(
                "BACKEND_NOT_INSTALLED",
                f"the {self.role} worker could not start ({last.strip()[:300] or 'no reason given'}): {INSTALL_HINT}",
                {"role": self.role, "exit_code": exc.exit_code, "stderr_tail": exc.stderr_tail},
            ) from exc
        except BaseException:
            self._stop(f"the {self.role} worker was stopped because it failed to start")
            self._release()
            raise
        if reply.get("role") != self.role or reply.get("protocol") != PROTOCOL_VERSION:
            self.close(timeout_s=5.0)
            raise WorkerFailure(
                "BACKEND_NOT_INSTALLED",
                f"the worker answered as role {reply.get('role')!r} with protocol {reply.get('protocol')!r}; this "
                f"server needs role {self.role!r} with protocol {PROTOCOL_VERSION}: {INSTALL_HINT}",
                {"role": reply.get("role"), "protocol": reply.get("protocol")},
            )
        self._hello = cast(HelloReply, reply)
        log.info("%s worker started (pid %d)", self.role, proc.pid)
        return self._hello

    def request(self, op: str, payload: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]:
        """Send one request and wait for its reply (see the module docstring for what it raises).

        ``timeout_s`` counts from this call: it covers waiting to be written as well as the worker's work.
        """
        deadline = time.monotonic() + timeout_s
        if "id" in payload or "op" in payload:
            raise ValueError("the payload must not carry id or op; the client sets them")
        proc = self._proc
        if proc is None:
            raise RuntimeError(f"the {self.role} worker client has not been started")
        pending = _Pending()
        with self._lock:
            self._raise_if_failed()
            rid = self._next_id
            self._next_id += 1
            self._pending[rid] = pending
        try:
            data = encode_message({"id": rid, "op": op, **payload})
        except ValueError:
            with self._lock:
                self._pending.pop(rid, None)
            raise
        self._outbox.put(data)
        self._wait(pending, proc, op, rid, deadline, timeout_s)
        return self._outcome(pending)

    def is_alive(self) -> bool:
        """True while the process runs and has kept to the protocol."""
        return self._proc is not None and self._proc.poll() is None and self._failure is None

    def close(self, *, timeout_s: float = 10.0) -> None:
        """Send ``shutdown``, wait up to ``timeout_s`` for the exit, then kill. Safe to call more than once."""
        proc = self._proc
        if proc is None:
            return
        self._closing = True
        if proc.poll() is None and self._failure is None:
            try:
                self.request("shutdown", {}, timeout_s=timeout_s)
            except (WorkerCrashed, WorkerTimeout, WorkerFailure) as exc:
                log.debug("shutdown of the %s worker: %s", self.role, exc)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("the %s worker did not exit within %s s of shutdown; killing it", self.role, timeout_s)
        self._stop(f"the {self.role} worker was closed")
        self._release()

    def __enter__(self) -> SubprocessWorkerClient:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def stderr_tail(self) -> str:
        """The last lines the worker wrote to stderr (at most ``stderr_tail_bytes``)."""
        with self._tail_lock:
            text = bytes(self._tail).decode("utf-8", errors="replace")
        return text

    # ------------------------------------------------------------------ waiting and failing
    def _wait(
        self, pending: _Pending, proc: subprocess.Popen[bytes], op: str, rid: int, deadline: float, timeout_s: float
    ) -> None:
        while not pending.event.wait(min(POLL_S, max(0.0, deadline - time.monotonic()))):
            if proc.poll() is not None:
                # it exited; the reader normally reports this at once, unless a grandchild holds stdout open
                if not pending.event.wait(self._exit_grace_s):
                    self._fail(f"the {self.role} worker exited with code {proc.returncode}", proc.returncode)
                return
            if time.monotonic() >= deadline:
                with self._lock:
                    self._pending.pop(rid, None)
                if pending.event.is_set():
                    return
                self._stop(f"the {self.role} worker was stopped: {op} (id {rid}) got no reply within {timeout_s} s")
                raise WorkerTimeout(
                    f"{op} (id {rid}) got no reply from the {self.role} worker within {timeout_s} s; "
                    "the worker was stopped"
                )

    def _outcome(self, pending: _Pending) -> dict[str, Any]:
        if pending.error is not None:
            raise pending.error
        reply = pending.reply
        assert reply is not None
        if reply["ok"] is True:
            return reply
        error = reply.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str):
            details = error.get("details")
            raise WorkerFailure(error["code"], error["message"], details if isinstance(details, dict) else None)
        raise WorkerFailure("INTERNAL", f"the {self.role} worker replied ok: false without a valid error: {reply}")

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            message, exit_code = self._failure
            raise WorkerCrashed(message, exit_code=exit_code, stderr_tail=self.stderr_tail())

    def _fail(self, message: str, exit_code: int | None) -> None:
        """Record that the worker is gone (once) and fail every waiting request with ``WorkerCrashed``."""
        with self._lock:
            if self._failure is not None:
                return
            self._failure = (message, exit_code)
            waiting = list(self._pending.values())
            self._pending.clear()
        tail = self.stderr_tail()
        if not self._closing:
            log.error("%s; stderr tail:\n%s", message, tail[-2000:])
        for pending in waiting:
            pending.fail(WorkerCrashed(message, exit_code=exit_code, stderr_tail=tail))

    def _stop(self, message: str) -> None:
        """Fail waiting requests with ``message``, then kill and reap the process."""
        self._fail(message, None)
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
        if proc is not None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - a kill that does not land
                log.error("the %s worker (pid %d) survived a kill", self.role, proc.pid)

    def _release(self) -> None:
        """Stop the writer, let the pipe threads finish, and close our ends of the pipes.

        A stream whose thread is still in a read or a write is left open: closing a buffered stream another
        thread is using blocks until that call returns, which is never while a grandchild of the worker holds
        the pipe. Such a thread is a daemon thread; it closes its stream itself when the pipe ends.
        """
        proc = self._proc
        if proc is None:
            return
        self._outbox.put(None)
        deadline = time.monotonic() + RELEASE_JOIN_S
        for label, thread, stream in (
            ("stdin", self._stdin_thread, proc.stdin),
            ("stdout", self._stdout_thread, proc.stdout),
            ("stderr", self._stderr_thread, proc.stderr),
        ):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if stream is None:
                continue
            if thread is not None and thread.is_alive():
                log.warning(
                    "the %s worker's %s pipe is still in use (a process it started holds it?); leaving it open",
                    self.role,
                    label,
                )
                continue
            _close_quietly(stream)

    # ------------------------------------------------------------------ pipe threads
    def _pump_stdin(self, stream: IO[bytes]) -> None:
        try:
            while (data := self._outbox.get()) is not None:
                stream.write(data)
                stream.flush()
        except (OSError, ValueError) as exc:
            log.debug("writing to the %s worker failed (%s); its exit will be reported", self.role, exc)
        finally:
            _close_quietly(stream)

    def _pump_stdout(self, stream: IO[bytes]) -> None:
        reason: str | None = None
        try:
            while True:
                try:
                    line = read_line(stream)
                except LineTooLong:
                    reason = "wrote a line longer than the protocol allows"
                    break
                if line is None:
                    break
                try:
                    message = decode_message(line)
                except FramingError as exc:
                    reason = f"wrote a line that is not a protocol message ({exc}): {line[:200]!r}"
                    break
                rid = message.get("id")
                if not is_request_id(rid) or not isinstance(message.get("ok"), bool):
                    reason = f"wrote a reply without an integer id and a boolean ok: {str(message)[:200]}"
                    break
                with self._lock:
                    pending = self._pending.pop(rid, None)
                if pending is None:
                    reason = f"replied to id {rid}, which no request is waiting for"
                    break
                pending.resolve(message)
        except (OSError, ValueError):
            pass  # our end was closed
        finally:
            self._stream_ended(reason)
            _close_quietly(stream)

    def _stream_ended(self, reason: str | None) -> None:
        proc = self._proc
        assert proc is not None
        if reason is not None:
            self._stop(f"the {self.role} worker {reason}; the client stopped it")
            return
        try:
            exit_code: int | None = proc.wait(timeout=self._exit_grace_s)
        except subprocess.TimeoutExpired:
            self._stop(f"the {self.role} worker closed its stdout but kept running; the client stopped it")
            return
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=STDERR_CATCH_UP_S)  # let the stderr tail catch up
        closed = self._closing and exit_code == 0
        self._fail(
            f"the {self.role} worker was closed" if closed else f"the {self.role} worker exited with code {exit_code}",
            exit_code,
        )

    def _pump_stderr(self, stream: IO[bytes]) -> None:
        pending_line = b""
        read = getattr(stream, "read1", stream.read)
        try:
            while chunk := read(65536):
                with self._tail_lock:
                    self._tail.extend(chunk)
                    if len(self._tail) > self._tail_limit:
                        del self._tail[: len(self._tail) - self._tail_limit]
                *lines, pending_line = (pending_line + chunk).split(b"\n")
                for line in lines:
                    self._worker_log.debug("%s", line.decode("utf-8", errors="replace").rstrip("\r"))
        except (OSError, ValueError):
            pass
        finally:
            _close_quietly(stream)


def _start_failed(exc: WorkerCrashed) -> bool:
    """Whether a worker that exited before ``hello`` could not start: its env is missing or broken."""
    return exc.exit_code == EXIT_START_FAILED or (
        exc.exit_code == 1 and _NO_WORKER_PACKAGE.search(exc.stderr_tail) is not None
    )


def _close_quietly(stream: IO[bytes]) -> None:
    """Close a pipe from the one thread that uses it; a broken pipe is not news by now."""
    with contextlib.suppress(OSError, ValueError):
        stream.close()
