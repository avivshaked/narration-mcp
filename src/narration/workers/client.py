"""The daemon's side of one worker process (design Appendix A; plan.md WP16).

``SubprocessWorkerClient`` implements ``narration.contracts.interfaces.WorkerClient``:

- it starts the worker from an argument list (never a shell), with the environment ``launch.worker_env``
  builds, and exchanges ``hello`` (the role and protocol version must match);
- a reader thread correlates replies with requests by id, so requests may come from several threads;
- ``request`` returns the reply, or raises ``WorkerFailure`` for ``ok: false``, ``WorkerTimeout`` when no
  reply comes in time (the worker is then stopped), and ``WorkerCrashed`` when the worker exits or writes
  anything that is not a protocol message (with its exit code and the tail of its stderr);
- a crash is reported as soon as the worker's stdout closes, and, should a grandchild keep that pipe open,
  within a fraction of a second of the process exiting; it is never waited out to the request's timeout;
- ``close`` sends ``shutdown``, waits, and kills the worker if it does not exit in time.

OS-specific start-up (below-normal priority, the kill-on-close group) is not done here. The daemon passes
``on_spawn``, called with the new process's pid before ``hello``, and ``creationflags``; both come from
``narration.platform`` (WP19, WP30). There is a moment between the start and ``on_spawn`` when the worker
is outside the group; it is waiting for its first request then, and starts nothing.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Final, cast

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

from .launch import WorkerCommand

log = logging.getLogger(__name__)

SpawnHook = Callable[[int], None]
"""Called with a new worker's pid right after it starts (e.g. add it to a kill-on-close group)."""

POLL_S: Final = 0.25
"""How often a waiting request checks that the worker process is still running."""
DEFAULT_HELLO_TIMEOUT_S: Final = 120.0
DEFAULT_STDERR_TAIL_BYTES: Final = 64 * 1024


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
        self._write_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._next_id = 1
        self._failure: tuple[str, int | None] | None = None
        self._closing = False
        self._tail = bytearray()
        self._tail_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
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

        Raises ``WorkerFailure`` (``BACKEND_NOT_INSTALLED``) when the process cannot start or speaks another
        protocol or role, ``WorkerCrashed`` when it exits during start-up, ``WorkerTimeout`` when ``hello``
        gets no reply in time. On any failure the process is stopped.
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
                f"the {self.role} worker could not be started ({exc}); check that its venv is installed",
                {"role": self.role, "python": self._command.argv[0]},
            ) from exc
        proc = self._proc
        assert proc.stdout is not None and proc.stderr is not None
        self._threads = [
            threading.Thread(target=self._pump_stdout, args=(proc.stdout,), name=f"{self.role}-stdout", daemon=True),
            threading.Thread(target=self._pump_stderr, args=(proc.stderr,), name=f"{self.role}-stderr", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        try:
            if self._on_spawn is not None:
                self._on_spawn(proc.pid)
            reply = self.request("hello", {}, timeout_s=self._hello_timeout_s)
        except BaseException:
            self._stop(f"the {self.role} worker was stopped because it failed to start")
            self._release()
            raise
        if reply.get("role") != self.role or reply.get("protocol") != PROTOCOL_VERSION:
            self.close(timeout_s=5.0)
            raise WorkerFailure(
                "BACKEND_NOT_INSTALLED",
                f"the worker answered as role {reply.get('role')!r} with protocol {reply.get('protocol')!r}; this "
                f"server needs role {self.role!r} with protocol {PROTOCOL_VERSION}: re-sync the worker venv",
                {"role": reply.get("role"), "protocol": reply.get("protocol")},
            )
        self._hello = cast(HelloReply, reply)
        log.info("%s worker started (pid %d)", self.role, proc.pid)
        return self._hello

    def request(self, op: str, payload: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]:
        """Send one request and wait for its reply (see the module docstring for what it raises)."""
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
        try:
            with self._write_lock:
                assert proc.stdin is not None
                proc.stdin.write(data)
                proc.stdin.flush()
        except (OSError, ValueError) as exc:
            log.debug("writing to the %s worker failed (%s); waiting for it to report", self.role, exc)
        self._wait(pending, proc, op, rid, timeout_s)
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
    def _wait(self, pending: _Pending, proc: subprocess.Popen[bytes], op: str, rid: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
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
        """Close our ends of the pipes and let the reader threads finish."""
        proc = self._proc
        if proc is None:
            return
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()

    # ------------------------------------------------------------------ reader threads
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
        if len(self._threads) > 1:
            self._threads[1].join(timeout=2)  # let the stderr tail catch up
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
