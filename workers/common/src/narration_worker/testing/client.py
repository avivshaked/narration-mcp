"""A small synchronous client for a worker process, for tests (standard library only).

It keeps every line the worker writes to stdout, so a test can check that stdout carried protocol messages
and nothing else. Every wait has a timeout, and ``stop()`` (also on leaving a ``with`` block) always kills and
reaps the process, so a test never leaves a worker behind, even one that hangs.
"""

from __future__ import annotations

import contextlib
import os
import queue
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from narration_worker.framing import FramingError, decode_message, encode_message, is_request_id


class WorkerProcess:
    """One worker process under test."""

    def __init__(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: Path | None = None) -> None:
        self.argv = list(argv)
        self.env = dict(env) if env is not None else None
        self.cwd = cwd
        self.lines: list[bytes] = []
        """Every line the worker wrote to stdout, in order, without newlines."""
        self._proc: subprocess.Popen[bytes] | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()  # None: end of stdout
        self._stderr = bytearray()
        self._threads: list[threading.Thread] = []
        self._next_id = 1

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> WorkerProcess:
        self._proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
        )
        assert self._proc.stdout is not None and self._proc.stderr is not None
        self._threads = [
            threading.Thread(target=self._read_stdout, args=(self._proc.stdout,), daemon=True),
            threading.Thread(target=self._read_stderr, args=(self._proc.stderr,), daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        return self

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def proc(self) -> subprocess.Popen[bytes]:
        if self._proc is None:
            raise RuntimeError("the worker has not been started")
        return self._proc

    def stop(self) -> int | None:
        """Kill the worker if it still runs, reap it, and close its pipes; returns its exit code."""
        if self._proc is None:
            return None
        proc = self._proc
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - a kill that does not land
            proc.kill()
            proc.wait(timeout=30)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        for thread in self._threads:
            thread.join(timeout=5)
        return proc.returncode

    def __enter__(self) -> WorkerProcess:
        return self.start() if self._proc is None else self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------ talking
    def send(self, message: Mapping[str, Any]) -> None:
        """Send one message."""
        self.send_raw(encode_message(message))

    def send_raw(self, data: bytes) -> None:
        """Send bytes as they are (for malformed-input tests)."""
        stdin = self.proc.stdin
        assert stdin is not None
        stdin.write(data)
        stdin.flush()

    def close_stdin(self) -> None:
        """Close the worker's stdin (its end of input)."""
        stdin = self.proc.stdin
        if stdin is not None:
            stdin.close()

    def receive(self, timeout_s: float = 60.0) -> dict[str, Any]:
        """The next reply; fails the test on a timeout, a non-protocol line, or the worker exiting."""
        try:
            line = self._queue.get(timeout=timeout_s)
        except queue.Empty:
            raise AssertionError(f"no reply within {timeout_s} s; stderr:\n{self.stderr_text()[-4000:]}") from None
        if line is None:
            self._queue.put(None)
            code = self.wait(10)
            raise AssertionError(f"the worker closed stdout (exit code {code}); stderr:\n{self.stderr_text()[-4000:]}")
        return check_reply(line)

    def request(self, op: str, *, timeout_s: float = 60.0, **fields: Any) -> dict[str, Any]:
        """Send a request with the next id and return its reply (whose id must match)."""
        rid = self._next_id
        self._next_id += 1
        self.send({"id": rid, "op": op, **fields})
        reply = self.receive(timeout_s)
        assert reply["id"] == rid, f"reply id {reply['id']} for request id {rid}"
        return reply

    def wait(self, timeout_s: float) -> int:
        """Wait for the worker to exit, and for its last stdout lines to be read; returns its exit code.

        Raises ``subprocess.TimeoutExpired`` if it does not exit in time.
        """
        code = self.proc.wait(timeout=timeout_s)
        if self._threads:
            self._threads[0].join(timeout=10)
        return code

    def stderr_text(self) -> str:
        return self._stderr.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ readers
    def _read_stdout(self, stream: IO[bytes]) -> None:
        try:
            for line in iter(stream.readline, b""):
                stripped = line[:-1] if line.endswith(b"\n") else line
                self.lines.append(stripped)
                self._queue.put(stripped)
        except (OSError, ValueError):
            pass
        finally:
            self._queue.put(None)

    def _read_stderr(self, stream: IO[bytes]) -> None:
        try:
            for chunk in iter(lambda: os.read(stream.fileno(), 65536), b""):
                self._stderr.extend(chunk)
                if len(self._stderr) > 1 << 20:
                    del self._stderr[: len(self._stderr) - (1 << 20)]
        except (OSError, ValueError):
            pass


def check_reply(line: bytes) -> dict[str, Any]:
    """Decode a stdout line and check that it is a protocol reply; fails the test otherwise."""
    try:
        message = decode_message(line)
    except FramingError as exc:
        raise AssertionError(f"stdout carried a line that is not a protocol message ({exc}): {line[:300]!r}") from None
    rid = message.get("id")
    assert rid is None or is_request_id(rid), f"reply id {rid!r} is neither an integer nor null"
    assert isinstance(message.get("ok"), bool), f"reply has no boolean ok: {message}"
    if message["ok"] is False:
        error = message.get("error")
        assert isinstance(error, dict) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str)
    return message
