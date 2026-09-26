"""Keep the worker's stdin and stdout for the protocol alone (design Appendix A).

Libraries print: a warning, a progress bar, a native library's ``printf``, a subprocess that inherits the
worker's stdout. Any of them would corrupt the JSON-lines stream. So at start-up the worker takes private
copies of file descriptors 0 and 1 for the protocol, then

- points file descriptor 1, ``sys.stdout`` and ``sys.__stdout__``'s descriptor at stderr, so Python
  ``print``, native writes to descriptor 1, and child processes that inherit stdout all land in the log;
- points file descriptor 0 and ``sys.stdin`` at the null device, so nothing but the request loop can read
  a request (a library that prompts reads end-of-input instead of eating a protocol line).

The private copies are not inheritable (PEP 446), so child processes never receive the protocol pipes.
On Windows the C runtime's ``dup2`` onto descriptors 0-2 also updates the process's standard handles in a
console program such as ``python.exe`` (BELIEVE; the fake's ``stdout_noise`` fault tests it through a
child process, which takes its stdout from that handle).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import IO


@dataclass(frozen=True, slots=True)
class ProtocolStreams:
    """The private protocol streams: requests in, replies out (both binary)."""

    reader: IO[bytes]
    writer: IO[bytes]


def claim_stdio() -> ProtocolStreams:
    """Take stdin and stdout for the protocol and redirect everything else (call once, at start-up)."""
    sys.stdout.flush()
    sys.stderr.flush()
    reader = os.fdopen(os.dup(0), "rb")
    writer = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    null = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(null, 0)
    finally:
        os.close(null)
    sys.stdout = sys.stderr
    sys.stdin = open(os.devnull, encoding="utf-8")  # noqa: SIM115 - replaces the process's stdin for its lifetime
    return ProtocolStreams(reader=reader, writer=writer)
