"""The Windows ``Platform`` (design sections 4, 4.1, 17.2 and 17.3; WP19).

ctypes and the standard library only. Every kernel32 function is declared with its argument and result
types, so handles keep their 64 bits, and errors come from ``GetLastError`` (``use_last_error=True``). Import
this module through ``narration.platform.get_platform()``; on another OS it refuses to import.
"""

from __future__ import annotations

import sys

if sys.platform != "win32":  # the factory imports this module only on Windows
    raise ImportError("narration.platform._windows is for Windows only; use narration.platform.get_platform()")

import contextlib
import ctypes
import hashlib
import ntpath
import os
import stat
import subprocess
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path
from typing import Any, Final

import psutil

from narration.contracts import codes
from narration.contracts.errors import NarrationError

from . import (
    BREAKAWAY_REFUSED,
    BREAKAWAY_REFUSED_MESSAGE,
    DAEMON_RETRY_AFTER_S,
    JOB_CHECK_FAILED,
    JOB_CHECK_FAILED_MESSAGE,
    LEFT_IN_JOB,
    LEFT_IN_JOB_MESSAGE,
    real_path,
    winpaths,
)

# ---------------------------------------------------------------- constants (Windows SDK values)
CREATE_BREAKAWAY_FROM_JOB: Final = 0x01000000
DETACHED_PROCESS: Final = 0x00000008
CREATE_NEW_PROCESS_GROUP: Final = 0x00000200
DETACHED_CREATION_FLAGS: Final = CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
"""The creation flags of a detached daemon, exactly as section 4.1 lists them."""

CREATE_SUSPENDED: Final = 0x00000004
"""Added to ``DETACHED_CREATION_FLAGS``: the daemon is created suspended, and runs its first instruction only
once Windows has confirmed that it is in no Job Object (``WindowsPlatform.spawn_detached``, "Nested jobs")."""

BELOW_NORMAL_PRIORITY_CLASS: Final = 0x00004000
CREATE_NO_WINDOW: Final = 0x08000000

NO_CWD_EXE_SEARCH: Final = "NoDefaultCurrentDirectoryInExePath"
"""Set (to ``1``) for every process the service starts: ``cmd.exe`` then stops looking for a program in the
current folder before ``PATH`` (section 17). Importing ``qwen_tts`` imports ``sox``, which runs
``os.popen("sox -h")`` through ``cmd.exe`` (KNOW, WP20's reading of qwen-tts 0.1.1)."""

_JOB_OBJECT_LIMIT_BREAKAWAY_OK: Final = 0x00000800
_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK: Final = 0x00001000
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Final = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: Final = 9

_PROCESS_TERMINATE: Final = 0x0001
_PROCESS_SET_QUOTA: Final = 0x0100
_PROCESS_SET_INFORMATION: Final = 0x0200
_PROCESS_QUERY_LIMITED_INFORMATION: Final = 0x1000

_WAIT_OBJECT_0: Final = 0x00000000
_WAIT_ABANDONED: Final = 0x00000080
_WAIT_TIMEOUT: Final = 0x00000102

_ERROR_ACCESS_DENIED: Final = 5

_DRIVE_UNKNOWN: Final = 0
_DRIVE_NO_ROOT_DIR: Final = 1
_DRIVE_REMOTE: Final = 4

_IO_REPARSE_TAG_MOUNT_POINT: Final = 0xA0000003
_IO_REPARSE_TAG_SYMLINK: Final = 0xA000000C
_LINK_TAGS: Final = frozenset({_IO_REPARSE_TAG_MOUNT_POINT, _IO_REPARSE_TAG_SYMLINK})
"""Name-surrogate reparse points: symbolic links and junctions (mount points), which redirect a path."""

_MAX_LINK_HOPS: Final = 40
"""Links followed while resolving one path before it counts as a loop (POSIX's customary SYMLOOP_MAX)."""

SINGLETON_PREFIX: Final = "Global\\narration-mcp.daemon."
"""The daemon mutex's name before the store path's hash. Never change it: a daemon of an older version and
one of a newer version must still exclude each other on the same store."""

# DAEMON_RETRY_AFTER_S, BREAKAWAY_REFUSED, LEFT_IN_JOB and JOB_CHECK_FAILED (and their messages) are defined in
# narration.platform
# (OS-neutral values the front-end and narration-admin read too) and imported above.


# ---------------------------------------------------------------- kernel32
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _function(name: str, restype: Any, *argtypes: Any) -> Callable[..., Any]:
    function = getattr(_kernel32, name)
    function.restype = restype
    function.argtypes = list(argtypes)
    return function


_HANDLE = wintypes.HANDLE
_CreateMutexW = _function("CreateMutexW", _HANDLE, ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
_WaitForSingleObject = _function("WaitForSingleObject", wintypes.DWORD, _HANDLE, wintypes.DWORD)
_ReleaseMutex = _function("ReleaseMutex", wintypes.BOOL, _HANDLE)
_CloseHandle = _function("CloseHandle", wintypes.BOOL, _HANDLE)
_CreateJobObjectW = _function("CreateJobObjectW", _HANDLE, ctypes.c_void_p, wintypes.LPCWSTR)
_SetInformationJobObject = _function(
    "SetInformationJobObject", wintypes.BOOL, _HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
)
_QueryInformationJobObject = _function(
    "QueryInformationJobObject",
    wintypes.BOOL,
    _HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
)
_AssignProcessToJobObject = _function("AssignProcessToJobObject", wintypes.BOOL, _HANDLE, _HANDLE)
_IsProcessInJob = _function("IsProcessInJob", wintypes.BOOL, _HANDLE, _HANDLE, ctypes.POINTER(wintypes.BOOL))
_GetCurrentProcess = _function("GetCurrentProcess", _HANDLE)
_OpenProcess = _function("OpenProcess", _HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_SetPriorityClass = _function("SetPriorityClass", wintypes.BOOL, _HANDLE, wintypes.DWORD)
_GetDriveTypeW = _function("GetDriveTypeW", wintypes.UINT, wintypes.LPCWSTR)
_GetDiskFreeSpaceExW = _function(
    "GetDiskFreeSpaceExW",
    wintypes.BOOL,
    wintypes.LPCWSTR,
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong),
)


def _last_error() -> OSError:
    return ctypes.WinError(ctypes.get_last_error())


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    )
    LimitFlags: int


class _IoCounters(ctypes.Structure):
    _fields_ = tuple(
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    )


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )
    BasicLimitInformation: _BasicLimitInformation


# ---------------------------------------------------------------- the singleton (sections 4 and 4.1)
def singleton_name(store_root: Path) -> str:
    """The daemon mutex's name for a store: ``SINGLETON_PREFIX`` + sha256 hex of the normalised store path.

    The path is normalised as the file system sees it: ``realpath`` (links, junctions and ``subst`` drives
    resolved; a mapped drive of an existing path becomes its UNC form), then ``normcase`` (case and
    separators). Two spellings of one folder therefore give one name. The namespace is ``Global\\``, not
    ``Local\\``: the store is one folder on this machine, and a daemon started in another logon session (a
    second user, a remote desktop, an SSH session) must still be excluded (section 4: two sessions must never
    load two models).
    """
    normalised = os.path.normcase(os.path.realpath(store_root))
    return SINGLETON_PREFIX + hashlib.sha256(normalised.encode("utf-8", "surrogatepass")).hexdigest()


class _MutexHolder:
    """Owns a named mutex from a thread of its own until ``release`` is called.

    A Windows mutex belongs to the thread that acquired it, and it is recursive for that thread. Holding it
    from a dedicated thread makes ownership independent of the caller's threads: a second acquirer in the
    same process is refused just like one in another process, and ``release`` may run on any thread. If the
    process dies without releasing, Windows marks the mutex abandoned and the next acquirer gets it
    (``WAIT_ABANDONED``, recorded in ``abandoned``).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.abandoned = False
        self._release = threading.Event()
        self._thread: threading.Thread | None = None

    def acquire(self) -> bool:
        """Try once, without waiting; True if this holder now owns the mutex."""
        ready = threading.Event()
        outcome: list[bool | BaseException] = []

        def hold() -> None:
            handle = None
            try:
                handle = _CreateMutexW(None, False, self.name)
                if not handle:
                    error = ctypes.get_last_error()
                    # In the Global namespace, a mutex another user created is not ours to open: it is held.
                    outcome.append(False if error == _ERROR_ACCESS_DENIED else ctypes.WinError(error))
                    return
                result = _WaitForSingleObject(handle, 0)
                if result not in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
                    outcome.append(False if result == _WAIT_TIMEOUT else _last_error())
                    return
                self.abandoned = result == _WAIT_ABANDONED
                outcome.append(True)
                ready.set()
                self._release.wait()
                _ReleaseMutex(handle)
            except BaseException as exc:  # handed to the acquiring thread, never lost in this one
                outcome.append(exc)
            finally:
                if handle:
                    _CloseHandle(handle)
                ready.set()

        thread = threading.Thread(target=hold, name=f"narration-singleton-{self.name[-8:]}", daemon=True)
        thread.start()
        ready.wait()
        result = outcome[0]
        if result is True:
            self._thread = thread
            return True
        thread.join()
        if isinstance(result, BaseException):
            raise result
        return False

    def release(self) -> None:
        """Release the mutex if held; safe to call more than once, from any thread."""
        thread, self._thread = self._thread, None
        if thread is not None:
            self._release.set()
            thread.join()


# ---------------------------------------------------------------- Job Objects (sections 4 and 4.1)
class _JobObject:
    """A Job Object this process holds a handle to.

    With ``kill_on_close``, every process in it is terminated when the last handle closes: when ``close`` is
    called, or when this process dies, since Windows closes a dead process's handles. The handle is not
    inheritable, so no child can keep the job alive. ``allow_breakaway`` lets the job's processes start
    children outside it with ``CREATE_BREAKAWAY_FROM_JOB``; ``silent_breakaway`` keeps their children out of
    it whether they ask or not. The tests use both to stand in for the jobs of MCP clients (the Python SDK's
    has neither; libuv's has both, so Node.js's does, as measured, and, we believe, Claude Code's) and for a
    venv launcher's (silent only).
    """

    def __init__(
        self, *, kill_on_close: bool = True, allow_breakaway: bool = False, silent_breakaway: bool = False
    ) -> None:
        handle = _CreateJobObjectW(None, None)
        if not handle:
            raise _last_error()
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = (
            (_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE if kill_on_close else 0)
            | (_JOB_OBJECT_LIMIT_BREAKAWAY_OK if allow_breakaway else 0)
            | (_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK if silent_breakaway else 0)
        )
        if not _SetInformationJobObject(
            handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = _last_error()
            _CloseHandle(handle)
            raise error
        self._handle: object = handle
        self._lock = threading.Lock()

    def add(self, pid: int) -> None:
        """Assign the process ``pid`` to the job; raises ``OSError`` if Windows refuses (for example, no such
        process), and ``ValueError`` once the job is closed."""
        with self._lock:
            if self._handle is None:
                raise ValueError("this kill-on-close group is closed")
            process = _OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            if not process:
                raise _last_error()
            try:
                if not _AssignProcessToJobObject(self._handle, process):
                    raise _last_error()
            finally:
                _CloseHandle(process)

    def contains(self, pid: int) -> bool:
        """Whether process ``pid`` is in this job (``IsProcessInJob``); raises ``OSError`` if it cannot be asked."""
        with self._lock:
            if self._handle is None:
                raise ValueError("this kill-on-close group is closed")
            return _in_job(pid, self._handle)

    def close(self) -> None:
        """Close this process's handle; with ``kill_on_close``, the job's processes are then terminated."""
        with self._lock:
            handle, self._handle = self._handle, None
            if handle is not None:
                _CloseHandle(handle)


def _in_job(pid: int, job: object | None) -> bool:
    """Whether process ``pid`` is in the Job Object ``job``, or in any job when ``job`` is None. Raises
    ``OSError`` when Windows cannot answer (no such process, say)."""
    process = _OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not process:
        raise _last_error()
    try:
        in_job = wintypes.BOOL()
        if not _IsProcessInJob(process, job, ctypes.byref(in_job)):
            raise _last_error()
        return bool(in_job.value)
    finally:
        _CloseHandle(process)


def _resume(pid: int) -> None:
    """Let a process created with ``CREATE_SUSPENDED`` run (every thread's suspend count goes down by one).
    Raises ``OSError`` if it cannot be resumed."""
    try:
        psutil.Process(pid).resume()
    except psutil.Error as exc:
        raise OSError(f"cannot resume process {pid}: {exc}") from exc


def _own_job_breakaway() -> tuple[bool | None, bool | None]:
    """(whether this process is in a job, whether its innermost job allows breakaway); None where unknown."""
    in_job = wintypes.BOOL()
    if not _IsProcessInJob(_GetCurrentProcess(), None, ctypes.byref(in_job)):
        return None, None
    if not in_job.value:
        return False, None
    limits = _ExtendedLimitInformation()
    if not _QueryInformationJobObject(
        None, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits), None
    ):
        return True, None
    flags = limits.BasicLimitInformation.LimitFlags
    return True, bool(flags & (_JOB_OBJECT_LIMIT_BREAKAWAY_OK | _JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK))


class _DetachedPopen(subprocess.Popen[bytes]):
    """A ``Popen`` for a process this one never waits for: dropping it does not warn that the child runs.

    Its process handle is still closed when the object is freed.
    """

    def __del__(self) -> None:
        return None


def _end(process: subprocess.Popen[bytes]) -> None:
    """End a suspended child that must never run, and wait until it is gone (``kill`` is harmless if it has
    already gone)."""
    process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


# ---------------------------------------------------------------- paths (sections 17.2 and 17.3)
def _drive_type(drive: str) -> int:
    return int(_GetDriveTypeW(drive + "\\"))


def _check_drive(drive: str, path: str, what: str) -> None:
    kind = _drive_type(drive)
    if kind == _DRIVE_REMOTE:
        raise winpaths.path_error(
            "network", f"{what} is on a network drive ({drive}); only local drives are read", path
        )
    if kind in (_DRIVE_UNKNOWN, _DRIVE_NO_ROOT_DIR):
        raise winpaths.path_error("no_such_drive", f"{what} is on drive {drive}, which does not exist", path)


def _resolve_links(drive: str, names: Sequence[str], path: str) -> str:
    """Follow the path's symbolic links and junctions one name at a time, checking each target as text
    before anything opens it, so a link to a network share or a device never reaches it (section 17.3)."""
    current = drive + "\\"
    pending = list(names)
    hops = 0
    while pending:
        candidate = ntpath.join(current, pending.pop(0))
        try:
            info = os.lstat(candidate)
        except (FileNotFoundError, NotADirectoryError):
            raise winpaths.path_error("not_found", "there is no file at the path", path) from None
        except OSError as exc:
            raise winpaths.path_error("unreadable", f"the path cannot be read ({exc.strerror})", path) from exc
        if info.st_reparse_tag not in _LINK_TAGS:
            current = candidate
            continue
        hops += 1
        if hops > _MAX_LINK_HOPS:
            raise winpaths.path_error("link_loop", f"the path goes through more than {_MAX_LINK_HOPS} links", path)
        try:
            target = winpaths.strip_verbatim(os.readlink(candidate))
        except OSError as exc:
            raise winpaths.path_error("unreadable", f"the link {candidate} cannot be read", path) from exc
        # A relative target is relative to the link's folder; ".." is collapsed as Windows does.
        what = f"the target of the link {candidate}"
        target_drive, target_names = winpaths.parse_absolute(ntpath.normpath(ntpath.join(current, target)), what=what)
        _check_drive(target_drive, path, what)
        current, pending = target_drive + "\\", [*target_names, *pending]
    return current


# ---------------------------------------------------------------- the platform
class WindowsPlatform:
    """The ``Platform`` of Windows 10 and 11 (``narration.contracts.interfaces.Platform``). It holds no state."""

    @contextmanager
    def singleton(self, store_root: Path) -> Iterator[bool]:
        """Hold the daemon's singleton for ``store_root`` (section 4): a named mutex (``singleton_name``).

        Yields True if this call acquired it and False if another holder has it (in this process or any
        other, in any logon session). It is released when the block exits, or by Windows if the process
        dies. It never waits.
        """
        holder = _MutexHolder(singleton_name(store_root))
        acquired = holder.acquire()
        try:
            yield acquired
        finally:
            holder.release()

    def spawn_detached(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
        """Start ``argv`` detached (section 4.1), in no Job Object, and return its pid without waiting for it.

        The pid is that of ``argv[0]``. When that is a venv's ``Scripts\\python.exe`` (a launcher), the
        interpreter runs as the launcher's child with a pid of its own, so a daemon should record its own
        ``os.getpid()`` rather than rely on this value.

        Creation flags ``CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``, plus
        ``CREATE_SUSPENDED`` (below); stdin, stdout and stderr on ``NUL``; no other handle inherited
        (``close_fds``); ``env`` is the whole environment. Raises ``NarrationError(DAEMON_UNAVAILABLE)``
        (retryable; ``details["reason"]`` says which) and starts nothing that runs when the daemon cannot leave
        this process's Job Objects: a daemon still in a client's kill-on-close job would die with the client
        mid-job. Any other failure to start (a missing executable, say) propagates as ``OSError``. Its messages
        (``narration.platform``'s ``*_MESSAGE``) name no caller ("this process", not "this client"): an MCP
        client and ``narration-admin daemon start`` in a terminal both show them.

        **Nested jobs.** KNOW (spike k): since Windows 8 a process can be in nested jobs, and ``CreateProcess``
        with ``CREATE_BREAKAWAY_FROM_JOB`` refuses (access denied, ``BREAKAWAY_REFUSED``) only when this
        process's *innermost* job forbids breakaway. When that job allows it and an enclosing one does not,
        the call succeeds and the child is put in the enclosing job. That is the front-end's situation under
        the MCP Python SDK: its kill-on-close job forbids breakaway, but the venv's ``python.exe`` launcher
        puts the interpreter in a job of its own, nested inside it, that allows silent breakaway. So the daemon
        is created suspended, Windows is asked whether it is in any job (``IsProcessInJob``), and only a
        daemon in none is let run. One still in a job is ended before its first instruction
        (``LEFT_IN_JOB``); so is one whose membership cannot be read (``JOB_CHECK_FAILED``).

        **The rule is "in no Job Object at all"** (lead decision, WP30-escape review). Any enclosing job that
        forbids breakaway refuses the detached start, even one that would not end the daemon: this process
        cannot read an enclosing job's limits or know who holds its handle (the MCP Python SDK, for one,
        terminates its job whatever the job's flags). The known case (KNOW, PR #37's CI): GitHub's hosted
        Windows runner puts its steps in such a job. The supported route on such a host is ``[daemon]
        autostart = false``, so that no client tries to start the daemon, with a daemon started by hand in its
        own process (``narration-admin daemon start --foreground``); or a host whose jobs allow breakaway.

        **What is checked.** The process created, ``argv[0]``. Under a venv that is the launcher; its child,
        the interpreter that records its own pid in ``run/daemon.json``, is born into the launcher's own
        kill-on-close job (the launcher's, not the client's; the launcher holds its only handle while it waits
        for the interpreter). So the daemon runs in no job of the client's, and its launcher must never be
        ended by pid: the daemon would go with it.

        **No orphan, from the moment ``Popen`` returns.** From then on, whatever interrupts the check or the
        resume (an ``OSError``, Ctrl+C, any exception) ends the suspended child before it propagates. The
        guarantee starts when ``Popen`` returns, not at ``CreateProcess``: an exception raised inside ``Popen``
        after Windows has created the child could leave it suspended, never run and never ended.
        """
        if not argv:
            raise ValueError("argv is empty")
        try:
            process = _DetachedPopen(
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=DETACHED_CREATION_FLAGS | CREATE_SUSPENDED,
            )
        except OSError as exc:
            if exc.winerror != _ERROR_ACCESS_DENIED:
                raise
            in_job, allows_breakaway = _own_job_breakaway()
            raise NarrationError(
                codes.DAEMON_UNAVAILABLE,
                BREAKAWAY_REFUSED_MESSAGE,
                details={
                    "reason": BREAKAWAY_REFUSED,
                    "winerror": exc.winerror,
                    "in_job": in_job,
                    "job_allows_breakaway": allows_breakaway,
                },
                retry_after_s=DAEMON_RETRY_AFTER_S,
            ) from exc
        # The child is suspended and this Popen holds its handle, so its pid names it until it is resumed.
        # Whatever interrupts the check or the resume (an OSError, Ctrl+C in narration-admin's main thread, any
        # exception at all), the suspended child is ended first: none is ever left suspended and unreaped.
        try:
            try:
                still_in_job = _in_job(process.pid, None)
            except OSError as exc:
                in_job, allows_breakaway = _own_job_breakaway()
                raise NarrationError(
                    codes.DAEMON_UNAVAILABLE,
                    JOB_CHECK_FAILED_MESSAGE.format(error=exc.strerror or exc),
                    details={
                        "reason": JOB_CHECK_FAILED,
                        "winerror": exc.winerror,
                        "in_job": in_job,
                        "job_allows_breakaway": allows_breakaway,
                    },
                    retry_after_s=DAEMON_RETRY_AFTER_S,
                ) from exc
            if still_in_job:
                in_job, allows_breakaway = _own_job_breakaway()
                raise NarrationError(
                    codes.DAEMON_UNAVAILABLE,
                    LEFT_IN_JOB_MESSAGE,
                    details={
                        "reason": LEFT_IN_JOB,
                        "in_job": in_job,
                        "job_allows_breakaway": allows_breakaway,
                    },
                    retry_after_s=DAEMON_RETRY_AFTER_S,
                )
            _resume(process.pid)
        except BaseException:
            _end(process)
            raise
        return process.pid

    @contextmanager
    def kill_on_close_group(self) -> Iterator[Callable[[int], None]]:
        """A Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` (sections 4 and 4.1).

        Yields ``add(pid)``, which assigns a started process to the job. When the block exits, or when this
        process dies, Windows terminates every process in the job, including their own children.
        """
        job = _JobObject(kill_on_close=True)
        try:
            yield job.add
        finally:
            job.close()

    def set_below_normal_priority(self, pid: int) -> None:
        """Set ``BELOW_NORMAL_PRIORITY_CLASS`` on process ``pid`` and its descendants (section 4.1).

        The descendants matter because a venv's ``Scripts\\python.exe`` is a launcher: the interpreter that
        does the work is its child, and a priority class set on a running parent does not reach a child it
        already started. ``pid`` is set first, so a child it starts afterwards inherits the class; then every
        descendant present is set (psutil checks each one's identity, so a reused pid is never touched).
        Raises ``OSError`` if ``pid`` cannot be set; a descendant that has exited meanwhile is skipped.
        """
        process = _OpenProcess(_PROCESS_SET_INFORMATION, False, pid)
        if not process:
            raise _last_error()
        try:
            if not _SetPriorityClass(process, BELOW_NORMAL_PRIORITY_CLASS):
                raise _last_error()
        finally:
            _CloseHandle(process)
        try:
            descendants = psutil.Process(pid).children(recursive=True)
        except psutil.NoSuchProcess:
            return
        for child in descendants:
            try:
                child.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as exc:
                raise PermissionError(f"cannot set the priority of process {child.pid}, a child of {pid}") from exc

    # ---- how Python processes are started (ProcessPlatform; WP30)
    def python_for(self, python: Path, *, console: bool) -> Path:
        """``pythonw.exe`` beside ``python`` when ``console`` is False, ``python.exe`` beside it when True, if that
        file exists; else ``python``.

        A venv's ``python.exe`` is a launcher that starts the interpreter as a console program, and that
        interpreter, started detached, gets a console of its own (KNOW, spike g), which Windows may show as a
        window a user could close. ``pythonw.exe`` gets none (KNOW, spike g): the daemon runs as it. A worker
        speaks over its standard streams, which a windowless Python does not promise, so it runs as
        ``python.exe``.
        """
        name = python.name.lower()
        wanted = "python.exe" if console else "pythonw.exe"
        if name in ("python.exe", "pythonw.exe") and name != wanted and python.with_name(wanted).is_file():
            return python.with_name(wanted)
        return python

    def worker_creationflags(self, *, below_normal: bool) -> int:
        """``CREATE_NO_WINDOW``, plus ``BELOW_NORMAL_PRIORITY_CLASS`` when ``below_normal``.

        A detached daemon has no console, and a console program it starts without ``CREATE_NO_WINDOW`` gets a
        new console of its own, which Windows may show as a window. With it, the worker still gets a console
        (spike g saw its ``conhost.exe``), but one with no window. The priority class holds from the
        process's first instant, and a venv launcher's child (the interpreter) inherits it.
        """
        return CREATE_NO_WINDOW | (BELOW_NORMAL_PRIORITY_CLASS if below_normal else 0)

    def hardening_env(self) -> Mapping[str, str]:
        """``NoDefaultCurrentDirectoryInExePath=1`` (see ``NO_CWD_EXE_SEARCH``). ``PATH`` is left alone: it is the
        operator's, and a scrubbed ``PATH`` can break DLL loading."""
        return {NO_CWD_EXE_SEARCH: "1"}

    def check_readable_path(self, path: str) -> Path:
        r"""Check a caller's path to a file the service will read (section 17.3); return it resolved.

        Accepted: an absolute path on a local drive (fixed, removable, optical or RAM disk) that resolves,
        through any links and junctions, to a regular file. Refused with ``PATH_NOT_ALLOWED`` (``details``:
        ``path`` and ``rule``): a relative path; a network path (``\\server\share``, a mapped network drive);
        a device or namespace path (``\\.\``, ``\\?\``) or a reserved device name (``CON``, ``nul.txt``,
        ``COM1``); a link whose target is any of those; a missing file; a folder. Everything that could reach
        the network is refused as text, before any file is opened.

        The returned path has no links left in it. The check and a later open are not atomic; the caller
        copies the file into the store at once (section 17.3).
        """
        drive, names = winpaths.parse_absolute(path)
        _check_drive(drive, path, "the path")
        resolved = _resolve_links(drive, names, path)
        try:
            info = os.stat(resolved)
        except (FileNotFoundError, NotADirectoryError):
            raise winpaths.path_error("not_found", "there is no file at the path", path) from None
        except OSError as exc:
            raise winpaths.path_error("unreadable", f"the path cannot be read ({exc.strerror})", path) from exc
        if not stat.S_ISREG(info.st_mode):
            raise winpaths.path_error("not_regular_file", "the path is not a regular file (a folder?)", path)
        return Path(resolved)

    def check_store_path(self, path: Path, root: Path) -> Path:
        """Check a path the store will write (section 17.2); return its ``realpath``.

        Refused with ``PATH_NOT_ALLOWED``: a relative path; a path outside ``root`` (``..`` included); a name
        below the root that is a reserved device name, has a colon (a stream), a character Windows does not
        allow, or a trailing dot or space; an existing file or folder below the root that is a reparse point
        (a link, a junction, or any other kind); and a ``realpath`` that leaves the root. The names that do
        not exist yet are only checked as text. The root itself is the operator's choice and may be reached
        through a link.
        """
        raw = str(path)
        root_path = os.path.abspath(root)
        below = winpaths.split_under_root(raw, root_path)
        current = ntpath.normpath(root_path)
        for name in below:
            current = ntpath.join(current, name)
            try:
                info = os.lstat(current)
            except (FileNotFoundError, NotADirectoryError):
                break
            if info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise winpaths.path_error(
                    "reparse_point",
                    f"{name!r} in the store is a link, a junction or another reparse point",
                    raw,
                    hint=winpaths.STORE_PATH_HINT,
                )
        real = real_path(raw)  # both sides without a \\?\ prefix: see real_path
        if winpaths.relative_names(real, real_path(root_path)) is None:
            raise winpaths.path_error(
                "outside_root", "the path resolves outside the store root", raw, hint=winpaths.STORE_PATH_HINT
            )
        return Path(real)

    def free_disk_bytes(self, path: Path) -> int:
        """Bytes free to this user on the volume holding ``path`` (quotas included; ``GetDiskFreeSpaceExW``).

        ``path`` need not exist yet: its nearest existing folder is measured.
        """
        probe = Path(os.path.abspath(path))
        while not probe.is_dir():
            if probe.parent == probe:
                raise FileNotFoundError(f"no existing folder above {path}")
            probe = probe.parent
        available = ctypes.c_ulonglong()
        if not _GetDiskFreeSpaceExW(str(probe), ctypes.byref(available), None, None):
            raise _last_error()
        return int(available.value)
