"""The daemon's control loop (design sections 4, 4.1 and 7.6; plan.md WP30).

``Daemon.run`` does, in order:

1. takes the store's singleton (``Platform.singleton``). If another daemon holds it and says it serves
   (``idle`` or ``busy``), this one exits quietly and writes nothing. Otherwise (the holder says
   ``stopping``, or already ``stopped`` while it has yet to release the singleton, or there is no status,
   or it cannot be read) this one waits up to ``takeover_wait_s`` for the singleton, and takes over, so a
   job queued while a daemon was exiting is not left waiting. It gives up as soon as the holder says
   ``idle`` or ``busy`` (it found work as it was about to exit), and at ``takeover_wait_s``;
2. records its own pid and start time, sweeps up after the previous daemon (``sweep``), looks for a stop
   posted since its launch that another daemon has already answered, and reads the pending commands once
   (see "Which stops a daemon honours"). A stop it honours stops it before any work, and it never says
   ``idle``; otherwise it writes ``run/daemon.json`` (``idle``);
3. opens the worker supervisor (its kill-on-close group) and starts the runner thread, which drives the
   ``JobRunner`` one ``step`` at a time, unloads idle models after ``idle_unload_s``, and asks to exit
   after ``idle_exit_s`` with nothing to do. It says ``stopping`` first, then asks the runner ``has_work``
   once more, and stays if it has (``seam``, "The idle exit");
4. on the main thread, answers commands from the store every ``poll_s``: ``release_gpu`` at once,
   ``stop`` by letting the step in flight finish, ``stop_now`` by killing every worker (the Job Object) so
   the step in flight fails and its job goes back to the queue;
5. when the runner thread has ended (it shut the runner down, which gave back its job), closes the
   workers, completes the stop commands, writes ``stopped``, and releases the singleton.

Every file the store publishes is written to a temp name and renamed, so a ``stop_now`` never leaves a
partial file published (section 4.1).

**Which stops a daemon honours** (lead's decisions, WP30 review and re-review). A ``stop`` is for the
service, not for one daemon process, but only for the service as it was when the stop was asked. A daemon
honours every ``stop`` or ``stop_now`` posted after it was launched (``launched_at``), pending or already
answered:

- a pending one, whenever it reads it: even one posted while it waited for the singleton, which the daemon
  it took over from never read;
- one the daemon it took over from answered ``stopped: true`` while this one waited (that daemon was
  exiting anyway). On taking the singleton, before any work, it lists the commands posted since its launch
  (``Store.commands_since``); if one of them is a ``stop`` or ``stop_now`` answered ``stopped: true``, it
  stops as a ``stop`` would, having done nothing, so the answer the other daemon gave holds for the
  service. A stop answered ``stopped: false`` (one that was stale for the daemon that read it) does not
  stop it (lead's ruling, WP30 third review).

It completes a pending stop posted before it was launched with ``stopped: false`` and keeps serving, and it
ignores an answered one: that stop was asked of a daemon that has gone, or of none. Both times are read
from the wall clock (``time.time``), as ``started_at`` is; a command's ``requested_at`` is cut to the
millisecond, so a stop posted in the millisecond of the launch counts as posted after it
(``posted_after_launch``).

**A stamp from the future.** After the wall clock steps back, a stop can carry a ``requested_at`` later
than now. An answered one is not trusted, and is logged and skipped: it is never consumed, so honouring it
would stop every new daemon at once until the clock caught up. A pending one is honoured, as any stop posted
after the launch is: the daemon answers it, so it stops one daemon at most.

So ``narration-admin daemon stop`` (WP37) posts a stop only when ``start.running_daemon`` says a daemon
runs: a stop posted with none running stops nothing, and is answered ``stopped: false`` by the next daemon.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Literal

from narration.config import Config
from narration.contracts.interfaces import Store
from narration.contracts.models import JobRecord
from narration.contracts.names import JobPhase
from narration.platform import ProcessPlatform
from narration.store.store import parse_iso, utc_iso

from .seam import DAEMON_HOLDER, GpuFacts, JobRunner, ShutdownReason, StopMode, return_job
from .settings import DaemonSettings
from .status import StatusBoard
from .supervisor import FAKE_ROLES, WorkerSupervisor
from .sweep import StatusUnreadable, read_status, sweep

log = logging.getLogger(__name__)

EXIT_OK: Final = 0
"""Ran and stopped, or another daemon runs for this store (a second daemon exits quietly)."""
EXIT_ERROR: Final = 1
"""An unexpected error; the daemon's log has it."""
EXIT_USAGE: Final = 2
"""Bad arguments or configuration."""

MAX_STEP_BACKOFF_S: Final = 30.0
"""The longest wait after a step that raised, before the next step."""

SERVING_STATES: Final = ("idle", "busy")
"""The states of a daemon that serves the queue; a daemon waiting for the singleton gives up on seeing one."""

STAMP_PRECISION_S: Final = 0.001
"""Store times (``requested_at``, ``started_at``) are written to the millisecond, cut short (``utc_iso``)."""

STALE_STOP_REASON: Final = "it was posted before this daemon was launched, and no daemon running then read it"
"""The ``reason`` of a stop a daemon does not honour (see "Which stops a daemon honours")."""


def posted_after_launch(requested_at: str, launched_at: float) -> bool:
    """Whether a command posted at ``requested_at`` (a store time, cut to the millisecond) may have been posted
    after ``launched_at`` (Unix seconds, the same wall clock). A command posted in the millisecond of the launch
    counts as after it, so a stop asked of this daemon is never taken for an older one."""
    return parse_iso(requested_at) + STAMP_PRECISION_S > launched_at


SupervisorFactory = Callable[[Callable[[], None]], WorkerSupervisor]
"""Makes the daemon's supervisor, given the callback it reports worker changes to."""

_Stopping = Literal["segment", "now", "idle"]


class _Host:
    """The daemon as the runner sees it (``seam.RunnerHost``)."""

    def __init__(self, daemon: Daemon, supervisor: WorkerSupervisor) -> None:
        self._daemon = daemon
        self._supervisor = supervisor

    @property
    def store(self) -> Store:
        return self._daemon.store

    @property
    def config(self) -> Config:
        return self._daemon.config

    @property
    def workers(self) -> WorkerSupervisor:
        return self._supervisor

    @property
    def platform(self) -> ProcessPlatform:
        return self._daemon.platform

    @property
    def holder(self) -> str:
        return DAEMON_HOLDER

    @property
    def stop_mode(self) -> StopMode | None:
        mode = self._daemon.stopping
        return "now" if mode == "now" else "segment" if mode is not None else None

    def should_stop(self) -> bool:
        return self._daemon.stopping is not None

    def sleep(self, seconds: float) -> bool:
        return not self._daemon.stop_event.wait(max(0.0, seconds))

    def job_started(self, job: JobRecord) -> None:
        self._daemon.board.job_started(job)

    def job_phase(self, phase: JobPhase | None) -> None:
        self._daemon.board.job_phase(phase)

    def job_finished(self) -> None:
        self._daemon.board.job_finished()

    def set_gpu_facts(self, facts: GpuFacts) -> None:
        self._daemon.board.set_gpu_facts(facts)

    def set_est_drain(self, seconds: float | None) -> None:
        self._daemon.board.set_est_drain(seconds)


class Daemon:
    """One daemon process's work (see the module docstring). ``run`` returns the process's exit code.

    ``supervisor_factory`` makes the worker supervisor (by default a ``WorkerSupervisor`` over this
    platform, with the fake role for every group when ``settings.fake_workers``). ``pid`` defaults to
    ``os.getpid()``. ``clock`` measures intervals and ``wall`` gives the times written to the store.
    """

    def __init__(
        self,
        *,
        settings: DaemonSettings,
        config: Config,
        store: Store,
        platform: ProcessPlatform,
        runner: JobRunner,
        supervisor_factory: SupervisorFactory | None = None,
        pid: int | None = None,
        launched_at: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.launched_at = wall() if launched_at is None else launched_at
        """When this daemon was launched (Unix seconds, ``wall``): stops posted before it are not for it."""
        self.config = config
        self.store = store
        self.platform = platform
        self._runner = runner
        self._supervisor_factory = supervisor_factory or self._default_supervisor
        self._pid = os.getpid() if pid is None else pid
        self._clock = clock
        self._wall = wall
        self.stop_event = threading.Event()
        self._wake = threading.Event()
        self._state_lock = threading.Lock()
        self._work_lock = threading.Lock()
        self._stopping: _Stopping | None = None
        self._now_deadline: float | None = None
        self._last_activity = clock()
        self._unload_reported = False
        self._runner_done = threading.Event()
        self._stop_commands: list[str] = []
        self._requeued: list[str] = []
        self._board: StatusBoard | None = None
        self._supervisor: WorkerSupervisor | None = None
        self.started_at: str | None = None

    # ------------------------------------------------------------------ what other threads read
    @property
    def stopping(self) -> _Stopping | None:
        """None while running; why the daemon is stopping once it is."""
        with self._state_lock:
            return self._stopping

    @property
    def board(self) -> StatusBoard:
        if self._board is None:
            raise RuntimeError("the daemon has not started")
        return self._board

    # ------------------------------------------------------------------ run
    def run(self) -> int:
        """Take the singleton and serve until stopped; the process's exit code."""
        root = Path(os.path.abspath(self.settings.store_root))
        root.mkdir(parents=True, exist_ok=True)  # the singleton's name hashes the root's realpath
        deadline = self._clock() + self.settings.takeover_wait_s
        waited = False
        unreadable_logged = False
        while True:
            with self.platform.singleton(root) as acquired:
                if acquired:
                    if waited:
                        log.info("the daemon that held the singleton has gone; taking over")
                    return self._run_held()
            try:
                holder = read_status(self.store)
                seen = holder.state if holder is not None else "no status"
            except StatusUnreadable as exc:
                if not unreadable_logged:
                    unreadable_logged = True
                    log.warning("%s; waiting for the daemon that holds the singleton to go", exc)
                holder, seen = None, "an unreadable status"
            if holder is not None and holder.state in SERVING_STATES:
                log.info("another daemon serves this store (pid %s, %s); exiting", holder.pid, holder.state)
                return EXIT_OK
            if self._clock() >= deadline:
                log.info(
                    "the daemon that holds the singleton (%s) did not go within %.0f s; exiting",
                    seen,
                    self.settings.takeover_wait_s,
                )
                return EXIT_OK
            waited = True
            time.sleep(min(0.2, self.settings.poll_s))

    def _run_held(self) -> int:
        self.started_at = utc_iso(self._wall())
        unreadable = False
        try:
            previous = read_status(self.store)
        except StatusUnreadable as exc:
            log.warning("%s; the daemon before this one is taken to have died", exc)
            previous, unreadable = None, True
        self._board = StatusBoard(self.store, pid=self._pid, started_at=self.started_at, wall=self._wall)
        report = sweep(self.store, previous, started_at=self.started_at, previous_unreadable=unreadable)
        log.info(
            "daemon started (pid %d, store %s; previous daemon: %s)",
            self._pid,
            self.settings.store_root,
            report.previous,
        )
        self._honour_answered_stops()  # a stop answered by the daemon before this one (see the docstring)
        self._handle_commands()  # a stop already pending is honoured before any work (see the docstring)
        if self.stopping is None:  # a daemon that is to stop at once never says it serves
            self._board.set_state("idle")
        code = EXIT_OK
        try:
            with self._supervisor_factory(self._workers_changed) as supervisor:
                self._supervisor = supervisor
                host = _Host(self, supervisor)
                thread = threading.Thread(target=self._runner_main, args=(host,), name="narration-runner", daemon=True)
                thread.start()
                try:
                    self._control_loop()
                except KeyboardInterrupt:
                    log.warning("interrupted; stopping now")
                    self._request_stop("now")
                except BaseException:
                    log.exception("the daemon's control loop failed; stopping now")
                    self._request_stop("now")
                    code = EXIT_ERROR
                self._finish(thread)
        finally:
            self._complete_stop_commands()
            self._board.stopped()
            log.info("daemon stopped (pid %d)", self._pid)
        return code

    # ------------------------------------------------------------------ the main thread
    def _control_loop(self) -> None:
        while True:
            self._handle_commands()
            supervisor = self._supervisor
            if supervisor is not None:
                supervisor.poll()
            self.board.flush()
            if self._runner_done.is_set():
                return
            if self._now_deadline is not None and self._clock() >= self._now_deadline:
                log.warning(
                    "the job runner did not give its job back within %.0f s of stop_now; stopping without it",
                    self.settings.stop_now_grace_s,
                )
                return
            self._wake.wait(self.settings.poll_s)
            self._wake.clear()

    def _handle_commands(self) -> None:
        try:
            pending = self.store.pending_commands()
        except Exception:
            log.exception("could not read the daemon's commands")
            return
        for command in pending:
            if command.command_id in self._stop_commands:
                continue
            if command.kind == "release_gpu":
                result = self._release_gpu()
                try:
                    self.store.complete_command(command.command_id, result)
                except Exception:
                    log.exception("could not complete command %s", command.command_id)
                else:
                    log.info("release_gpu: %s", result)
            elif command.kind in ("stop", "stop_now"):
                stale = not posted_after_launch(command.requested_at, self.launched_at)
                if stale and self.stopping is None:
                    self._complete_stale_stop(command.command_id, command.kind)
                    continue
                self._stop_commands.append(command.command_id)  # completed as stopped, with the others
                if stale:  # already stopping: an older stop_now does not hurry it
                    continue
                log.info("%s requested", command.kind)
                self._request_stop("now" if command.kind == "stop_now" else "segment")
            else:  # pragma: no cover - the store refuses other kinds
                log.error("unknown daemon command %r", command.kind)

    def _honour_answered_stops(self) -> None:
        """Before any work: stop if a stop posted after this daemon's launch was already answered
        ``stopped: true`` by another daemon (see "Which stops a daemon honours")."""
        try:
            since = self.store.commands_since(utc_iso(self.launched_at - STAMP_PRECISION_S))
        except Exception:
            log.exception("could not list the commands posted since this daemon was launched")
            return
        now = self._wall()
        skipped_future = False
        for command in since:
            if (
                command.kind in ("stop", "stop_now")
                and command.done_at is not None
                and command.result is not None
                and command.result.get("stopped") is True
                and posted_after_launch(command.requested_at, self.launched_at)
            ):
                if parse_iso(command.requested_at) > now + STAMP_PRECISION_S:
                    if not skipped_future:
                        skipped_future = True
                        log.info(
                            "%s %s says it was posted at %s, later than now (the clock stepped back?); not "
                            "trusted, so it does not stop this daemon",
                            command.kind,
                            command.command_id,
                            command.requested_at,
                        )
                    continue
                log.info(
                    "%s %s was posted after this daemon was launched and answered by the daemon before it; "
                    "stopping before any work",
                    command.kind,
                    command.command_id,
                )
                self._request_stop("segment")  # nothing is in flight, so a stop_now needs nothing more
                return

    def _complete_stale_stop(self, command_id: str, kind: str) -> None:
        """Answer a stop posted before this daemon was launched, and keep serving."""
        try:
            self.store.complete_command(command_id, {"stopped": False, "reason": STALE_STOP_REASON})
        except Exception:
            log.exception("could not complete command %s", command_id)
        else:
            log.info("%s %s was posted before this daemon was launched; not honoured", kind, command_id)

    def _release_gpu(self) -> dict[str, Any]:
        """``release_gpu`` (section 7.6): unload an idle model at once; while a job runs, change nothing and
        name the job."""
        supervisor = self._supervisor
        holder = supervisor.gpu_holder if supervisor is not None else None
        current = self.board.current_job
        if current is not None or supervisor is None:
            return {"released": False, "holder_before": holder, "busy_job": current.job_id if current else None}
        if not self._work_lock.acquire(timeout=self.settings.command_wait_s):
            current = self.board.current_job
            return {"released": False, "holder_before": holder, "busy_job": current.job_id if current else None}
        try:
            current = self.board.current_job
            if current is not None:
                return {"released": False, "holder_before": supervisor.gpu_holder, "busy_job": current.job_id}
            holder = supervisor.release_all()
            self._unload_reported = False
            self.board.set_unload_at(None)
            return {"released": holder is not None, "holder_before": holder, "busy_job": None}
        finally:
            self._work_lock.release()

    def _request_stop(self, mode: _Stopping) -> None:
        with self._state_lock:
            current = self._stopping
            if current == "now" or current == mode or (mode == "idle" and current is not None):
                return
            self._stopping = mode
        self.stop_event.set()
        self._wake.set()
        self.board.set_state("stopping")
        if mode == "now":
            self._now_deadline = self._clock() + self.settings.stop_now_grace_s
            supervisor = self._supervisor
            if supervisor is not None:
                supervisor.kill_all()

    def _finish(self, thread: threading.Thread) -> None:
        """After the control loop: wait for the runner thread (a ``stop`` lets its step finish), give back
        any job it still holds, answer the last commands, close the workers."""
        while not self._runner_done.wait(self.settings.poll_s):
            if self._now_deadline is not None and self._clock() >= self._now_deadline:
                break
            self._handle_commands()  # a stop_now may escalate a stop; release_gpu is answered as busy
        if not self._runner_done.is_set():
            self._give_back("stop_now: the job runner did not stop in time")
        self._handle_commands()
        supervisor = self._supervisor
        if supervisor is not None:
            supervisor.close()
        if thread.is_alive():
            log.warning("the runner thread still runs; the daemon exits without it")

    def _complete_stop_commands(self) -> None:
        try:
            pending = {c.command_id: c for c in self.store.pending_commands()}
        except Exception:
            log.exception("could not read the daemon's commands")
            return
        for command in pending.values():
            if command.kind not in ("stop", "stop_now"):
                continue
            try:
                self.store.complete_command(command.command_id, {"stopped": True, "requeued": list(self._requeued)})
            except Exception:
                log.exception("could not complete command %s", command.command_id)

    # ------------------------------------------------------------------ the runner thread
    def _runner_main(self, host: _Host) -> None:
        backoff = 0.0
        try:
            while not self.stop_event.is_set():
                self._idle_unload()
                if self.stop_event.is_set():
                    break
                try:
                    with self._work_lock:
                        if self.stop_event.is_set():
                            break
                        did = self._runner.step(host)
                except Exception:
                    if self.stop_event.is_set():
                        log.info("the step in flight ended with the stop", exc_info=True)
                        break
                    backoff = min(max(2 * backoff, self.settings.poll_s), MAX_STEP_BACKOFF_S)
                    log.exception("the job runner's step failed (a bug); the next step comes in %.1f s", backoff)
                    self.stop_event.wait(backoff)
                    continue
                backoff = 0.0
                if did or self.board.current_job is not None:
                    self._busy()
                    continue
                if self._idle(host):
                    break
        finally:
            reason: ShutdownReason = self.stopping or "idle"
            held = self.board.current_job
            try:
                self._runner.shutdown(host, reason)
            except Exception:
                log.exception("the job runner's shutdown failed")
            self._give_back(f"the daemon stopped ({reason})")
            if held is not None:
                self._note_if_requeued(held.job_id)
            self._runner_done.set()
            self._wake.set()

    def _busy(self) -> None:
        self._last_activity = self._clock()
        if self._unload_reported:
            self._unload_reported = False
            self.board.set_unload_at(None)

    def _idle(self, host: _Host) -> bool:
        """Nothing to do: report when idle models will be unloaded, and decide the idle exit. True to exit."""
        idle_for = self._clock() - self._last_activity
        supervisor = self._supervisor
        if not self._unload_reported and supervisor is not None and supervisor.gpu_holder is not None:
            self._unload_reported = True
            self.board.set_unload_at(self._wall() + max(0.0, self.settings.idle_unload_s - idle_for))
        if idle_for >= self.settings.idle_exit_s and self._idle_exit(host, idle_for):
            return True
        waits = [self.settings.poll_s, self.settings.idle_exit_s - idle_for]
        if supervisor is not None and supervisor.loaded():
            waits.append(self.settings.idle_unload_s - idle_for)
        self.stop_event.wait(max(0.001, min(waits)))
        return False

    def _idle_exit(self, host: _Host, idle_for: float) -> bool:
        """Exit for want of work, without stranding a job queued meanwhile (``seam``, "The idle exit"): say
        ``stopping`` first, then ask the runner once more. True to exit; False when work turned up."""
        with self._state_lock:
            if self._stopping is not None:
                return True
        self.board.set_state("stopping")
        try:
            work = self._runner.has_work(host)
        except Exception:
            log.exception("the job runner's has_work failed (a bug); the daemon keeps serving")
            work = True
        if work:
            with self._state_lock:
                if self._stopping is None:  # a stop asked meanwhile keeps "stopping"
                    self.board.set_state("idle")
            self._last_activity = self._clock()
            log.info("idle for %.1f s, but work arrived as the daemon was about to exit; it keeps serving", idle_for)
            return False
        log.info("idle for %.1f s; exiting", idle_for)
        self._request_stop("idle")
        return True

    def _idle_unload(self) -> None:
        """Between steps: stop the workers once no job has used them for ``idle_unload_s`` (section 4)."""
        supervisor = self._supervisor
        if supervisor is None or not supervisor.loaded() or self.board.current_job is not None:
            return
        if self._clock() - self._last_activity < self.settings.idle_unload_s:
            return
        with self._work_lock:
            holder = supervisor.release_all()
        log.info("idle for %.0f s; unloaded the models (%s was on the GPU)", self.settings.idle_unload_s, holder)
        self._unload_reported = False
        self.board.set_unload_at(None)

    def _give_back(self, reason: str) -> None:
        """The safety net after the runner's shutdown: re-queue a job it still announces as held."""
        current = self.board.current_job
        if current is None:
            return
        try:
            return_job(self.store, current.job_id, reason=reason)
        except Exception:
            log.exception("could not give job %s back to the queue", current.job_id)
            return
        self._note_if_requeued(current.job_id)
        self.board.job_finished()

    def _note_if_requeued(self, job_id: str) -> None:
        """Record for the stop command's result that ``job_id`` is back in the queue."""
        try:
            job = self.store.get_job(job_id)
        except Exception:
            log.exception("could not read job %s", job_id)
            return
        if job is not None and job.status == "queued" and job_id not in self._requeued:
            self._requeued.append(job_id)

    # ------------------------------------------------------------------ workers
    def _workers_changed(self) -> None:
        supervisor = self._supervisor
        if supervisor is None or self._board is None:
            return
        self._board.set_workers(supervisor.workers(), supervisor.gpu_holder)

    def _default_supervisor(self, on_change: Callable[[], None]) -> WorkerSupervisor:
        return WorkerSupervisor(
            self.config,
            self.platform,
            roles=FAKE_ROLES if self.settings.fake_workers else None,
            below_normal=self.settings.below_normal,
            close_timeout_s=self.settings.worker_close_s,
            on_change=on_change,
        )
