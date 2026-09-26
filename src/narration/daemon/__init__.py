"""narrationd: the detached singleton daemon (design sections 4, 4.1, 7.6 and 15; plan.md WP30).

One daemon runs per store. It is started detached, by the first submission or by
``narration-admin daemon start``, as::

    python -m narration.daemon --store <store_root> --config <path>

and it:

- holds the singleton for its store (``Platform.singleton``); a second daemon for the same store exits
  quietly;
- caps its own CPU thread pools as it caps its workers', before anything imports numpy (section 4.1);
- sweeps up after a daemon that died: jobs it left ``running`` go back to the queue (section 4.1);
- supervises the workers (``supervisor``): each one starts from its venv's Python with the offline
  environment, at below-normal priority, inside a kill-on-close group, so workers die with the daemon;
- writes ``run/daemon.json`` (``status``) on start and on every phase change, for ``get_server_status``;
- answers commands posted through the store (``release_gpu``, ``stop``, ``stop_now``; there are no
  sockets, section 4);
- unloads idle models after ``[daemon] idle_unload_s`` and exits after ``idle_exit_min`` idle;
- drives a ``JobRunner`` (``seam``), the job engine that claims and runs the work (WP31).

The modules:

- ``seam``: the ``JobRunner``, ``RunnerHost`` and ``WorkerPool`` protocols between the daemon and the job
  engine, and ``NullRunner``;
- ``service``: ``Daemon``, the control loop;
- ``supervisor``: ``WorkerSupervisor``, the worker processes;
- ``status``: ``StatusBoard``, which writes ``run/daemon.json``;
- ``sweep``: what a daemon does on start about the one before it;
- ``start``: how a front-end or the operator CLI starts the daemon detached, and tells whether one runs;
- ``settings``: the daemon's settings, from the configuration;
- ``testing``: ``FakeWorkerRunner``, a runner that drives the fake worker, for tests.

This package's ``__init__`` imports nothing: ``python -m narration.daemon`` imports it before the entry
point has capped the thread pools, and numpy must not load before that.
"""
