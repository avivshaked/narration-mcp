"""Test support shared by every worker's tests (plan.md WP16).

- ``client.WorkerProcess``: a small synchronous client that starts a worker, talks the protocol, records
  every line the worker writes to stdout, and always stops and reaps the process.
- ``contract.WorkerContract``: the worker contract tests. Each worker (the fake here, ``qwen3`` in WP20,
  ``qa`` in WP22) subclasses it once and runs the same tests.

Only tests import this package; it needs pytest, which the worker package itself never imports.
"""
