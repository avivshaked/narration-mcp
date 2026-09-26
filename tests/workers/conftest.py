"""Fixtures for the worker client tests: a store, and clients of the fake worker that are always closed."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from narration_worker.fake.faults import SPEC_ENV

from narration.config import Config, EnginesConfig
from narration.contracts.names import MAX_NEW_TOKENS_CEILING, max_new_tokens_for
from narration.workers import SubprocessWorkerClient, WorkerCommand, worker_command

ClientFactory = Callable[..., SubprocessWorkerClient]


def call_cap(text: str, *, design: bool = False) -> int:
    """The ``max_new_tokens`` the daemon passes for a call that speaks ``text`` (design section 10.1, DC-4),
    with the default engine settings."""
    engines = EnginesConfig()
    engine = engines.qwen3_design if design else engines.qwen3_base
    return max_new_tokens_for(
        text,
        per_char=engine.max_new_tokens_per_char,
        floor=engine.max_new_tokens_floor,
        ceiling=MAX_NEW_TOKENS_CEILING,
    )


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    (root / "scratch").mkdir(parents=True)
    return root


@pytest.fixture
def config(store: Path) -> Config:
    return Config.for_tests(store)


def fake_command(config: Config, spec: dict[str, Any] | None = None, *, spec_dir: Path | None = None) -> WorkerCommand:
    """The fake worker's command, with a fault spec in its environment when one is given."""
    base = {k: v for k, v in os.environ.items() if k != SPEC_ENV}
    if spec is not None:
        assert spec_dir is not None
        path = spec_dir / "fake-spec.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        base[SPEC_ENV] = str(path)
    return worker_command(config, "fake", base_env=base)


@pytest.fixture
def make_client(config: Config, tmp_path: Path) -> Iterator[ClientFactory]:
    """Makes fake-worker clients; every one is closed (and its process killed) at teardown."""
    clients: list[SubprocessWorkerClient] = []

    def make(spec: dict[str, Any] | None = None, **options: Any) -> SubprocessWorkerClient:
        client = SubprocessWorkerClient(fake_command(config, spec, spec_dir=tmp_path), **options)
        clients.append(client)
        return client

    try:
        yield make
    finally:
        for client in clients:
            client.close(timeout_s=5.0)
