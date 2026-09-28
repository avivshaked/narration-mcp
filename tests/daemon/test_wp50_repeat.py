"""TEMPORARY (WP50): runs the start-failure tests 30 times each on CI, to show the race is gone. Removed before
review; pytest-repeat is not a dependency, so the repetition is a parametrization."""

from __future__ import annotations

from pathlib import Path

import pytest

from narration.config import Config
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore

from . import test_daemon, test_supervisor
from .conftest import DaemonFactory
from .test_supervisor import SupervisorFactory

supervisors = test_supervisor.supervisors  # the fixture

pytestmark = pytest.mark.timeout(300)

RUNS = range(30)


@pytest.mark.parametrize("run", RUNS)
def test_repeat_a_worker_that_cannot_start_fails_the_job_once(
    run: int, run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    test_daemon.test_a_worker_that_cannot_start_fails_the_job_once_s14(run_daemon, store)


@pytest.mark.parametrize("run", RUNS)
def test_repeat_a_poll_during_a_failed_start(
    run: int, supervisors: SupervisorFactory, platform: StandInPlatform, config: Config, tmp_path: Path
) -> None:
    test_supervisor.test_a_poll_during_a_failed_start_leaves_it_backend_not_installed_s14(
        supervisors, platform, config, tmp_path
    )
