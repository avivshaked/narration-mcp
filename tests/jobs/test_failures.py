"""How the engine meets a failure: a guard that refuses a load, a pool whose residency disagrees with the
engine's, a voice the worker cannot prepare, out of memory while loading, and a stop that kills a request in
flight (design sections 4, 4.1, 10.1 and 14; plan.md WP31)."""

from __future__ import annotations

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import EngineProfile
from narration.contracts.names import CanaryStatus
from narration.contracts.worker import HelloReply
from narration.jobs.host import RunnerHost

from .conftest import World
from .support import KETTLE, LAMPS

# ======================================================================== the engine guard (WP32's canary gate)


class DriftOnce:
    """An engine guard whose first check finds the engine drifted, and every later one a match."""

    def __init__(self) -> None:
        self.calls = 0

    def after_load(self, host: RunnerHost, profile: EngineProfile, hello: HelloReply | None) -> CanaryStatus:
        self.calls += 1
        if self.calls == 1:
            raise NarrationError(codes.ENGINE_DRIFT, "the canary's audio differs, and so does its similarity")
        return "hash_match"


def test_a_load_the_guard_refuses_is_unloaded_and_the_next_job_is_checked_again_s10_1(world: World) -> None:
    guard = DriftOnce()
    world.new_engine(guard=guard)
    first = world.submit(LAMPS)
    world.run()
    refused = world.job(first.job_id)
    assert refused.status == "failed" and refused.error is not None and refused.error.code == codes.ENGINE_DRIFT
    assert world.pool.unloads == ["qwen"] and "qwen" not in world.pool.loaded()
    assert world.pool.texts() == []  # nothing is rendered on a load the guard refused

    second = world.submit(KETTLE)
    world.run()
    done = world.job(second.job_id)
    assert done.status == "completed"
    assert guard.calls == 2  # the next job loaded again, and was checked again
    render = world.store.get_render(done.items[0].attempts[0].render_key)
    assert render is not None and render.canary.batch_status == "hash_match"
    assert world.pool.texts() == [KETTLE]
