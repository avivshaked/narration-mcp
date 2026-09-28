"""The job report resource's "Failures" section (plan.md WP48), read through the backend as an MCP client reads
``narration://jobs/{job_id}/report``: a job whose first take was cut short and retaken. Every text is invented
for these tests."""

from __future__ import annotations

import anyio

from narration import keys
from narration.contracts import codes
from tests.jobs.support import KETTLE, LAMPS, voice_hash

from .conftest import Service


def test_the_report_resource_lists_the_failed_take_and_its_retake_wp48(service: Service) -> None:
    first = keys.seed(voice_hash=voice_hash(service.world.clip_sha256), engine_text=LAMPS, attempt=0)
    service.world.faults({"kind": "token_cap", "when": {"text_contains": "lamplighter", "seed": first}})
    submitted = service.backend.submit_job_sync(service.request(LAMPS, KETTLE))
    service.world.run()
    job = service.world.store.get_job(submitted["job_id"])
    assert job is not None
    lamps = next(item for item in job.items if item.segment_id == "p01")
    failed, retake = lamps.attempts
    assert (failed.verdict, retake.verdict) == ("fail", "pass")

    async def read() -> str:
        return (await service.backend.read_resource(f"narration://jobs/{job.job_id}/report")).text

    report = anyio.run(read)
    section = report.split("## Failures\n", 1)[1].split("\n## ", 1)[0]
    assert (
        f"- p01 · attempt 0 · `{failed.take_id}` · **fail** · replaced: the slot was filled by attempt 1 "
        f"`{retake.take_id}` (pass)"
    ) in section
    assert f"`{codes.TOKEN_CAP_HIT}`" in section
    assert "p02" not in section  # the kettle's take passed
    assert str(service.world.store.root) not in report and ".wav" not in report  # no file path
