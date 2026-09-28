"""``audition_pronunciation`` on this machine's GPU (markers ``gpu``, ``model`` and ``slow``; design sections 7.6,
11.1, 17.4): the real Qwen Base and QA workers render an invented term in an invented carrier with two respelling
variants, cloning a voice the service designed, and each take comes back scored with what the recogniser heard
and its similarity to the clip.

It pins every engine in a fresh store under the test's temporary folder (``engine pin``), designs one candidate
(so the voice is synthetic and on the provenance list, with no allowlist edit), then submits the audition through
the backend with that candidate's clip and exact transcript. The voice is never measured: an audition does not
need it. Each job runs on the daemon's job engine (``installed_engine``) over the real worker supervisor. Nothing
it renders leaves the temporary folder.

Run it (the Qwen models need about 6 GB of VRAM; pinning takes several minutes)::

    uv run python -m pytest -m "gpu and model" tests/audition/test_audition_gpu.py -s

It needs what ``tests/engine/test_canary_gpu.py`` needs: ``NARRATION_MODELS_ROOT`` with the pinned models, the
Qwen and QA worker venvs synced (under ``NARRATION_WORKERS_ROOT``, else this checkout's ``workers/``), an NVIDIA
GPU with NVML, and the developers' GPU lock (holder ``$NARRATION_GPU_LOCK_HOLDER``, default
``audition-gpu-tests``). It skips with the reason when one is missing.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from narration.engine.pinning import SupervisedStarter, pin
from narration.platform import get_platform
from narration.store import NarrationStore
from tests.backend.support import make_backend, sha256_of
from tests.design import test_design_gpu as design_gpu
from tests.engine import test_canary_gpu as canary_gpu

DESCRIPTION = (
    "A calm, clear narrator with a warm middle voice, steady pace and crisp consonants, reading as if to a friend."
)
"""Invented for this test, and positive only (section 3.5)."""
TERM = "Quillomar"
"""An invented name."""
CARRIER = "The lighthouse keeper rowed out to Quillomar before the storm."
"""An invented carrier sentence that holds the term."""
VARIANTS = ({"label": "kwil", "respell": "Kwill-oh-mar"}, {"label": "keel", "respell": "Keel-oh-mahr"})
"""Two invented respellings."""

pytestmark = [pytest.mark.gpu, pytest.mark.model, pytest.mark.slow, pytest.mark.timeout(canary_gpu.RUN_TIMEOUT_S)]

gpu_lock = design_gpu.gpu_lock


@pytest.mark.usefixtures("gpu_lock")
def test_an_audition_renders_each_variant_on_the_real_engine_s7_6_s11_1(tmp_path: Path) -> None:
    config = canary_gpu._config(tmp_path / "store")  # pyright: ignore[reportPrivateUsage]
    platform = get_platform()
    with NarrationStore(config.server.store_root, platform) as store:
        started = time.monotonic()
        with SupervisedStarter(config, platform) as starter:
            pin(config, store, mode="pin", starter=starter)
        print(f"pinned in {time.monotonic() - started:.0f} s")

        backend, _, _ = make_backend(SimpleNamespace(store=store, config=config))
        design = backend.design_voice_sync({"name": "audition voice", "description": DESCRIPTION, "takes": 1})
        job = design_gpu._run_alone(config, platform, store, design["job_id"])  # pyright: ignore[reportPrivateUsage]
        assert job.status == "completed", job.error
        (candidate,) = store.get_design(design["design_id"])
        voice = {"path": candidate.clip.path, "sha256": candidate.clip.sha256, "transcript": candidate.transcript}
        assert sha256_of(Path(candidate.clip.path)) == candidate.clip.sha256

        submitted = backend.audition_pronunciation_sync(
            {"voice": voice, "term": TERM, "variants": [dict(v) for v in VARIANTS], "carrier": CARRIER}
        )
        started = time.monotonic()
        job = design_gpu._run_alone(config, platform, store, submitted["job_id"])  # pyright: ignore[reportPrivateUsage]
        print(f"the audition: {job.status} {job.outcome} in {time.monotonic() - started:.0f} s")
        assert job.status == "completed", job.error

        audition = backend.get_results_sync({"job_id": submitted["job_id"], "include_words": False})["audition"]
        assert [v["label"] for v in audition["variants"]] == [v["label"] for v in VARIANTS]
        for variant in audition["variants"]:
            assert variant["engine_text"] == CARRIER.replace(TERM, variant["respell"])
            assert variant["takes"], "each variant has a take"
            for take, heard, sim in zip(variant["takes"], variant["heard"], variant["spk_sim_clip"], strict=True):
                qa = take["qa"]
                print(
                    f"{variant['label']} attempt {take['attempt']}: {qa['verdict']}, wer_adj {qa['wer_adj']}, "
                    f"heard {heard!r}, similarity to the clip {sim}, flags {[f['code'] for f in qa['flags']]}"
                )
                assert Path(take["delivery"]["path"]).is_file()
                assert sha256_of(Path(take["delivery"]["path"])) == take["delivery"]["sha256"]
                assert qa["spk_sim_anchor"] is None, "no speaker check in an audition's verdict"
                assert sim is not None and sim >= config.measurement.sim_fail_floor, "the take is the voice"
            assert variant["takes"][-1]["qa"]["verdict"] in ("pass", "warn"), "the last take is usable"
