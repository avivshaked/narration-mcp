"""``engine pin | repin | bridge`` with the fake worker (design section 10.1; plan.md DC-3, WP32): the profiles
recorded, the canary designed on this machine, the repeat test's tier, the calibrated threshold, and what a
changed installation does to each."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from narration.config import Config, EnginesConfig, QwenBaseConfig
from narration.contracts import names
from narration.contracts.models import CanaryMaterial
from narration.engine.canary import CALIBRATION_SEEDS, CANARY_MARGIN, find_canary, material_id
from narration.engine.models import QWEN_BASE, QWEN_DESIGN
from narration.engine.pinning import PinRefused, bridge, pin, plan
from narration.store import NarrationStore
from tests.store.standin import StandInPlatform

from .support import FAKE_WORKER_PACKAGES, CountingStarter, Install, fake_install, write_lock

WORKER_PACKAGES = FAKE_WORKER_PACKAGES
"""The fake worker reports only its own package, so the tests' profiles pin only that one."""
BASE_P1, BASE_P2 = "qwen3-base-1.7b.p1", "qwen3-base-1.7b.p2"
DESIGN_P1 = "qwen3-design-1.7b.p1"


@dataclasses.dataclass
class Pinned:
    install: Install
    store: NarrationStore
    starter: CountingStarter
    material: CanaryMaterial

    def pin(self, mode: Any = "pin", config: Config | None = None) -> Any:
        return pin(
            config or self.install.config,
            self.store,
            mode=mode,
            starter=self.starter,
            packages=WORKER_PACKAGES,
            device="cpu",
        )


@pytest.fixture
def pinned(tmp_path: Path) -> Iterator[Pinned]:
    install = fake_install(tmp_path)
    store = NarrationStore(install.store_root, StandInPlatform())
    try:
        with CountingStarter(install.config) as starter:
            yield Pinned(install=install, store=store, starter=starter, material=find_canary())
    finally:
        store.close()


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_first_pin_records_both_profiles_and_designs_the_canary_here_dc3(pinned: Pinned) -> None:
    reports = pinned.pin()

    assert [(r.kind, r.action, r.engine_profile_id) for r in reports] == [
        ("design", "new", DESIGN_P1),
        ("base", "new", BASE_P1),
    ]
    store, material = pinned.store, pinned.material
    design, base = store.current_engine_profile("design"), store.current_engine_profile("base")
    assert design is not None and base is not None
    assert (design.engine_profile_id, base.engine_profile_id) == (DESIGN_P1, BASE_P1)
    assert design.model_revision == QWEN_DESIGN.revision and base.model_revision == QWEN_BASE.revision
    assert base.packages == {"narration-worker": importlib.metadata.version("narration-worker")}
    # The repeat test: the fake repeats byte for byte, in one process and in a fresh one.
    for report in reports:
        assert report.tier == "bit_exact" and len(report.repeat_sha256) == 3 and len(set(report.repeat_sha256)) == 1
    assert design.tier == base.tier == "bit_exact"
    assert pinned.starter.started == ["qa", "qwen3", "qwen3"]  # QA, then the first and the fresh Qwen process

    assert design.canary is not None and base.canary is not None
    # VoiceDesign's gate render is the canary clip itself.
    assert design.canary.seed == material.design_seed and design.canary.raw_sha256 == design.canary.clip.sha256
    assert design.canary.transcript == material.design_text
    # Base clones that clip speaking the gate text, with the gate seed; each profile keeps its own copy.
    assert base.canary.seed == material.gate_seed and base.canary.clip.sha256 == design.canary.clip.sha256
    assert base.canary.raw_sha256 != base.canary.clip.sha256
    for profile in (design, base):
        assert profile.canary is not None
        clip = Path(profile.canary.clip.path)
        assert clip == store.root / "engines" / profile.engine_profile_id / "canary.wav"
        assert _sha(str(clip)) == profile.canary.clip.sha256
        assert profile.canary.material == material_id(material)
        assert len(profile.canary.embedding) > 0
    assert not list((store.root / "scratch" / "pin").glob("*/*.wav"))  # the work files are gone


def test_the_threshold_is_the_lowest_calibration_similarity_less_the_margin_dc3(pinned: Pinned) -> None:
    reports = {r.kind: r for r in pinned.pin()}
    for report in reports.values():
        assert len(report.calibration) == CALIBRATION_SEEDS
        assert report.threshold == pytest.approx(min(report.calibration) - CANARY_MARGIN)
    # The fake's takes of one voice are about 0.99 similar: Base's calibration reads the gate text in the
    # canary voice with other seeds.
    assert min(reports["base"].calibration) > 0.95


def test_pinning_again_keeps_what_is_pinned_and_starts_no_worker_s10_1(pinned: Pinned) -> None:
    first = {r.kind: r for r in pinned.pin()}
    pinned.starter.started.clear()
    again = pinned.pin()
    assert [r.action for r in again] == ["keep", "keep"]
    assert {r.kind: r.hash for r in again} == {k: r.hash for k, r in first.items()}
    assert pinned.starter.started == []


def test_pin_refuses_an_installation_that_differs_and_says_to_repin_s10_1(pinned: Pinned) -> None:
    pinned.pin()
    (pinned.install.snapshot(QWEN_BASE) / "model.safetensors").write_bytes(b"new weights")
    with pytest.raises(PinRefused) as caught:
        pinned.pin()
    assert "weights" in caught.value.message and "repin" in caught.value.hint
    assert pinned.store.current_engine_profile("base") is not None
    current = pinned.store.current_engine_profile("base")
    assert current is not None and current.engine_profile_id == BASE_P1


def test_repin_makes_a_new_profile_the_one_in_use_and_keeps_the_other_s10_1(pinned: Pinned) -> None:
    first = {r.kind: r for r in pinned.pin()}
    design_before = pinned.store.current_engine_profile("design")
    config = dataclasses.replace(
        pinned.install.config, engines=EnginesConfig(qwen3_base=QwenBaseConfig(max_new_tokens_per_char=3.0))
    )
    reports = {r.kind: r for r in pinned.pin("repin", config)}

    assert reports["design"].action == "keep" and reports["design"].hash == first["design"].hash
    assert reports["base"].action == "new" and reports["base"].engine_profile_id == BASE_P2
    assert reports["base"].changed == ("settings",)
    assert reports["base"].hash != first["base"].hash
    base = pinned.store.current_engine_profile("base")
    assert base is not None and base.engine_profile_id == BASE_P2 and base.settings["max_new_tokens_per_char"] == 3.0
    old = pinned.store.get_engine_profile(BASE_P1)
    assert old is not None and old.hash == first["base"].hash  # the old pin is kept, unchanged
    # The new Base canary clones the VoiceDesign profile's pinned clip.
    assert design_before is not None and design_before.canary is not None and base.canary is not None
    assert base.canary.clip.sha256 == design_before.canary.clip.sha256
    assert pinned.store.current_engine_profile("design") == design_before


def test_repin_with_nothing_changed_keeps_both_s10_1(pinned: Pinned) -> None:
    pinned.pin()
    assert [r.action for r in pinned.pin("repin")] == ["keep", "keep"]


def test_a_changed_canary_text_needs_a_repin_dc3(pinned: Pinned) -> None:
    pinned.pin()
    changed = dataclasses.replace(pinned.material, sha256="0" * 64)
    with pytest.raises(PinRefused) as caught:
        plan(pinned.install.config, pinned.store, "pin", changed, packages=WORKER_PACKAGES)
    assert "canary material" in caught.value.message
    plans = plan(pinned.install.config, pinned.store, "repin", changed, packages=WORKER_PACKAGES)
    assert {k: p.action for k, p in plans.items()} == {"design": "new", "base": "new"}


def test_a_worker_venv_that_does_not_match_its_lock_is_refused_before_rendering_s10_1(pinned: Pinned) -> None:
    write_lock(pinned.install.project, {"narration-worker": "9.9.9"})
    with pytest.raises(PinRefused) as caught:
        pinned.pin()
    assert "narration-worker" in caught.value.message and "Sync" in caught.value.hint
    assert pinned.store.list_engine_profiles() == ()


def test_bridge_compares_two_runnable_profiles_render_by_render_s10_1(pinned: Pinned) -> None:
    pinned.pin()
    config = dataclasses.replace(
        pinned.install.config, engines=EnginesConfig(qwen3_base=QwenBaseConfig(max_new_tokens_per_char=3.0))
    )
    pinned.pin("repin", config)
    report = bridge(config, pinned.store, BASE_P1, BASE_P2, starter=pinned.starter, device="cpu")
    data = report.as_dict()

    assert data["not_runnable"] == {}
    assert report.items[0].item == "canary" and len(report.items) == 4  # the canary and 3 paragraphs
    for item in report.items:
        # The cap only truncates (ADR 0003): these renders end under both caps, so they are the same bytes.
        assert item.old_from == "rendered" and item.same_bytes is True and item.similarity == pytest.approx(1.0)
    assert data["min_similarity"] == pytest.approx(1.0)


def test_bridge_uses_the_pinned_canary_when_the_old_profile_cannot_render_here_s10_1(pinned: Pinned) -> None:
    pinned.pin()
    (pinned.install.snapshot(QWEN_BASE) / "model.safetensors").write_bytes(b"new weights")
    pinned.pin("repin")
    report = bridge(pinned.install.config, pinned.store, BASE_P1, BASE_P2, starter=pinned.starter, device="cpu")

    assert list(report.not_runnable) == [BASE_P1]
    assert report.not_runnable[BASE_P1][0]["what"] == "weights"
    canary, *corpus = report.items
    assert canary.old_from == "pinned" and canary.similarity is not None and canary.similarity > 0.95
    assert corpus and all(i.old_from == "none" and i.similarity is None for i in corpus)


def test_bridge_refuses_unknown_or_mismatched_profiles_s10_1(pinned: Pinned) -> None:
    pinned.pin()
    with pytest.raises(PinRefused):
        bridge(pinned.install.config, pinned.store, BASE_P1, "qwen3-base-1.7b.p9", starter=pinned.starter)
    with pytest.raises(PinRefused):
        bridge(pinned.install.config, pinned.store, BASE_P1, DESIGN_P1, starter=pinned.starter)


def test_the_first_ids_are_the_designs_s6(pinned: Pinned) -> None:
    ids = {r.engine_profile_id for r in pinned.pin()}
    assert ids == {names.ENGINE_PROFILE_BASE, names.ENGINE_PROFILE_DESIGN}


def test_the_pin_starts_its_workers_as_the_daemon_does_s17(pinned: Pinned) -> None:
    """Through the daemon's own supervisor (which also sets the worker's folder and environment, section 17):
    each worker joins the kill-on-close group before its hello, at the configured priority."""
    pinned.pin()
    platform = pinned.starter.platform
    assert len(platform.added) == len(pinned.starter.started) == 3  # QA, then the first and the fresh Qwen
    assert platform.groups_opened == 1
    assert platform.lowered == platform.added  # [workers] priority is below_normal by default
