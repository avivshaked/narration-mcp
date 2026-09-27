"""``narration-admin engine pin | repin | bridge``: recording the engine profiles and designing the canary on
this machine (design section 10.1; plan.md DC-3, WP32).

**pin** records the Base and VoiceDesign profiles this installation would pin now (``profile.build_profile``)
and makes each one's canary:

1. **The canary clip** is designed with VoiceDesign from the canary's description, design text and seed
   (``material/canary/canary.v1``); it is the VoiceDesign profile's gate render.
2. **The Base gate render** clones that clip (its transcript is the design text) speaking the gate's text
   with the gate's seed.
3. **The repeat test** (ADR 0002 decision 4): each gate render is made twice in one worker process and once
   in a fresh one. All three hashes equal is the ``bit_exact`` tier; anything else is ``similar``.
4. **The threshold** (``canary.calibrate``): each canary is rendered with ``canary.CALIBRATION_SEEDS`` more
   seeds, embedded with WavLM on the QA worker's CPU, and compared with the gate render's embedding.
5. The clip goes under ``engines/<engine_profile_id>/canary.wav`` (immutable, never collected), and the
   profile, with its tier, the machine observed and the ``CanaryPin``, is stored and made current.

The worker is checked against the profile before anything is rendered: a venv that does not match its lock
(``drift.DriftCheck.fingerprint``) is refused with what to do.

**pin** changes nothing that is already pinned. A profile whose installation still matches is kept; one that
differs (weights, lock, settings, the canary's text) is refused, with the fields that differ. A kept profile
is updated in place where an unhashed field differs from what this build would record, so no render key changes:
its ``snapshot_dir`` when the snapshot folder moved (a new ``[server] models_root``, same files), and its
``vram_need_mb`` when this build estimates Qwen's VRAM need anew (DC-16).

**repin** makes a new profile (the next id) for each engine whose installation differs, and makes it current:
its hash differs, so every render key differs, measurements are made again, and a caller that sends
``expect_engine_profile`` gets ``ENGINE_CHANGED`` until it accepts the new hash. It also starts the Qwen worker
to see the machine: an engine pinned on another GPU, driver, CUDA or cuDNN (``profile.OBSERVED_KEYS``, the
change section 10.1 names) gets a new profile and a fresh canary too, though no pinned file changed.
``repin --force`` gives both engines new profiles whatever changed: the operator's way out when a gate fails
for a reason nothing here can see. Neither changes a cached file.

**bridge** renders the canary and the calibration corpus's paragraphs under an old and a new profile, in the
old profile's canary voice, and reports how similar each pair is, so the owner can judge whether a re-pin will
be heard before making it. A profile's files are looked for where it recorded them, then under the configured
models root (a root that moved). A profile whose environment is gone (another lock, missing weights) cannot
render here; its side of the canary is then its pinned embedding, and the corpus is skipped.

**All or nothing.** Every canary is rendered, embedded and calibrated before anything is stored; only then are
the clips and profiles stored, and the profiles in use set, both last. A refusal or a worker failure on either
engine leaves the store as it was.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from narration import keys
from narration.config import Config
from narration.contracts import names
from narration.contracts.errors import WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.interfaces import Store, WorkerClient
from narration.contracts.models import AudioRef, CanaryMaterial, EngineProfile
from narration.contracts.names import DeterminismTier, EngineKind, GpuHolder
from narration.daemon.supervisor import FAKE_ROLES, WorkerSupervisor
from narration.jobs.gpu import LOAD_TIMEOUT_S, UNLOAD_TIMEOUT_S
from narration.jobs.pins import qwen_load_payload
from narration.platform import ProcessPlatform
from narration.store.store import utc_iso

from .canary import (
    calibrate,
    calibration_seeds,
    canary_voice_hash,
    clone,
    cosine,
    design,
    embed,
    embedding_problem,
    engine_kind,
    find_canary,
    find_set,
    material_id,
    pin_record,
    read_set,
    render,
    sv_load_payload,
)
from .drift import DriftCheck
from .models import WAVLM_SV, snapshot_dir
from .profile import (
    QWEN_PACKAGES,
    FileHashes,
    build_profile,
    machine_differences,
    next_profile_id,
    observed,
    pin_differences,
    qwen_project,
    unobserved,
    with_hash,
)
from .qa import model_pin

log = logging.getLogger(__name__)

KINDS: Final[tuple[EngineKind, ...]] = ("design", "base")
"""The order the pin makes the canaries in: the Base canary clones the VoiceDesign one's clip."""
Mode = Literal["pin", "repin"]
Action = Literal["keep", "new", "canary"]
WorkerRoleName = Literal["qwen3", "qa"]
SCRATCH: Final = "pin"
CLOSE_TIMEOUT_S: Final = 30.0
KEPT_UPDATES: Final = ("snapshot_dir", "vram_need_mb")
"""The unhashed fields a kept profile takes from what this build would record, in place: where its snapshot
folder is now, and the VRAM Qwen needs by this build's estimate (DC-16). Neither changes a render key."""


class PinRefused(Exception):
    """The pin cannot go ahead as asked; ``hint`` says what to do instead."""

    def __init__(self, message: str, *, hint: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = dict(details or {})


class Starter(Protocol):
    """Starts a worker process for the pin: a fresh process on every call."""

    def start(self, role: WorkerRoleName, *, cublas_workspace_config: str | None = None) -> WorkerClient:
        """A started worker of ``role`` (its ``hello`` done)."""
        ...


class SupervisedStarter:
    """Starts the pin's workers as the daemon starts its own, through the daemon's ``WorkerSupervisor``: the
    worker venv's Python with ``-P``, run in the worker project's folder (never the store root, section 17),
    with the offline environment, the thread cap, NVML's GPU order and this OS's hardening, at the configured
    priority and in a kill-on-close group. So the canary is rendered as the daemon will render it, on the same
    GPU. Every ``start`` is a fresh process. Use it as a context manager: leaving stops every worker. With
    ``fake``, the fake worker serves every role (tests)."""

    def __init__(
        self,
        config: Config,
        platform: ProcessPlatform,
        *,
        fake: bool = False,
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        self._pool = WorkerSupervisor(
            config,
            platform,
            roles=FAKE_ROLES if fake else None,
            below_normal=config.workers.priority == "below_normal",
            base_env=base_env,
        )

    def __enter__(self) -> SupervisedStarter:
        self._pool.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._pool.close()

    def start(self, role: WorkerRoleName, *, cublas_workspace_config: str | None = None) -> WorkerClient:
        group: GpuHolder = "qwen" if role == "qwen3" else "qa"
        self._pool.stop(group)  # a fresh process: the repeat test's second process must be a new one
        return self._pool.client(group, cublas_workspace_config=cublas_workspace_config)


# ======================================================================== what to pin


@dataclass(frozen=True, slots=True, kw_only=True)
class Plan:
    """What the pin does for one engine: ``keep`` the current profile, make its missing ``canary``, or pin a
    ``new`` profile. ``changed`` lists the hashed fields in which the installation differs from the current
    profile (and ``canary material`` when the canary's text changed)."""

    kind: EngineKind
    action: Action
    profile: EngineProfile
    current: EngineProfile | None
    changed: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    """For ``keep``: the unhashed fields of the current profile to update in place (``KEPT_UPDATES``)."""
    replaced: EngineProfile | None = None
    """For ``new`` over a current profile whose snapshot folder moved (the same revision and weights): that
    profile with its ``snapshot_dir`` where its files are now, stored in place with the new pin, so that
    ``engine bridge`` can still render it."""


def plan(
    config: Config,
    store: Store,
    mode: Mode,
    material: CanaryMaterial,
    *,
    hashes: FileHashes | None = None,
    packages: Sequence[str] = QWEN_PACKAGES,
    machine: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[EngineKind, Plan]:
    """What ``pin`` or ``repin`` would do for each engine (see the module docstring). ``machine`` is what the
    Qwen worker observes of this machine (``profile.observed``; repin only): an engine pinned on another is
    re-pinned. ``force`` (repin only) re-pins both engines. Raises ``PinRefused`` when ``pin`` finds an
    installation that differs from what is pinned; raises ``EngineSetupError`` when what a profile is built
    from is missing."""
    if (force or machine is not None) and mode != "repin":
        raise ValueError("only repin compares the machine or can be forced")
    hashes = hashes if hashes is not None else FileHashes()
    ids = [p.engine_profile_id for p in store.list_engine_profiles()]
    plans: dict[EngineKind, Plan] = {}
    for kind in KINDS:
        current = store.current_engine_profile(kind)
        if current is None:
            new_id = next_profile_id(ids, kind)
            ids.append(new_id)
            plans[kind] = Plan(
                kind=kind,
                action="new",
                profile=build_profile(config, kind, engine_profile_id=new_id, hashes=hashes, packages=packages),
                current=None,
            )
            continue
        built = build_profile(
            config, kind, engine_profile_id=current.engine_profile_id, hashes=hashes, packages=packages
        )
        changed = list(pin_differences(current, built))
        if current.canary is not None and current.canary.material != material_id(material):
            changed.append("canary material")
        if not changed and force:
            changed.append("forced")
        if not changed and machine is not None:
            unseen = unobserved(current.observed, machine)
            if unseen:
                raise PinRefused(
                    f"the Qwen worker could not observe this machine's {', '.join(unseen)}, so repin cannot tell "
                    f"whether it has changed since engine profile {current.engine_profile_id} was pinned",
                    hint=(
                        "Run narration-admin doctor to check the GPU and its driver, then repin again; "
                        "narration-admin engine repin --force re-pins both engines regardless. Nothing was pinned."
                    ),
                    details={"engine_profile_id": current.engine_profile_id, "unobserved": list(unseen)},
                )
            changed += [f"machine {k}" for k in machine_differences(current.observed, machine)]
        if not changed:
            if current.canary is None:
                plans[kind] = Plan(kind=kind, action="canary", profile=built, current=current)
                continue
            updated = tuple(f for f in KEPT_UPDATES if getattr(built, f) != getattr(current, f))
            kept = dataclasses.replace(current, **{f: getattr(built, f) for f in updated})
            plans[kind] = Plan(kind=kind, action="keep", profile=kept, current=current, updated=updated)
            continue
        if mode == "pin":
            raise PinRefused(
                f"this installation differs from the pinned engine profile {current.engine_profile_id} in: "
                + ", ".join(changed),
                hint=(
                    "To make a new profile the one in use, run narration-admin engine repin (every render key "
                    "changes and voices are measured again; narration-admin engine bridge compares the two "
                    "first). To keep the pinned one, restore the installation it pinned."
                ),
                details={"engine_profile_id": current.engine_profile_id, "changed": changed},
            )
        new_id = next_profile_id(ids, kind)
        ids.append(new_id)
        replaced = None
        if built.snapshot_dir != current.snapshot_dir and (built.model_revision, dict(built.weights)) == (
            current.model_revision,
            dict(current.weights),
        ):
            replaced = dataclasses.replace(current, snapshot_dir=built.snapshot_dir)
        plans[kind] = Plan(
            kind=kind,
            action="new",
            profile=with_hash(dataclasses.replace(built, engine_profile_id=new_id)),
            current=current,
            changed=tuple(changed),
            replaced=replaced,
        )
    return plans


# ======================================================================== the canary runs


@dataclass(slots=True)
class _Made:
    """One engine's canary as the pin makes it."""

    plan: Plan
    clip: AudioRef
    transcript: str
    seed: int
    first: Path
    hashes: list[str] = field(default_factory=list)
    calibration: list[Path] = field(default_factory=list)
    hello: Any = None


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineReport:
    """What the pin did for one engine."""

    kind: EngineKind
    action: Action
    engine_profile_id: str
    hash: str
    changed: tuple[str, ...] = ()
    tier: DeterminismTier | None = None
    raw_sha256: str | None = None
    repeat_sha256: tuple[str, ...] = ()
    threshold: float | None = None
    calibration: tuple[float, ...] = ()
    updated: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """The report as JSON values."""
        return {
            "kind": self.kind,
            "action": self.action,
            "engine_profile_id": self.engine_profile_id,
            "hash": self.hash,
            "changed": list(self.changed),
            "updated": list(self.updated),
            "tier": self.tier,
            "raw_sha256": self.raw_sha256,
            "repeat_sha256": list(self.repeat_sha256),
            "threshold": self.threshold,
            "calibration": list(self.calibration),
        }


def pin(
    config: Config,
    store: Store,
    *,
    mode: Mode,
    starter: Starter,
    material: CanaryMaterial | None = None,
    device: str | None = None,
    hashes: FileHashes | None = None,
    packages: Sequence[str] = QWEN_PACKAGES,
    clock: Callable[[], float] = time.time,
    force: bool = False,
) -> tuple[EngineReport, ...]:
    """``engine pin`` (``mode="pin"``) or ``engine repin`` (``force``: both engines anew): see the module
    docstring. Returns what was done for each engine. Raises ``PinRefused``, ``EngineSetupError``
    (``BACKEND_NOT_INSTALLED``), and what a worker raises (``WorkerFailure``, ``WorkerCrashed``,
    ``WorkerTimeout``); the store is then as it was, for both engines: every canary is rendered, embedded and
    calibrated before anything is stored."""
    material = material if material is not None else find_canary(config)
    hashes = hashes if hashes is not None else FileHashes()
    plans = plan(config, store, mode, material, hashes=hashes, packages=packages, force=force)
    if mode == "repin" and any(p.action == "keep" for p in plans.values()):
        machine = _observe(starter, list(plans.values()))
        plans = plan(config, store, mode, material, hashes=hashes, packages=packages, machine=machine)
    todo = [plans[k] for k in KINDS if plans[k].action != "keep"]
    if not todo:
        _update_in_place(store, plans.values())
        return tuple(_kept(plans[k]) for k in KINDS)
    device = device if device is not None else config.gpu.device
    sv = model_pin(config, WAVLM_SV)
    work = store.scratch_path(SCRATCH, uuid.uuid4().hex, "work").parent
    try:
        with _worker(starter, "qa") as qa:
            qa.request("load", sv_load_payload(sv), timeout_s=LOAD_TIMEOUT_S)
            made = _first_pass(starter, plans, todo, material, device, work, mode)
            _repeat(starter, made, material, device, work, mode)
            measured = _measure(qa, made)
        _update_in_place(store, plans.values())
        reports = _store(store, measured, material, clock, work)
        by_kind = {r.kind: r for r in reports}
        return tuple(by_kind.get(k) or _kept(plans[k]) for k in KINDS)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _kept(p: Plan) -> EngineReport:
    profile = p.profile
    pin_ = profile.canary
    return EngineReport(
        kind=p.kind,
        action="keep",
        engine_profile_id=profile.engine_profile_id,
        hash=profile.hash,
        tier=profile.tier,
        raw_sha256=pin_.raw_sha256 if pin_ is not None else None,
        threshold=pin_.threshold if pin_ is not None else None,
        updated=p.updated,
    )


def _observe(starter: Starter, plans: Sequence[Plan]) -> dict[str, Any]:
    """What a fresh Qwen worker observes of this machine (its ``hello``), started as the pin starts it."""
    with _worker(starter, "qwen3", cublas=_cublas(plans)) as qwen:
        return observed(qwen.hello)


@contextmanager
def _worker(starter: Starter, role: WorkerRoleName, *, cublas: str | None = None) -> Iterator[WorkerClient]:
    client = starter.start(role, cublas_workspace_config=cublas)
    try:
        yield client
    finally:
        try:
            client.close(timeout_s=CLOSE_TIMEOUT_S)
        except (WorkerFailure, WorkerCrashed, WorkerTimeout, OSError) as exc:
            log.warning("closing the %s worker failed: %s", role, exc)


def _load(qwen: WorkerClient, profile: EngineProfile, device: str, mode: Mode) -> None:
    """Load a profile on the pin's Qwen worker, after checking the worker runs as the profile's lock says. A
    refusal names the command that ran (``mode``), to run again once the venv is synced."""
    drift = DriftCheck().fingerprint(profile, qwen.hello)
    if drift:
        raise PinRefused(
            "the qwen3 worker does not run as its lock says: "
            + ", ".join(f"{d.name} {d.found} (locked {d.pinned})" for d in drift),
            hint=f"Sync the worker's venv to its lock (narration-admin install), then run narration-admin engine "
            f"{mode} again; nothing was pinned.",
            details={"drift": [d.as_dict() for d in drift]},
        )
    qwen.request("load", qwen_load_payload(profile, device), timeout_s=LOAD_TIMEOUT_S)


def _cublas(todo: Sequence[Plan]) -> str:
    values = {p.profile.determinism.cublas_workspace_config for p in todo}
    if len(values) != 1:
        raise PinRefused(
            "the two engine profiles pin different CUBLAS_WORKSPACE_CONFIG values",
            hint="Set one CUBLAS_WORKSPACE_CONFIG in [workers] env, then pin again.",
        )
    return values.pop()


def _first_pass(
    starter: Starter,
    plans: Mapping[EngineKind, Plan],
    todo: Sequence[Plan],
    material: CanaryMaterial,
    device: str,
    work: Path,
    mode: Mode,
) -> list[_Made]:
    """Process A: each canary twice, and the calibration renders."""
    made: list[_Made] = []
    with _worker(starter, "qwen3", cublas=_cublas(todo)) as qwen:
        clip: AudioRef | None = None
        transcript = material.design_text
        for p in todo:
            _load(qwen, p.profile, device, mode)
            if p.kind == "design":
                seed = material.design_seed
                first = work / "design-first.wav"
                sha = render(qwen, p.profile, material, seed=seed, out=first)
                clip = AudioRef(path=str(first), sha256=sha)
                entry = _Made(plan=p, clip=clip, transcript=transcript, seed=seed, first=first, hello=qwen.hello)
            else:
                if clip is None:  # the VoiceDesign profile is kept: clone its pinned canary clip
                    design_pin = plans["design"].profile.canary
                    assert design_pin is not None
                    copy = work / "design-clip.wav"
                    shutil.copyfile(design_pin.clip.path, copy)
                    clip = AudioRef(path=str(copy), sha256=design_pin.clip.sha256)
                    transcript = design_pin.transcript
                seed = material.gate_seed
                first = work / "base-first.wav"
                sha = render(qwen, p.profile, material, seed=seed, out=first, clip=clip, transcript=transcript)
                entry = _Made(plan=p, clip=clip, transcript=transcript, seed=seed, first=first, hello=qwen.hello)
            entry.hashes.append(sha)
            again = work / f"{p.kind}-again.wav"
            entry.hashes.append(
                render(qwen, p.profile, material, seed=seed, out=again, clip=clip, transcript=transcript)
            )
            for s in calibration_seeds(seed):
                out = work / f"{p.kind}-calibration-{s}.wav"
                render(qwen, p.profile, material, seed=s, out=out, clip=clip, transcript=transcript)
                entry.calibration.append(out)
            qwen.request("unload", {}, timeout_s=UNLOAD_TIMEOUT_S)
            made.append(entry)
    return made


def _repeat(
    starter: Starter, made: Sequence[_Made], material: CanaryMaterial, device: str, work: Path, mode: Mode
) -> None:
    """Process B, a fresh worker: each canary once more."""
    with _worker(starter, "qwen3", cublas=_cublas([m.plan for m in made])) as qwen:
        for m in made:
            _load(qwen, m.plan.profile, device, mode)
            out = work / f"{m.plan.kind}-fresh.wav"
            m.hashes.append(
                render(qwen, m.plan.profile, material, seed=m.seed, out=out, clip=m.clip, transcript=m.transcript)
            )
            qwen.request("unload", {}, timeout_s=UNLOAD_TIMEOUT_S)


@dataclass(frozen=True, slots=True)
class _Measured:
    """One engine's canary, measured and ready to store."""

    made: _Made
    embedding: tuple[float, ...]
    threshold: float
    calibration: tuple[float, ...]
    tier: DeterminismTier


def _measure(qa: WorkerClient, made: Sequence[_Made]) -> list[_Measured]:
    """Embed, check and calibrate every canary, and settle each tier, before anything is stored: a refusal or
    a worker failure here leaves the store as it was, for both engines."""
    measured: list[_Measured] = []
    for m in made:
        embedding = embed(qa, m.first)
        problem = embedding_problem(embedding)
        if problem is not None:
            raise PinRefused(
                f"the {m.plan.kind} canary render could not be embedded ({problem.replace('_', ' ')}), so no "
                "gate can be calibrated from it",
                hint="Run narration-admin doctor to check the QA models, then pin again; nothing was pinned.",
                details={"kind": m.plan.kind, "embedding": problem},
            )
        threshold, sims = calibrate(embedding, [embed(qa, path) for path in m.calibration])
        tier: DeterminismTier = "bit_exact" if len(set(m.hashes)) == 1 else "similar"
        measured.append(_Measured(made=m, embedding=embedding, threshold=threshold, calibration=sims, tier=tier))
    return measured


def _update_in_place(store: Store, plans: Iterable[Plan]) -> None:
    """The unhashed updates of the profiles the pin keeps or replaces (``Plan.updated``, ``Plan.replaced``)."""
    for p in plans:
        if p.action == "keep" and p.updated:
            store.put_engine_profile(p.profile)
            log.info("engine profile %s: updated %s", p.profile.engine_profile_id, ", ".join(p.updated))
        if p.replaced is not None:
            store.put_engine_profile(p.replaced)
            replaced = p.replaced
            log.info("engine profile %s: its snapshot is now at %s", replaced.engine_profile_id, replaced.snapshot_dir)


def _store(
    store: Store, measured: Sequence[_Measured], material: CanaryMaterial, clock: Callable[[], float], work: Path
) -> list[EngineReport]:
    """Store each measured profile with its canary, then make them the profiles in use, both last, so the
    engines in use change together. Each profile's clip is a copy made in the run's ``work`` folder, which the
    pin removes whatever happens, so a clip the store could not take is never left behind."""
    reports: list[EngineReport] = []
    pinned_at = utc_iso(clock())
    for x in measured:
        m, profile = x.made, x.made.plan.profile
        copy = work / f"{profile.engine_profile_id}-canary.wav"
        shutil.copyfile(m.clip.path, copy)
        clip = store.put_canary_clip(profile.engine_profile_id, copy)
        canary = pin_record(
            material=material,
            clip=clip,
            transcript=m.transcript,
            seed=m.seed,
            raw_sha256=m.hashes[0],
            embedding=x.embedding,
            threshold=x.threshold,
            pinned_at=pinned_at,
        )
        store.put_engine_profile(dataclasses.replace(profile, tier=x.tier, observed=observed(m.hello), canary=canary))
        reports.append(
            EngineReport(
                kind=m.plan.kind,
                action=m.plan.action,
                engine_profile_id=profile.engine_profile_id,
                hash=profile.hash,
                changed=m.plan.changed,
                tier=x.tier,
                raw_sha256=m.hashes[0],
                repeat_sha256=tuple(m.hashes),
                threshold=x.threshold,
                calibration=x.calibration,
            )
        )
    for x in measured:
        profile = x.made.plan.profile
        store.set_current_engine_profile(x.made.plan.kind, profile.engine_profile_id)
        log.info("pinned %s (%s, %s)", profile.engine_profile_id, profile.hash, x.tier)
    return reports


# ======================================================================== bridge


@dataclass(frozen=True, slots=True, kw_only=True)
class BridgeItem:
    """One text rendered under both profiles: how similar the two renders are, and whether they are the same
    bytes. ``old_from`` says where the old side came from: ``rendered``, or ``pinned`` (the old profile's
    stored canary embedding, when it cannot render here)."""

    item: str
    similarity: float | None
    same_bytes: bool | None
    old_from: Literal["rendered", "pinned", "none"]

    def as_dict(self) -> dict[str, Any]:
        """The item as JSON values."""
        return {
            "item": self.item,
            "similarity": self.similarity,
            "same_bytes": self.same_bytes,
            "old_from": self.old_from,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class BridgeReport:
    """``engine bridge``'s answer: each item's similarity, and why a side could not render here."""

    old: str
    new: str
    items: tuple[BridgeItem, ...]
    not_runnable: dict[str, list[dict[str, Any]]]

    def as_dict(self) -> dict[str, Any]:
        """The report as JSON values."""
        sims = [i.similarity for i in self.items if i.similarity is not None]
        return {
            "old": self.old,
            "new": self.new,
            "min_similarity": min(sims) if sims else None,
            "items": [i.as_dict() for i in self.items],
            "not_runnable": self.not_runnable,
        }


def corpus_texts(config: Config, set_id: str = names.CORPUS) -> list[tuple[str, str]]:
    """The calibration corpus's calibration paragraphs, as (segment id, text): each paragraph's cues joined
    with single spaces, as a request joins them (section 7.2)."""
    root = find_set(config, "calibration", set_id)
    _, _, files = read_set(root, "calibration", set_id)
    data = json.loads(files["paragraphs.json"].decode("utf-8"))
    return [(str(p["segment_id"]), " ".join(str(c["text"]) for c in p["cues"])) for p in data["calibration"]]


def bridge(
    config: Config,
    store: Store,
    old_id: str,
    new_id: str,
    *,
    starter: Starter,
    material: CanaryMaterial | None = None,
    device: str | None = None,
) -> BridgeReport:
    """``engine bridge <old> <new>`` (see the module docstring). Changes nothing in the store."""
    old, new = store.get_engine_profile(old_id), store.get_engine_profile(new_id)
    for ident, profile in ((old_id, old), (new_id, new)):
        if profile is None:
            raise PinRefused(
                f"no engine profile {ident} is pinned",
                hint="Name two pinned profiles (narration-admin engine show lists them).",
            )
    assert old is not None and new is not None
    old, new = _located(config, old), _located(config, new)
    kind = engine_kind(old)
    if engine_kind(new) != kind:
        raise PinRefused(
            f"{old_id} and {new_id} are profiles of different engines",
            hint="Bridge two profiles of the same engine.",
        )
    if old.canary is None:
        raise PinRefused(f"{old_id} has no canary to compare with", hint="Bridge from a profile that was pinned whole.")
    material = material if material is not None else find_canary(config)
    device = device if device is not None else config.gpu.device
    voice, transcript, seed = old.canary.clip, old.canary.transcript, old.canary.seed
    items: list[tuple[str, str, int]] = [("canary", material.gate_text, seed)]
    if kind == "base":
        vh = canary_voice_hash(voice.sha256, transcript)
        items += [
            (sid, text, keys.seed(voice_hash=vh, engine_text=text, attempt=0)) for sid, text in corpus_texts(config)
        ]
    sv = model_pin(config, WAVLM_SV)
    work = store.scratch_path(SCRATCH, uuid.uuid4().hex, "work").parent
    project = qwen_project(config)
    try:
        with _worker(starter, "qwen3", cublas=new.determinism.cublas_workspace_config) as qwen:
            not_runnable: dict[str, list[dict[str, Any]]] = {}
            for profile in (old, new):
                drift = DriftCheck().find(profile, qwen.hello, project=project)
                if drift:
                    not_runnable[profile.engine_profile_id] = [d.as_dict() for d in drift]
            if new_id in not_runnable:
                raise PinRefused(
                    f"{new_id} cannot render on this installation",
                    hint="Bridge to the profile this installation matches (narration-admin engine pin shows it).",
                    details={"drift": not_runnable[new_id]},
                )
            renders: dict[str, dict[str, Path]] = {}
            for profile in (old, new):
                if profile.engine_profile_id in not_runnable:
                    continue
                qwen.request("load", qwen_load_payload(profile, device), timeout_s=LOAD_TIMEOUT_S)
                out: dict[str, Path] = {}
                for name, text, s in items:
                    path = work / f"{profile.engine_profile_id}-{name}.wav"
                    if kind == "design":
                        design(
                            qwen, profile, description=material.description, text=material.design_text, seed=s, out=path
                        )
                    else:
                        clone(qwen, profile, clip=voice, transcript=transcript, text=text, seed=s, out=path)
                    out[name] = path
                renders[profile.engine_profile_id] = out
                qwen.request("unload", {}, timeout_s=UNLOAD_TIMEOUT_S)
        with _worker(starter, "qa") as qa:
            qa.request("load", sv_load_payload(sv), timeout_s=LOAD_TIMEOUT_S)
            report_items = [
                _compare(qa, name, renders.get(old_id, {}).get(name), renders[new_id][name], old, first=i == 0)
                for i, (name, _, _) in enumerate(items)
            ]
        return BridgeReport(old=old_id, new=new_id, items=tuple(report_items), not_runnable=not_runnable)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _located(config: Config, profile: EngineProfile) -> EngineProfile:
    """The profile with its snapshot folder where it is now. A profile's ``snapshot_dir`` is where its files
    were when it was pinned or last kept; after the models root moved, a profile no longer in use still names
    the old folder, so its revision's folder under ``[server] models_root`` is used when that exists (the
    fingerprint check then judges its files). A recorded folder that holds none of the profile's weight files
    counts as gone. Nothing is stored."""
    if _holds_weights(Path(profile.snapshot_dir), profile):
        return profile
    here = snapshot_dir(config.server.models_root, profile.model_repo, profile.model_revision)
    return dataclasses.replace(profile, snapshot_dir=str(here)) if _holds_weights(here, profile) else profile


def _holds_weights(folder: Path, profile: EngineProfile) -> bool:
    """Whether ``folder`` holds any of the profile's weight files: a folder that is gone, an empty skeleton or
    one with only an unfinished download does not."""
    return folder.is_dir() and any((folder / rel).is_file() for rel in profile.weights)


def _compare(
    qa: WorkerClient, name: str, old: Path | None, new: Path, old_profile: EngineProfile, *, first: bool
) -> BridgeItem:
    new_embedding = embed(qa, new)
    if old is not None:
        same = old.read_bytes() == new.read_bytes()
        return BridgeItem(
            item=name, similarity=round(cosine(embed(qa, old), new_embedding), 6), same_bytes=same, old_from="rendered"
        )
    if first and old_profile.canary is not None:  # the canary: the old side is its pinned render
        return BridgeItem(
            item=name,
            similarity=round(cosine(old_profile.canary.embedding, new_embedding), 6),
            same_bytes=None,
            old_from="pinned",
        )
    return BridgeItem(item=name, similarity=None, same_bytes=None, old_from="none")


__all__ = [
    "KEPT_UPDATES",
    "KINDS",
    "BridgeItem",
    "BridgeReport",
    "EngineReport",
    "PinRefused",
    "Plan",
    "Starter",
    "SupervisedStarter",
    "bridge",
    "corpus_texts",
    "pin",
    "plan",
]
