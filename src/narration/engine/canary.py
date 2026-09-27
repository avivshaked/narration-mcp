"""The service's canary and the canary gate (design section 10.1; plan.md DC-3, WP32).

**The canary** is the service's own voice, designed **on the installing machine** by ``narration-admin engine
pin`` from text that ships with the source (``material/canary/canary.v1``): a positive-only description, the
design text and a seed. No canary audio ships, since nothing is promised across a GPU, driver or CUDA change.
Each engine profile keeps its canary (``EngineProfile.canary``, a ``CanaryPin``): the clip, the seed of its
gate render, that render's raw sha256 and speaker embedding, and a calibrated similarity threshold.

- For the **Base** profile the gate render clones the canary clip (ICL, its transcript is the design text)
  speaking the gate's fixed text with the gate's seed.
- For the **VoiceDesign** profile the gate render is the canary design itself: the description and design
  text with the design seed. Its raw sha256 is the clip's.

**The gate** (``CanaryGuard``, the job engine's ``EngineGuard``) runs after every Qwen load, before anything
renders on it:

1. the worker fingerprint check (``drift``): any difference is ``ENGINE_DRIFT``;
2. the gate render (a few seconds);
3. **hash equal**: go on (``hash_match``), with no QA and no model swap;
4. **hash different**: the QA worker embeds the render with WavLM **on the CPU** (the ``sv`` model only, while
   Qwen stays on the GPU), and its cosine similarity to the pinned embedding is compared with the pinned
   threshold. At or above it: go on (``similarity_pass``; in the ``bit_exact`` tier every take of the batch
   is flagged ``CANARY_MISMATCH``, info). Below it: ``ENGINE_DRIFT``, and the job fails before it renders.

A canary that cannot be rendered or measured fails the job with ``ENGINE_DRIFT`` too (the engine cannot be
vouched for), except running out of GPU memory, a worker that is not installed, and a crash, which the job
engine handles as it handles any render's (section 4 item 5, section 14).

**The threshold** is calibrated at the pin (``calibrate``): the canary is rendered with ``CALIBRATION_SEEDS``
more seeds, each is compared with the pinned render, and the threshold is the lowest of those similarities
less ``CANARY_MARGIN``, and never below ``THRESHOLD_FLOOR``. A render whose numbers changed (a driver update)
diverges like another seed would, so it passes; one that differs more than another seed does fails. For Base
those seeds read the same text in the same voice; for VoiceDesign another seed designs another voice, so its
threshold is lower and its gate weaker. KNOW (the first real pin, 2026-09-27): Base 0.973, VoiceDesign 0.331.
The VoiceDesign gate is kept as a drift alarm in v1 (the lead's decision; section 1: not an identity proof).

A pinned embedding that is empty, all zero or not finite, or whose length differs from the render's, is
``ENGINE_DRIFT`` with the reason in ``details.canary``: the gate never passes or fails on a similarity that
could not be computed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from narration import keys
from narration.config import Config
from narration.contracts import codes, names
from narration.contracts.errors import MaterialError, NarrationError, WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.interfaces import WorkerClient
from narration.contracts.models import AudioRef, CanaryMaterial, CanaryPin, EngineProfile
from narration.contracts.names import CanaryStatus, EngineKind
from narration.contracts.worker import HelloReply
from narration.jobs.core import worker_code
from narration.jobs.gpu import LOAD_TIMEOUT_S, UNLOAD_TIMEOUT_S
from narration.jobs.host import RunnerHost
from narration.jobs.pins import ModelPin, call_cap

from .drift import DriftCheck
from .profile import FAMILIES, EngineSetupError, qwen_project

log = logging.getLogger(__name__)

CANARY_SET: Final = "canary.v1"
"""The canary's material set (``material/canary/canary.v1``)."""
CALIBRATION_SEEDS: Final = 3
"""How many more seeds the pin renders the canary with to calibrate its threshold (ASSUME: enough to see the
spread of one text in one voice; more cost a few seconds each)."""
CANARY_MARGIN: Final = 0.01
"""How far below the lowest calibration similarity the threshold sits (ASSUME; section 16's
``sim_warn_margin`` is the same size)."""
THRESHOLD_FLOOR: Final = 0.10
"""The lowest threshold a calibration may set (ASSUME). Without it a canary whose calibration renders share
nothing with it would get a threshold at or below zero, and every render would pass the gate. On this
project's first real pin (2026-09-27) VoiceDesign calibrated to 0.33 and Base to 0.97, both above it."""
X_VECTOR_ONLY_MODE: Final = False
"""The canary is cloned in ICL mode, as every voice is (section 10.1)."""
RENDER_TIMEOUT_S: Final = 600.0
PREPARE_TIMEOUT_S: Final = 300.0
EMBED_TIMEOUT_S: Final = 300.0
SCRATCH: Final = "canary"
"""The gate's renders go to ``scratch/canary/`` and are removed once measured."""
REPIN_HINT: Final = "Ask the operator to re-pin the engine (narration-admin engine repin); nothing was rendered."
"""What to do about a drift that ``engine repin`` sees (the canary's text changed): it pins the engine anew."""
FORCE_HINT: Final = (
    "Ask the operator to re-pin the engine with narration-admin engine repin --force (new render keys; voices "
    "are measured again): nothing in the installation or on this machine is seen to have changed, so a plain "
    "repin keeps this profile. Nothing was rendered."
)
"""What to do about a drift that ``engine repin`` cannot see (the gate render moved, or cannot be compared)."""
WORKER_HINT: Final = (
    "Ask the operator to run narration-admin doctor; if the workers are sound, narration-admin engine repin "
    "--force re-pins the engine (a plain repin keeps a profile whose installation and machine are unchanged). "
    "Nothing was rendered."
)
"""What to do when a worker fails during the gate: check the installation first, then re-pin by force."""


# ======================================================================== the canary's text (material/)


def material_roots(config: Config | None = None) -> list[Path]:
    """Where the service's own material may be: next to this source checkout, then next to the config file."""
    roots = [Path(__file__).resolve().parents[3] / "material"]
    if config is not None and config.service_root is not None:
        roots.append(config.service_root / "material")
    return roots


def read_set(root: Path, kind: str, set_id: str) -> tuple[dict[str, Any], str, dict[str, bytes]]:
    """A material set's manifest, the manifest's sha256 and every file it lists, each checked against the
    manifest (``MaterialError`` when a file is missing, malformed or not the bytes the manifest lists)."""
    folder = root / kind / set_id
    try:
        manifest_bytes = (folder / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        files: dict[str, bytes] = {}
        for entry in manifest["files"]:
            data = (folder / entry["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise MaterialError(f"{folder / entry['path']} is not the file its manifest lists")
            files[str(entry["path"])] = data
    except MaterialError:
        raise
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise MaterialError(f"the material set {folder} cannot be read: {exc}") from exc
    return manifest, hashlib.sha256(manifest_bytes).hexdigest(), files


def load_canary(root: Path, set_id: str = CANARY_SET) -> CanaryMaterial:
    """The canary's text, checked against its manifest (``read_set``). ``sha256`` is the manifest's, so it
    names the set's exact content."""
    manifest, sha, files = read_set(root, "canary", set_id)
    try:
        canary = json.loads(files["canary.json"].decode("utf-8"))
        voice, gate = canary["voice"], canary["gate"]
        return CanaryMaterial(
            set_id=str(manifest["set"]),
            status=manifest["status"],
            sha256=sha,
            description=str(voice["description"]),
            design_text=str(voice["design_text"]),
            design_seed=int(voice["seed"]),
            gate_text=str(gate["text"]),
            gate_seed=int(gate["seed"]),
        )
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise MaterialError(f"the canary material {root / 'canary' / set_id} cannot be read: {exc}") from exc


def find_set(config: Config | None, kind: str, set_id: str) -> Path:
    """The first material folder (``material_roots``) that has the set; ``BACKEND_NOT_INSTALLED`` when none
    does, since the service's own files are then incomplete."""
    for root in material_roots(config):
        if (root / kind / set_id / "manifest.json").is_file():
            return root
    raise EngineSetupError(
        f"the service's material set {kind}/{set_id} is missing",
        hint="Restore the service's files (a fresh checkout of the same version); nothing was rendered.",
    )


def find_canary(config: Config | None = None, set_id: str = CANARY_SET) -> CanaryMaterial:
    """The canary's text from the first material folder that has it; a missing or damaged set is
    ``BACKEND_NOT_INSTALLED``."""
    root = find_set(config, "canary", set_id)
    try:
        return load_canary(root, set_id)
    except MaterialError as exc:
        raise EngineSetupError(
            f"the service's canary material is damaged: {exc}",
            hint="Restore the service's files (a fresh checkout of the same version); nothing was rendered.",
        ) from exc


def material_id(material: CanaryMaterial) -> str:
    """How a ``CanaryPin`` names its material: the set id and its manifest's sha256."""
    return f"{material.set_id}@{names.HASH_PREFIX}{material.sha256}"


# ======================================================================== rendering and measuring


def engine_kind(profile: EngineProfile) -> EngineKind:
    """Which engine a profile is, by its id's family (``qwen3-base-1.7b.`` or ``qwen3-design-1.7b.``)."""
    for kind, family in FAMILIES.items():
        if profile.engine_profile_id.startswith(family + "."):
            return kind
    raise EngineSetupError(
        f"{profile.engine_profile_id} is not a profile of a known engine ({', '.join(FAMILIES.values())})",
        hint="Ask the operator to re-pin the engine (narration-admin engine repin).",
    )


def canary_voice_hash(clip_sha256: str, transcript: str) -> str:
    """The canary clip's voice hash (section 10.2), as ``prepare_voice`` names it."""
    return keys.voice_hash(
        model=names.MODEL_QWEN_BASE,
        clip_sha256=clip_sha256,
        transcript=transcript,
        language=names.LANGUAGE,
        x_vector_only_mode=X_VECTOR_ONLY_MODE,
    )


def sha256_file(path: Path) -> str:
    """A file's sha256, 64 hex."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def render(
    qwen: WorkerClient,
    profile: EngineProfile,
    material: CanaryMaterial,
    *,
    seed: int,
    out: Path,
    clip: AudioRef | None = None,
    transcript: str | None = None,
) -> str:
    """Render the canary on a loaded Qwen worker; returns the raw WAV's sha256 (the worker's file repeats byte
    for byte, ADR 0002). For VoiceDesign: design from the description and design text. For Base: prepare
    ``clip`` (its ``transcript``, the design text by default) and speak the gate text. Every call passes its
    own cap (DC-4)."""
    if engine_kind(profile) == "design":
        return design(qwen, profile, description=material.description, text=material.design_text, seed=seed, out=out)
    if clip is None:
        raise ValueError("the Base canary clones the canary clip: pass clip")
    ref_text = transcript if transcript is not None else material.design_text
    return clone(qwen, profile, clip=clip, transcript=ref_text, text=material.gate_text, seed=seed, out=out)


def design(qwen: WorkerClient, profile: EngineProfile, *, description: str, text: str, seed: int, out: Path) -> str:
    """Design a voice speaking ``text`` on a loaded VoiceDesign worker; the raw WAV's sha256."""
    out.parent.mkdir(parents=True, exist_ok=True)
    qwen.request(
        "design",
        {
            "description": description,
            "design_text": text,
            "language": names.LANGUAGE,
            "seed": seed,
            "max_new_tokens": call_cap(profile, text),
            "out_path": str(out),
        },
        timeout_s=RENDER_TIMEOUT_S,
    )
    return sha256_file(out)


def clone(
    qwen: WorkerClient, profile: EngineProfile, *, clip: AudioRef, transcript: str, text: str, seed: int, out: Path
) -> str:
    """Clone ``clip`` (ICL, with its ``transcript``) speaking ``text`` on a loaded Base worker; the raw WAV's
    sha256. The voice is prepared each time: a load clears the worker's prepared voices."""
    out.parent.mkdir(parents=True, exist_ok=True)
    voice_hash = canary_voice_hash(clip.sha256, transcript)
    qwen.request(
        "prepare_voice",
        {
            "voice_hash": voice_hash,
            "ref_wav": clip.path,
            "ref_text": transcript,
            "x_vector_only_mode": X_VECTOR_ONLY_MODE,
        },
        timeout_s=PREPARE_TIMEOUT_S,
    )
    qwen.request(
        "synthesize",
        {
            "voice_hash": voice_hash,
            "engine_text": text,
            "language": names.LANGUAGE,
            "seed": seed,
            "max_new_tokens": call_cap(profile, text),
            "out_path": str(out),
        },
        timeout_s=RENDER_TIMEOUT_S,
    )
    return sha256_file(out)


def sv_load_payload(sv: ModelPin) -> dict[str, Any]:
    """``load`` for the QA worker with WavLM-SV alone, on the CPU: the canary never swaps Qwen out (10.1)."""
    return {"device": "cpu", "models": {"sv": sv.ref()}}


def embed(qa: WorkerClient, wav: Path) -> tuple[float, ...]:
    """WavLM-SV's embedding of ``wav``, computed on the CPU."""
    reply = qa.request("embed", {"wav": str(wav), "device": "cpu"}, timeout_s=EMBED_TIMEOUT_S)
    return tuple(float(v) for v in reply["embedding"])


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """The cosine similarity of two embeddings (0.0 when either is zero or they differ in length)."""
    if len(a) != len(b):
        return 0.0
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def calibration_seeds(seed: int) -> tuple[int, ...]:
    """The seeds the pin renders the canary with to calibrate its threshold: the next ``CALIBRATION_SEEDS``."""
    return tuple((seed + k) & keys.SEED_MASK for k in range(1, CALIBRATION_SEEDS + 1))


def calibrate(pinned: Sequence[float], others: Sequence[Sequence[float]]) -> tuple[float, tuple[float, ...]]:
    """The threshold from the calibration renders' embeddings: the lowest similarity to the pinned render's,
    less ``CANARY_MARGIN``, and at least ``THRESHOLD_FLOOR``; and each similarity (rounded to 6 places).
    Whether the floor applied is ``floored(threshold)``."""
    if not others:
        raise ValueError("calibrating the canary needs at least one more render")
    sims = tuple(round(cosine(pinned, o), 6) for o in others)
    return max(round(min(sims) - CANARY_MARGIN, 6), THRESHOLD_FLOOR), sims


def floored(threshold: float) -> bool:
    """Whether a pinned threshold is the floor rather than a calibrated value. ``CanaryPin`` has no field
    for it, so it is read from the threshold itself (a calibration landing exactly on the floor counts as
    floored, which says the same thing: the gate is as weak as it may be)."""
    return threshold <= THRESHOLD_FLOOR


def embedding_problem(embedding: Sequence[float]) -> str | None:
    """Why an embedding cannot be compared: ``empty``, ``not_finite`` or ``all_zero``; None when it can."""
    if not embedding:
        return "empty"
    if not all(math.isfinite(v) for v in embedding):
        return "not_finite"
    if not any(v != 0.0 for v in embedding):
        return "all_zero"
    return None


# ======================================================================== the gate


class CanaryGuard:
    """The job engine's ``EngineGuard`` (``narration.jobs.hooks``): the fingerprint check, then the canary gate
    (see the module docstring). One instance serves one daemon; it keeps the file hashes of the drift check."""

    def __init__(
        self, *, sv: ModelPin, drift: DriftCheck | None = None, material: CanaryMaterial | None = None
    ) -> None:
        self.sv = sv
        self.drift = drift if drift is not None else DriftCheck()
        self._material = material

    def after_load(self, host: RunnerHost, profile: EngineProfile, hello: HelloReply | None) -> CanaryStatus:
        """Check the Qwen engine just loaded; the batch's canary outcome, or ``NarrationError(ENGINE_DRIFT)``."""
        self.drift.require_none(profile, hello, project=qwen_project(host.config))
        pin = profile.canary
        if pin is None:
            raise EngineSetupError(
                f"engine profile {profile.engine_profile_id} has no canary",
                hint=REPIN_HINT,
            )
        material = self.material(host.config)
        if material_id(material) != pin.material:
            raise NarrationError(
                codes.ENGINE_DRIFT,
                f"the canary's text is {material_id(material)}, but engine profile {profile.engine_profile_id} "
                f"was pinned with {pin.material}",
                hint=REPIN_HINT,
                details={"engine_profile_id": profile.engine_profile_id, "canary": "material", "pinned": pin.material},
                retryable=False,
            )
        problem = embedding_problem(pin.embedding)
        if problem is not None:
            raise NarrationError(
                codes.ENGINE_DRIFT,
                f"the pinned canary embedding of engine profile {profile.engine_profile_id} cannot be compared "
                f"({problem.replace('_', ' ')}), so the engine cannot be vouched for",
                hint=FORCE_HINT,
                details={"engine_profile_id": profile.engine_profile_id, "canary": f"pinned_embedding_{problem}"},
                retryable=False,
            )
        out = host.store.scratch_path(SCRATCH, f"{profile.engine_profile_id}-{uuid.uuid4().hex}.wav")
        qwen = host.workers.client("qwen", cublas_workspace_config=profile.determinism.cublas_workspace_config)
        try:
            try:
                sha = render(qwen, profile, material, seed=pin.seed, out=out, clip=pin.clip, transcript=pin.transcript)
            except WorkerFailure as exc:
                raise _unless_handled(exc, profile, "rendered") from exc
            if sha == pin.raw_sha256:
                log.info("canary of %s: the hash matches", profile.engine_profile_id)
                return "hash_match"
            similarity = self._similarity(host, out, pin, profile)
            facts = {
                "engine_profile_id": profile.engine_profile_id,
                "tier": profile.tier,
                "raw_sha256": sha,
                "pinned_sha256": pin.raw_sha256,
                "similarity": round(similarity, 6),
                "threshold": pin.threshold,
            }
            if similarity >= pin.threshold:
                log.warning(
                    "canary of %s: the hash differs, the similarity passes: %s", profile.engine_profile_id, facts
                )
                return "similarity_pass"
            raise NarrationError(
                codes.ENGINE_DRIFT,
                f"the canary of engine profile {profile.engine_profile_id} is {similarity:.4f} similar to its pinned "
                f"render, below its threshold {pin.threshold:.4f}",
                hint=FORCE_HINT,
                details={**facts, "canary": "below_threshold"},
                retryable=False,
            )
        finally:
            out.unlink(missing_ok=True)

    def material(self, config: Config) -> CanaryMaterial:
        """The canary's text, read once."""
        if self._material is None:
            self._material = find_canary(config)
        return self._material

    def _similarity(self, host: RunnerHost, wav: Path, pin: CanaryPin, profile: EngineProfile) -> float:
        """The render's similarity to the pinned embedding, with WavLM on the QA worker's CPU. The QA worker's
        models are unloaded again afterwards, so the scheduler's view of the GPU is unchanged; if the unload
        fails, the QA worker is stopped, so no worker is left holding a model the residency does not know."""
        pool = host.workers
        try:
            try:
                pool.load("qa", sv_load_payload(self.sv), gpu=False, timeout_s=LOAD_TIMEOUT_S)
                embedding = embed(pool.client("qa"), wav)
            except WorkerFailure as exc:
                raise _unless_handled(exc, profile, "measured") from exc
        finally:
            try:
                pool.unload("qa", timeout_s=UNLOAD_TIMEOUT_S)
            except (WorkerFailure, WorkerCrashed, WorkerTimeout) as exc:
                log.warning("unloading the QA worker after the canary failed (%s); stopping it", exc)
                pool.stop("qa")
        problem = embedding_problem(embedding)
        if problem is None and len(embedding) != len(pin.embedding):
            problem = "length"
        if problem is not None:
            raise NarrationError(
                codes.ENGINE_DRIFT,
                f"the canary render of engine profile {profile.engine_profile_id} cannot be compared with its "
                f"pinned embedding ({problem.replace('_', ' ')}: {len(embedding)} values, pinned "
                f"{len(pin.embedding)})",
                hint=WORKER_HINT,
                details={
                    "engine_profile_id": profile.engine_profile_id,
                    "canary": f"embedding_{problem}",
                    "dim": len(embedding),
                    "pinned_dim": len(pin.embedding),
                },
                retryable=False,
            )
        return cosine(embedding, pin.embedding)


def _unless_handled(exc: WorkerFailure, profile: EngineProfile | None, done: str) -> Exception:
    """A worker's failure during the canary: running out of GPU memory and a worker that is not installed go
    on to the job engine's own handling; anything else means the engine cannot be vouched for
    (``ENGINE_DRIFT``)."""
    code = worker_code(exc)
    if code in (codes.GPU_OOM, codes.BACKEND_NOT_INSTALLED):
        return exc
    details: dict[str, Any] = {"canary": f"not_{done}", "worker_code": exc.code}
    if profile is not None:
        details["engine_profile_id"] = profile.engine_profile_id
    return NarrationError(
        codes.ENGINE_DRIFT,
        f"the canary could not be {done}: {exc.message}",
        hint=WORKER_HINT,
        details=details,
        retryable=False,
    )


def pin_record(
    *,
    material: CanaryMaterial,
    clip: AudioRef,
    transcript: str,
    seed: int,
    raw_sha256: str,
    embedding: Sequence[float],
    threshold: float,
    pinned_at: str,
) -> CanaryPin:
    """The ``CanaryPin`` the pin stores with a profile."""
    return CanaryPin(
        material=material_id(material),
        clip=clip,
        transcript=transcript,
        seed=seed,
        raw_sha256=raw_sha256,
        embedding=tuple(float(v) for v in embedding),
        threshold=threshold,
        pinned_at=pinned_at,
    )


def gate_facts(pin: CanaryPin) -> Mapping[str, Any]:
    """What an operator reads of a profile's canary (``narration-admin engine show``)."""
    return {
        "material": pin.material,
        "clip_sha256": pin.clip.sha256,
        "seed": pin.seed,
        "raw_sha256": pin.raw_sha256,
        "threshold": pin.threshold,
        "threshold_floored": floored(pin.threshold),
        "embedding_dim": len(pin.embedding),
        "pinned_at": pin.pinned_at,
    }


__all__ = [
    "CALIBRATION_SEEDS",
    "CANARY_MARGIN",
    "CANARY_SET",
    "FORCE_HINT",
    "REPIN_HINT",
    "THRESHOLD_FLOOR",
    "WORKER_HINT",
    "CanaryGuard",
    "calibrate",
    "calibration_seeds",
    "canary_voice_hash",
    "cosine",
    "embed",
    "embedding_problem",
    "engine_kind",
    "find_canary",
    "floored",
    "gate_facts",
    "load_canary",
    "material_id",
    "material_roots",
    "pin_record",
    "render",
    "sha256_file",
    "sv_load_payload",
]
