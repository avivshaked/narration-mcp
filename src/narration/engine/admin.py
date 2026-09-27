"""``narration-admin engine``: the operator's engine commands (design sections 7.1 and 10.1; plan.md WP32).

``register(subparsers)`` adds the ``engine`` command group to ``narration-admin``'s parser; WP37's dispatcher
(``narration.admin``) calls it with the other groups:

- ``engine pin``: record the engine profiles this installation matches and design the canary on this machine
  (``pinning``). A profile already pinned and still matched is kept; one that differs is refused, and the
  message says to re-pin.
- ``engine repin``: make a new profile the one in use for each engine whose installation or machine (GPU,
  driver, CUDA, cuDNN) differs; ``--force``: for both engines, whatever changed.
- ``engine bridge <old> <new>``: how similar the canary and the calibration corpus sound under two profiles.
- ``engine show``: the profiles pinned, which are in use, their tier and canary threshold.

**The interface is the dispatcher's** (``narration.admin.cli``): each command's ``handler(admin, args)``
returns the exit code, and takes the configuration, the platform and the store from ``admin``, which finds the
configuration by the one rule ``narration-mcp`` uses. A refusal the operator can act on is an ``AdminError``
that says what to do next; a ``NarrationError`` goes to the dispatcher, which prints its code and hint.
``--json`` prints the result for scripts.

``pin``, ``repin`` and ``bridge`` load Qwen, so they run only while no daemon holds the store (the daemon's
singleton, section 4) and only with the free VRAM a Qwen load needs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from narration.admin.cli import EXIT_OK, PROGRAM, Admin, AdminError, Handler, Subparsers
from narration.config import Config
from narration.contracts.errors import WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.serial import to_json
from narration.jobs.gpu import VramProbe
from narration.platform import ProcessPlatform

from .canary import engine_kind, gate_facts
from .installed import probe_for
from .pinning import BridgeReport, EngineReport, Mode, PinRefused, Starter, SupervisedStarter, bridge, pin
from .profile import FAMILIES, QWEN_PACKAGES, QWEN_VRAM_MB


@dataclass(frozen=True, slots=True, kw_only=True)
class Environment:
    """What the commands use of the machine beyond ``admin``; tests replace it (fake workers, no NVML).
    ``starter`` makes the pin's worker starter, a context manager that stops its workers on leaving."""

    starter: Callable[[Config, ProcessPlatform], AbstractContextManager[Starter]] = SupervisedStarter
    probe: Callable[[Config], VramProbe] = field(default=lambda config: probe_for(config.gpu.device))
    packages: Sequence[str] = QWEN_PACKAGES


ENVIRONMENT = Environment()
"""The environment ``register`` gives the commands when the dispatcher calls it (tests replace it)."""


def register(subparsers: Subparsers, env: Environment | None = None) -> None:
    """Add ``engine pin | repin | bridge | show`` to ``narration-admin``'s parser. Each command sets
    ``handler(admin, args) -> int``, as the dispatcher runs it."""
    env = env if env is not None else ENVIRONMENT
    engine = subparsers.add_parser(
        "engine",
        help="pin, re-pin or bridge the engine profile, and design the canary on this machine (section 10.1)",
        description=(
            "Pin the engine profiles this installation matches and design the service's canary on this "
            "machine; re-pin after a change; compare two profiles."
        ),
    )
    commands = engine.add_subparsers(dest="engine_command", required=True, metavar="{pin,repin,bridge,show}")

    pin_parser = commands.add_parser(
        "pin", help="record the engine profiles this installation matches, and design the canary on this machine"
    )
    pin_parser.add_argument("--json", action="store_true", help="print the result as JSON")
    pin_parser.set_defaults(handler=_pin_handler("pin", env))

    repin_parser = commands.add_parser(
        "repin", help="make a new engine profile the one in use where the installation changed (new render keys)"
    )
    repin_parser.add_argument(
        "--force",
        action="store_true",
        help="give both engines new profiles and fresh canaries even when nothing is seen to have changed "
        "(new render keys; voices are measured again)",
    )
    repin_parser.add_argument("--json", action="store_true", help="print the result as JSON")
    repin_parser.set_defaults(handler=_pin_handler("repin", env))

    bridge_parser = commands.add_parser(
        "bridge", help="compare how the canary and the calibration corpus sound under two engine profiles"
    )
    bridge_parser.add_argument("old", help="the engine profile in use now, e.g. qwen3-base-1.7b.p1")
    bridge_parser.add_argument("new", help="the engine profile to compare with, e.g. qwen3-base-1.7b.p2")
    bridge_parser.add_argument("--json", action="store_true", help="print the result as JSON")
    bridge_parser.set_defaults(handler=_bridge_handler(env))

    show_parser = commands.add_parser("show", help="the engine profiles pinned, and which are in use")
    show_parser.add_argument("--json", action="store_true", help="print the profiles as JSON")
    show_parser.set_defaults(handler=run_show)


# ======================================================================== the commands


def _pin_handler(mode: Mode, env: Environment) -> Handler:
    def handler(admin: Admin, args: argparse.Namespace) -> int:
        return run_pin(admin, args, mode=mode, env=env)

    return handler


def _bridge_handler(env: Environment) -> Handler:
    def handler(admin: Admin, args: argparse.Namespace) -> int:
        return run_bridge(admin, args, env=env)

    return handler


def run_pin(admin: Admin, args: argparse.Namespace, *, mode: Mode, env: Environment) -> int:
    """``engine pin`` or ``engine repin``: see the module docstring."""
    name = f"engine {mode}"
    config, platform = admin.config(), admin.platform()
    with platform.singleton(config.server.store_root) as held:
        if not held:
            raise _daemon_running(name)
        short = _short_of_vram(config, env)
        if short is not None:
            raise _refused(name, short, "Wait until the GPU has room (or free it), then run this again.")
        try:
            with env.starter(config, platform) as starter:
                reports = pin(
                    config,
                    admin.store(),
                    mode=mode,
                    starter=starter,
                    packages=env.packages,
                    force=bool(getattr(args, "force", False)),
                )
        except PinRefused as exc:
            raise _refused(name, exc.message, exc.hint) from exc
        except (WorkerFailure, WorkerCrashed, WorkerTimeout) as exc:
            raise _worker_failed(name, exc) from exc
    _print_pin(admin, reports, as_json=bool(args.json), mode=mode)
    return EXIT_OK


def run_bridge(admin: Admin, args: argparse.Namespace, *, env: Environment) -> int:
    """``engine bridge <old> <new>``: see the module docstring."""
    name = "engine bridge"
    config, platform = admin.config(), admin.platform()
    with platform.singleton(config.server.store_root) as held:
        if not held:
            raise _daemon_running(name)
        short = _short_of_vram(config, env)
        if short is not None:
            raise _refused(name, short, "Wait until the GPU has room (or free it), then run this again.")
        try:
            with env.starter(config, platform) as starter:
                report = bridge(config, admin.store(), args.old, args.new, starter=starter)
        except PinRefused as exc:
            raise _refused(name, exc.message, exc.hint) from exc
        except (WorkerFailure, WorkerCrashed, WorkerTimeout) as exc:
            raise _worker_failed(name, exc) from exc
    _print_bridge(admin, report, as_json=bool(args.json))
    return EXIT_OK


def run_show(admin: Admin, args: argparse.Namespace) -> int:
    """``engine show``: the profiles pinned, which are in use, their tier and canary threshold (and whether it is
    the floor). The calibration similarities are printed by ``engine pin`` and ``repin``; a ``CanaryPin`` does
    not record them. Creates no store."""
    rows: list[dict[str, Any]] = []
    if admin.store_exists():
        store = admin.store()
        current = {kind: store.current_engine_profile(kind) for kind in FAMILIES}
        rows = [
            {
                "engine_profile_id": p.engine_profile_id,
                "hash": p.hash,
                "in_use_for": next(
                    (k for k, c in current.items() if c and c.engine_profile_id == p.engine_profile_id), None
                ),
                "kind": engine_kind(p),
                "model": f"{p.model_repo}@{p.model_revision}",
                "tier": p.tier,
                "observed": to_json(p.observed),
                "canary": dict(gate_facts(p.canary)) if p.canary is not None else None,
            }
            for p in store.list_engine_profiles()
        ]
    if args.json:
        admin.say(json.dumps({"profiles": rows}, ensure_ascii=False, indent=2))
    elif not rows:
        admin.say(f"No engine profile is pinned. `{PROGRAM} engine pin` pins them.")
    else:
        for row in rows:
            use = f"in use for {row['in_use_for']}" if row["in_use_for"] else "not in use"
            canary = row["canary"]
            if canary is None:
                gate = "no canary"
            else:
                gate = f"canary threshold {canary['threshold']:.4f}"
                if canary["threshold_floored"]:
                    gate += " (the floor)"
            admin.say(f"{row['engine_profile_id']}  {row['hash']}  {use}  tier {row['tier']}  {gate}")
        if any(r["kind"] == "design" and r["canary"] is not None for r in rows):
            admin.say(
                "VoiceDesign's gate is a drift alarm: another seed designs another voice, so its threshold is low. "
                "Base's gate guards every take."
            )
        admin.say(f"Calibration similarities: `{PROGRAM} engine pin --json` prints them when it pins.")
    return EXIT_OK


# ======================================================================== helpers


def _short_of_vram(config: Config, env: Environment) -> str | None:
    reading = env.probe(config).read()
    need = QWEN_VRAM_MB + config.gpu.min_free_margin_mb
    if reading is None or reading.free_mb is None or reading.free_mb >= need:
        return None
    return f"the GPU has {reading.free_mb} MB free; loading Qwen needs {need} MB ({QWEN_VRAM_MB} plus the margin)"


def _daemon_running(name: str) -> AdminError:
    return _refused(
        name,
        "a narration daemon is running on this store, and it may load models on the GPU meanwhile",
        f"Stop it (`{PROGRAM} daemon stop`; it finishes the segment in flight), then run this again.",
    )


def _refused(name: str, message: str, hint: str) -> AdminError:
    """What the dispatcher prints (``narration-admin: engine pin: <message>. <hint>``), exit code 1."""
    return AdminError(f"{name}: {message.rstrip('.')}. {hint}")


def _worker_failed(name: str, exc: WorkerFailure | WorkerCrashed | WorkerTimeout) -> AdminError:
    if isinstance(exc, WorkerFailure):
        what = f"a worker replied {exc.code}: {exc.message}"
    elif isinstance(exc, WorkerCrashed):
        what = f"a worker stopped: {exc}"
    else:
        what = f"a worker did not answer in time: {exc}"
    return _refused(name, what, f"Run `{PROGRAM} doctor` to see what is missing; nothing was pinned.")


def _print_pin(admin: Admin, reports: Sequence[EngineReport], *, as_json: bool, mode: Mode = "pin") -> None:
    if as_json:
        admin.say(json.dumps({"engines": [r.as_dict() for r in reports]}, ensure_ascii=False, indent=2))
        return
    for r in reports:
        facts: list[str] = [r.action, r.hash]
        if r.tier is not None:
            facts.append(f"tier {r.tier}")
        if r.threshold is not None:
            facts.append(f"canary threshold {r.threshold:.4f}")
        if r.changed:
            facts.append("changed: " + ", ".join(r.changed))
        if r.updated:
            facts.append("updated: " + ", ".join(r.updated))
        if r.calibration:
            facts.append("calibration " + ", ".join(f"{c:.4f}" for c in r.calibration))
        admin.say(f"{r.engine_profile_id}  " + "  ".join(facts))
    if any(r.action == "new" and r.changed for r in reports):
        admin.say("New render keys: voices must be measured again under the new profile (measure_voice).")
    if mode == "repin" and all(r.action == "keep" for r in reports):
        admin.say(
            "Nothing changed in the installation or on this machine, so both profiles are kept. "
            f"`{PROGRAM} engine repin --force` re-pins both engines regardless (new render keys)."
        )


def _print_bridge(admin: Admin, report: BridgeReport, *, as_json: bool) -> None:
    data: dict[str, Any] = report.as_dict()
    if as_json:
        admin.say(json.dumps(data, ensure_ascii=False, indent=2))
        return
    admin.say(f"{report.old} -> {report.new}: lowest similarity {data['min_similarity']}")
    for item in report.items:
        same = "" if item.same_bytes is None else ("  same bytes" if item.same_bytes else "  different bytes")
        admin.say(f"  {item.item}: {item.similarity} ({item.old_from}){same}")
    for ident, drift in report.not_runnable.items():
        admin.say(f"  {ident} cannot render here: " + ", ".join(f"{d['what']} {d['name']}" for d in drift[:5]))


__all__ = [
    "ENVIRONMENT",
    "Environment",
    "register",
    "run_bridge",
    "run_pin",
    "run_show",
]
