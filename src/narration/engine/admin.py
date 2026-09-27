"""``narration-admin engine``: the operator's engine commands (design sections 7.1 and 10.1; plan.md WP32).

``register(subparsers)`` adds the ``engine`` command group to ``narration-admin``'s parser; WP37's dispatcher
(``narration.admin``) calls it with the other groups:

- ``engine pin``: record the engine profiles this installation matches and design the canary on this machine
  (``pinning``). A profile already pinned and still matched is kept; one that differs is refused, and the
  message says to re-pin.
- ``engine repin``: make a new profile the one in use for each engine whose installation differs.
- ``engine bridge <old> <new>``: how similar the canary and the calibration corpus sound under two profiles.
- ``engine show``: the profiles pinned, which are in use, their tier and canary.

**The interface is the dispatcher's** (``narration.admin.cli``): each command's ``handler(admin, args)``
returns the exit code (0 done, 1 refused or failed), and takes the configuration, the platform and the store
from ``admin``, which finds the configuration by the one rule ``narration-mcp`` uses. A ``NarrationError``
goes to the dispatcher, which prints its code and hint. A refusal the operator can act on is printed here,
with what to do next. ``--json`` prints the result for scripts.

``pin``, ``repin`` and ``bridge`` load Qwen, so they run only while no daemon holds the store (the daemon's
singleton, section 4) and only with the free VRAM a Qwen load needs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from narration.config import Config
from narration.contracts.errors import WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.serial import to_json
from narration.jobs.gpu import VramProbe
from narration.platform import ProcessPlatform
from narration.store import NarrationStore

from .canary import gate_facts
from .installed import probe_for
from .pinning import BridgeReport, EngineReport, Mode, PinRefused, Starter, SupervisedStarter, bridge, pin
from .profile import FAMILIES, QWEN_PACKAGES, QWEN_VRAM_MB

PROGRAM: Final = "narration-admin"
EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
"""The dispatcher's exit codes (``narration.admin.cli``) that these commands return."""


class Subparsers(Protocol):
    """What ``register`` is given: what ``ArgumentParser.add_subparsers`` returns."""

    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Add the parser of the command ``name``."""
        ...


class AdminContext(Protocol):
    """What the dispatcher hands each command (``narration.admin.cli.Admin``): the configuration, loaded on
    first use; this OS's platform; the store, opened once; and the output streams."""

    def config(self) -> Config:
        """The configuration (the dispatcher's error, with what to do, when there is none)."""
        ...

    def platform(self) -> ProcessPlatform:
        """This OS's ``Platform``."""
        ...

    def store_exists(self) -> bool:
        """Whether the configured store has a database yet."""
        ...

    def store(self) -> NarrationStore:
        """The store at ``[server] store_root``, opened once."""
        ...

    def say(self, text: str = "") -> None:
        """Print a line of the command's output (stdout)."""
        ...

    def warn(self, text: str) -> None:
        """Print a line to stderr."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Environment:
    """What the commands use of the machine beyond ``admin``; tests replace it (fake workers, no NVML).
    ``starter`` makes the pin's worker starter, a context manager that stops its workers on leaving."""

    starter: Callable[[Config, ProcessPlatform], AbstractContextManager[Starter]] = SupervisedStarter
    probe: Callable[[Config], VramProbe] = field(default=lambda config: probe_for(config.gpu.device))
    packages: Sequence[str] = QWEN_PACKAGES


Handler = Callable[[AdminContext, argparse.Namespace], int]


def register(subparsers: Subparsers, env: Environment | None = None) -> None:
    """Add ``engine pin | repin | bridge | show`` to ``narration-admin``'s parser. Each command sets
    ``handler(admin, args) -> int``, as the dispatcher runs it."""
    env = env if env is not None else Environment()
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
    def handler(admin: AdminContext, args: argparse.Namespace) -> int:
        return run_pin(admin, args, mode=mode, env=env)

    return handler


def _bridge_handler(env: Environment) -> Handler:
    def handler(admin: AdminContext, args: argparse.Namespace) -> int:
        return run_bridge(admin, args, env=env)

    return handler


def run_pin(admin: AdminContext, args: argparse.Namespace, *, mode: Mode, env: Environment) -> int:
    """``engine pin`` or ``engine repin``: see the module docstring."""
    name = f"engine {mode}"
    config, platform = admin.config(), admin.platform()
    with platform.singleton(config.server.store_root) as held:
        if not held:
            return _daemon_running(admin, name)
        short = _short_of_vram(config, env)
        if short is not None:
            return _fail(admin, name, short, "Wait until the GPU has room (or free it), then run this again.")
        try:
            with env.starter(config, platform) as starter:
                reports = pin(config, admin.store(), mode=mode, starter=starter, packages=env.packages)
        except PinRefused as exc:
            return _fail(admin, name, exc.message, exc.hint)
        except (WorkerFailure, WorkerCrashed, WorkerTimeout) as exc:
            return _worker_failed(admin, name, exc)
    _print_pin(admin, reports, as_json=bool(args.json))
    return EXIT_OK


def run_bridge(admin: AdminContext, args: argparse.Namespace, *, env: Environment) -> int:
    """``engine bridge <old> <new>``: see the module docstring."""
    name = "engine bridge"
    config, platform = admin.config(), admin.platform()
    with platform.singleton(config.server.store_root) as held:
        if not held:
            return _daemon_running(admin, name)
        short = _short_of_vram(config, env)
        if short is not None:
            return _fail(admin, name, short, "Wait until the GPU has room (or free it), then run this again.")
        try:
            with env.starter(config, platform) as starter:
                report = bridge(config, admin.store(), args.old, args.new, starter=starter)
        except PinRefused as exc:
            return _fail(admin, name, exc.message, exc.hint)
        except (WorkerFailure, WorkerCrashed, WorkerTimeout) as exc:
            return _worker_failed(admin, name, exc)
    _print_bridge(admin, report, as_json=bool(args.json))
    return EXIT_OK


def run_show(admin: AdminContext, args: argparse.Namespace) -> int:
    """``engine show``: the profiles pinned, which are in use, their tier and canary. Creates no store."""
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
        admin.say(f"No engine profile is pinned. `{PROGRAM} engine pin` pins one.")
    else:
        for row in rows:
            use = f"in use for {row['in_use_for']}" if row["in_use_for"] else "not in use"
            admin.say(f"{row['engine_profile_id']}  {row['hash']}  {use}  tier {row['tier']}")
    return EXIT_OK


# ======================================================================== helpers


def _short_of_vram(config: Config, env: Environment) -> str | None:
    reading = env.probe(config).read()
    need = QWEN_VRAM_MB + config.gpu.min_free_margin_mb
    if reading is None or reading.free_mb is None or reading.free_mb >= need:
        return None
    return f"the GPU has {reading.free_mb} MB free; loading Qwen needs {need} MB ({QWEN_VRAM_MB} plus the margin)"


def _daemon_running(admin: AdminContext, name: str) -> int:
    return _fail(
        admin,
        name,
        "a narration daemon is running on this store, and it may load models on the GPU meanwhile",
        f"Stop it ({PROGRAM} daemon stop; it finishes the segment in flight), then run this again.",
    )


def _fail(admin: AdminContext, name: str, message: str, hint: str) -> int:
    admin.warn(f"{PROGRAM} {name}: {message}")
    admin.warn(f"next: {hint}")
    return EXIT_FAILED


def _worker_failed(admin: AdminContext, name: str, exc: WorkerFailure | WorkerCrashed | WorkerTimeout) -> int:
    if isinstance(exc, WorkerFailure):
        what = f"a worker replied {exc.code}: {exc.message}"
    elif isinstance(exc, WorkerCrashed):
        what = f"a worker stopped: {exc}"
    else:
        what = f"a worker did not answer in time: {exc}"
    return _fail(admin, name, what, f"Run {PROGRAM} doctor to see what is missing; nothing was pinned.")


def _print_pin(admin: AdminContext, reports: Sequence[EngineReport], *, as_json: bool) -> None:
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
        admin.say(f"{r.engine_profile_id}  " + "  ".join(facts))
    if any(r.action == "new" and r.changed for r in reports):
        admin.say("New render keys: voices must be measured again under the new profile (measure_voice).")


def _print_bridge(admin: AdminContext, report: BridgeReport, *, as_json: bool) -> None:
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
    "EXIT_FAILED",
    "EXIT_OK",
    "AdminContext",
    "Environment",
    "Handler",
    "Subparsers",
    "register",
    "run_bridge",
    "run_pin",
    "run_show",
]
