"""``narration-admin engine``: the operator's engine commands (design sections 7.1 and 10.1; plan.md WP32).

``register(subparsers)`` adds the ``engine`` command group to ``narration-admin``'s parser (WP37's dispatcher
calls it with the other groups):

- ``engine pin``: record the engine profiles this installation matches and design the canary on this machine
  (``pinning``). A profile already pinned and still matched is kept; one that differs is refused, and the
  message says to re-pin.
- ``engine repin``: make a new profile the one in use for each engine whose installation differs.
- ``engine bridge <old> <new>``: how similar the canary and the calibration corpus sound under two profiles.
- ``engine show``: the profiles pinned, which are in use, their tier and canary.

Each command reads the configuration from ``--config`` or ``NARRATION_CONFIG``, prints what it did (``--json``
for machine-readable output), and returns an exit code: 0 done, 1 refused or failed (with what to do next), 2
a usage or configuration error. ``pin``, ``repin`` and ``bridge`` load Qwen on the GPU, so they run only while no
daemon holds the store (the daemon's singleton, section 4) and only with the free VRAM a Qwen load needs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from narration.config import Config, load_config
from narration.contracts.errors import ConfigError, NarrationError, WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.interfaces import Platform
from narration.contracts.serial import to_json
from narration.jobs.gpu import VramProbe
from narration.platform import get_platform
from narration.store import NarrationStore

from .canary import gate_facts
from .installed import probe_for
from .pinning import BridgeReport, EngineReport, Mode, PinRefused, Starter, SubprocessStarter, bridge, pin
from .profile import FAMILIES, QWEN_PACKAGES, QWEN_VRAM_MB

CONFIG_ENV: Final = "NARRATION_CONFIG"
"""The environment variable that names the configuration file when ``--config`` is not given."""
EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
EXIT_USAGE: Final = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class Environment:
    """What the commands use of the machine; tests replace it (fake workers, a stand-in platform, no NVML)."""

    platform: Callable[[], Platform] = get_platform
    starter: Callable[[Config], Starter] = SubprocessStarter
    probe: Callable[[Config], VramProbe] = field(default=lambda config: probe_for(config.gpu.device))
    packages: Sequence[str] = QWEN_PACKAGES


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], env: Environment | None = None) -> None:  # pyright: ignore[reportPrivateUsage]
    """Add ``engine pin | repin | bridge | show`` to ``narration-admin``'s parser. Each subcommand sets
    ``handler``, a function of the parsed arguments that returns the exit code."""
    env = env if env is not None else Environment()
    engine = subparsers.add_parser(
        "engine",
        help="pin the engine profiles and design the canary on this machine (design section 10.1)",
        description=__doc__.split("\n\n")[0] if __doc__ else None,
    )
    commands = engine.add_subparsers(dest="engine_command", required=True, metavar="{pin,repin,bridge,show}")

    pin_parser = commands.add_parser(
        "pin", help="record the engine profiles this installation matches, and design the canary on this machine"
    )
    _common(pin_parser)
    pin_parser.set_defaults(handler=lambda args: _run_pin(args, "pin", env))

    repin_parser = commands.add_parser(
        "repin", help="make a new engine profile the one in use where the installation changed (new render keys)"
    )
    _common(repin_parser)
    repin_parser.set_defaults(handler=lambda args: _run_pin(args, "repin", env))

    bridge_parser = commands.add_parser(
        "bridge", help="compare how the canary and the calibration corpus sound under two engine profiles"
    )
    bridge_parser.add_argument("old", help="the engine profile in use now, e.g. qwen3-base-1.7b.p1")
    bridge_parser.add_argument("new", help="the engine profile to compare with, e.g. qwen3-base-1.7b.p2")
    _common(bridge_parser)
    bridge_parser.set_defaults(handler=lambda args: _run_bridge(args, env))

    show_parser = commands.add_parser("show", help="the engine profiles pinned, and which are in use")
    _common(show_parser)
    show_parser.set_defaults(handler=lambda args: _run_show(args, env))


def _common(parser: argparse.ArgumentParser) -> None:
    # SUPPRESS: when not given here, a --config given to narration-admin itself is kept.
    parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help=f"the configuration file (design section 16); default: ${CONFIG_ENV}",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")


# ======================================================================== the commands


def _run_pin(args: argparse.Namespace, mode: Mode, env: Environment) -> int:
    name = f"engine {mode}"
    config = _config(args, name)
    if config is None:
        return EXIT_USAGE
    try:
        platform = env.platform()
        with (
            NarrationStore(config.server.store_root, platform, retention=config.retention) as store,
            platform.singleton(config.server.store_root) as held,
        ):
            if not held:
                return _daemon_running(name)
            short = _short_of_vram(config, env)
            if short is not None:
                return _fail(name, short, "Wait until the GPU has room (or release it), then run this again.")
            reports = pin(config, store, mode=mode, starter=env.starter(config), packages=env.packages)
    except PinRefused as exc:
        return _fail(name, exc.message, exc.hint)
    except NarrationError as exc:
        return _fail(name, f"{exc.code}: {exc.message}", exc.hint)
    except (WorkerFailure, WorkerCrashed, WorkerTimeout) as exc:
        return _worker_failed(name, exc)
    _print_pin(reports, as_json=bool(args.json))
    return EXIT_OK


def _run_bridge(args: argparse.Namespace, env: Environment) -> int:
    name = "engine bridge"
    config = _config(args, name)
    if config is None:
        return EXIT_USAGE
    try:
        platform = env.platform()
        with (
            NarrationStore(config.server.store_root, platform, retention=config.retention) as store,
            platform.singleton(config.server.store_root) as held,
        ):
            if not held:
                return _daemon_running(name)
            short = _short_of_vram(config, env)
            if short is not None:
                return _fail(name, short, "Wait until the GPU has room (or release it), then run this again.")
            report = bridge(config, store, args.old, args.new, starter=env.starter(config))
    except PinRefused as exc:
        return _fail(name, exc.message, exc.hint)
    except NarrationError as exc:
        return _fail(name, f"{exc.code}: {exc.message}", exc.hint)
    except (WorkerFailure, WorkerCrashed, WorkerTimeout) as exc:
        return _worker_failed(name, exc)
    _print_bridge(report, as_json=bool(args.json))
    return EXIT_OK


def _run_show(args: argparse.Namespace, env: Environment) -> int:
    name = "engine show"
    config = _config(args, name)
    if config is None:
        return EXIT_USAGE
    try:
        with NarrationStore(config.server.store_root, env.platform(), retention=config.retention) as store:
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
    except NarrationError as exc:
        return _fail(name, f"{exc.code}: {exc.message}", exc.hint)
    if args.json:
        print(json.dumps({"profiles": rows}, ensure_ascii=False, indent=2))
    elif not rows:
        print("no engine profile is pinned; run: narration-admin engine pin")
    else:
        for row in rows:
            use = f"in use for {row['in_use_for']}" if row["in_use_for"] else "not in use"
            print(f"{row['engine_profile_id']}  {row['hash']}  {use}  tier {row['tier']}")
    return EXIT_OK


# ======================================================================== helpers


def _config(args: argparse.Namespace, name: str) -> Config | None:
    raw = getattr(args, "config", None) or os.environ.get(CONFIG_ENV)
    if not raw:
        print(
            f"narration-admin {name}: no configuration file; pass --config <path> or set {CONFIG_ENV}",
            file=sys.stderr,
        )
        return None
    try:
        return load_config(Path(raw))
    except ConfigError as exc:
        print(f"narration-admin {name}: {exc}", file=sys.stderr)
        return None


def _short_of_vram(config: Config, env: Environment) -> str | None:
    reading = env.probe(config).read()
    need = QWEN_VRAM_MB + config.gpu.min_free_margin_mb
    if reading is None or reading.free_mb is None or reading.free_mb >= need:
        return None
    return f"the GPU has {reading.free_mb} MB free; loading Qwen needs {need} MB ({QWEN_VRAM_MB} plus the margin)"


def _daemon_running(name: str) -> int:
    return _fail(
        name,
        "a narration daemon is running on this store, and it may load models on the GPU meanwhile",
        "Stop it (narration-admin daemon stop; it finishes the segment in flight), then run this again.",
    )


def _fail(name: str, message: str, hint: str) -> int:
    print(f"narration-admin {name}: {message}", file=sys.stderr)
    print(f"next: {hint}", file=sys.stderr)
    return EXIT_FAILED


def _worker_failed(name: str, exc: WorkerFailure | WorkerCrashed | WorkerTimeout) -> int:
    if isinstance(exc, WorkerFailure):
        what = f"a worker replied {exc.code}: {exc.message}"
    elif isinstance(exc, WorkerCrashed):
        what = f"a worker stopped: {exc}"
    else:
        what = f"a worker did not answer in time: {exc}"
    return _fail(name, what, "Run narration-admin doctor to see what is missing; nothing was pinned.")


def _print_pin(reports: Sequence[EngineReport], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"engines": [r.as_dict() for r in reports]}, ensure_ascii=False, indent=2))
        return
    for r in reports:
        facts: list[str] = [r.action, r.hash]
        if r.tier is not None:
            facts.append(f"tier {r.tier}")
        if r.threshold is not None:
            facts.append(f"canary threshold {r.threshold:.4f}")
        if r.changed:
            facts.append("changed: " + ", ".join(r.changed))
        print(f"{r.engine_profile_id}  " + "  ".join(facts))
    if any(r.action == "new" and r.changed for r in reports):
        print("New render keys: voices must be measured again under the new profile (measure_voice).")


def _print_bridge(report: BridgeReport, *, as_json: bool) -> None:
    data: dict[str, Any] = report.as_dict()
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    print(f"{report.old} -> {report.new}: lowest similarity {data['min_similarity']}")
    for item in report.items:
        same = "" if item.same_bytes is None else ("  same bytes" if item.same_bytes else "  different bytes")
        print(f"  {item.item}: {item.similarity} ({item.old_from}){same}")
    for ident, drift in report.not_runnable.items():
        print(f"  {ident} cannot render here: " + ", ".join(f"{d['what']} {d['name']}" for d in drift[:5]))


__all__ = ["CONFIG_ENV", "EXIT_FAILED", "EXIT_OK", "EXIT_USAGE", "Environment", "register"]
