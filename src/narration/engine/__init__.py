"""Engine profiles, the worker fingerprint check and the canary gate (design sections 6, 10.1 and 10.2; plan.md
WP32, DC-3).

- ``models``: the models the service pins, by repo and 40-hex revision, and where their snapshots live.
- ``profile``: building and hashing an engine profile, and the profile in use (``current_profile``, which the
  render keys, measurements and ``expect_engine_profile`` read).
- ``drift``: the worker fingerprint check after a load (``ENGINE_DRIFT``).
- ``canary``: the service's canary and the canary gate, the job engine's ``EngineGuard``.
- ``qa``: the QA group's pins and the cue aligner as this installation provides them.
- ``installed``: the job engine the daemon builds (``narration.jobs.runner.installed_engine``).
- ``pinning``: ``narration-admin engine pin | repin | bridge``.
- ``admin``: those commands, as ``narration-admin``'s ``engine`` group.

The names below load ``profile`` on first use, so importing ``narration.engine.models`` alone (as
``narration-admin``'s model checks do) stays light: it needs only the contracts, not the store or numpy.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .profile import EngineSetupError, current_profile, current_ref, require_expected

__all__ = ["EngineSetupError", "current_profile", "current_ref", "require_expected"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(importlib.import_module(".profile", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
