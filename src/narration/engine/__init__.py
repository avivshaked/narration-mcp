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
"""

from __future__ import annotations

from .profile import EngineSetupError, current_profile, current_ref, require_expected

__all__ = ["EngineSetupError", "current_profile", "current_ref", "require_expected"]
