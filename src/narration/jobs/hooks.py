"""Where the engine checks the engine it loaded (design section 10.1; WP32 fills these in).

After Qwen is loaded for a batch, the engine calls ``EngineGuard.after_load``. WP32's guard checks the
worker's fingerprint against the pinned engine profile (``ENGINE_DRIFT``) and runs the **canary gate** on the
service's own canary (DC-3): hash equal, go on (``hash_match``); hash different but similar enough, go on and
flag every take of the batch ``CANARY_MISMATCH`` (``similarity_pass``); otherwise raise
``NarrationError(ENGINE_DRIFT)``, and the job fails before it renders anything. The guard may use the host's
workers (the canary render on Qwen, the similarity on the QA worker's CPU).

Until WP32 is built, ``NoGuard`` checks nothing and reports ``not_run``, which every render then records.
"""

from __future__ import annotations

from typing import Protocol

from narration.contracts.models import EngineProfile
from narration.contracts.names import CanaryStatus
from narration.contracts.worker import HelloReply

from .host import RunnerHost


class EngineGuard(Protocol):
    """Checks a freshly loaded Qwen engine before a batch renders on it (see the module docstring)."""

    def after_load(self, host: RunnerHost, profile: EngineProfile, hello: HelloReply | None) -> CanaryStatus:
        """The batch's canary outcome; raises ``NarrationError(ENGINE_DRIFT)`` when the engine drifted."""
        ...


class NoGuard:
    """The guard until WP32: no fingerprint check and no canary (``not_run``)."""

    def after_load(self, host: RunnerHost, profile: EngineProfile, hello: HelloReply | None) -> CanaryStatus:
        return "not_run"


__all__ = ["EngineGuard", "NoGuard"]
