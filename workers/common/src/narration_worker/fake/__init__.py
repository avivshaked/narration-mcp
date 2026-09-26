"""The ``fake`` worker role (plan.md WP16): deterministic synthetic outputs for every op, and planted faults.

See ``handler.py`` for what each op returns, ``audio.py`` for the synthetic speech, and ``faults.py`` for the
fault spec (``NARRATION_FAKE_SPEC``).
"""

from __future__ import annotations

from .handler import FakeHandler

__all__ = ["FakeHandler"]
