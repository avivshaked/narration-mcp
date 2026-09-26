"""The positive-only lint of voice descriptions (design section 3.5; WP10).

``NegationLinter`` implements ``narration.contracts.interfaces.DescriptionLinter``.
"""

from __future__ import annotations

from .negation import NegationLinter, lint

__all__ = ["NegationLinter", "lint"]
