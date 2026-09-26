"""QA logic, plain Python (design sections 8, 11.1, 11.3 and 12; plan.md WP14).

Raw worker outputs (transcript and word times, embeddings, alignment, signal facts) plus the request's inputs
and the voice's measurement go in; verdicts, flags, suggestions, the consistency report, ``listen_first`` and
the report come out. No model, no GPU, no I/O beyond what the caller passes in.
"""

from __future__ import annotations

from .errors import QaUnavailable
from .fit import fit_report
from .normaliser import NumberReader
from .profile import DEFAULT_PROFILE, QaProfile
from .scorer import Scorer

__all__ = ["DEFAULT_PROFILE", "NumberReader", "QaProfile", "QaUnavailable", "Scorer", "fit_report"]
