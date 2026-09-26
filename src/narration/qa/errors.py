"""What QA raises when it cannot score a take (design section 14, ``QA_UNAVAILABLE``).

``QaUnavailable`` is the contract's exception (``narration.contracts.errors``), re-exported here: ``score``
raises it when a check the inputs call for cannot run (for example, the voice has an anchor but the take has
no speaker embedding, or the two embeddings cannot be compared). The job engine catches it and records the
segment's ``QA_UNAVAILABLE``; a take is never passed without the check.
"""

from __future__ import annotations

from narration.contracts.errors import QaUnavailable

__all__ = ["QaUnavailable"]
