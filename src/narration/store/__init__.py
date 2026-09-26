"""The store: SQLite (WAL) rows plus content-addressed files under the store root (design sections 4, 6,
15 and 17.2; plan.md WP12).

``NarrationStore`` implements ``narration.contracts.interfaces.Store``. It is stateless in the design's
sense (sections 0.2 and 2): it keeps caches of work done, the job queue and the service's own records,
never a caller's script, choice, approval, pronunciation list or voice.
"""

from .db import SCHEMA_VERSION, StoreSchemaError
from .layout import InvalidIdError, StoreLayout, StorePathError
from .store import (
    DEFAULT_GRACE_S,
    LeaseLostError,
    NarrationStore,
    NotFoundError,
    RetentionKind,
    SqliteLease,
    StoreError,
    StoreIntegrityError,
)

__all__ = [
    "DEFAULT_GRACE_S",
    "SCHEMA_VERSION",
    "InvalidIdError",
    "LeaseLostError",
    "NarrationStore",
    "NotFoundError",
    "RetentionKind",
    "SqliteLease",
    "StoreError",
    "StoreIntegrityError",
    "StoreLayout",
    "StorePathError",
    "StoreSchemaError",
]
