"""The models the service pins, by repo and 40-hex revision (design sections 4, 6, 10.2 and 18).

These are "the service's pins" of section 10.2: every key and every engine profile names its models by these
revisions, which are known before any work. A model loads only from a local snapshot folder named by its
revision, ``<models_root>/models--<org>--<name>/snapshots/<revision>/`` (section 4), and never from the network.
``narration-admin install`` fills those folders; this table is what it installs.

The licences are those section 18 records with every render and analysis. Where the design says "per model
card" (Whisper, WavLM), the service records exactly that, since the card is to be read at install.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from narration.contracts import names

REVISION_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
"""A pinned revision: a Hugging Face commit SHA, 40 lower-case hex characters."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PinnedModel:
    """One model as the service pins it: its repo, its revision and its licence (section 18)."""

    repo: str
    revision: str
    licence: str

    def __post_init__(self) -> None:
        if REVISION_PATTERN.fullmatch(self.revision) is None:
            raise ValueError(f"{self.repo}: a pinned revision is 40 lower-case hex characters, not {self.revision!r}")

    def snapshot_dir(self, models_root: Path) -> Path:
        """Where this model's snapshot lives under ``models_root`` (``snapshot_dir``)."""
        return snapshot_dir(models_root, self.repo, self.revision)


def snapshot_dir(models_root: Path, repo: str, revision: str) -> Path:
    """``<models_root>/models--<org>--<name>/snapshots/<revision>``: the Hugging Face cache layout, a folder
    named by the commit it holds (section 4)."""
    return models_root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


QWEN_BASE: Final = PinnedModel(
    repo=names.MODEL_QWEN_BASE, revision="fd4b254389122332181a7c3db7f27e918eec64e3", licence="Apache-2.0"
)
QWEN_DESIGN: Final = PinnedModel(
    repo=names.MODEL_QWEN_DESIGN, revision="5ecdb67327fd37bb2e042aab12ff7391903235d3", licence="Apache-2.0"
)
WHISPER: Final = PinnedModel(
    repo=names.MODEL_ASR, revision="06f233fe06e710322aca913c1bc4249a0d71fce1", licence="per model card"
)
WAVLM_SV: Final = PinnedModel(
    repo=names.MODEL_SV, revision="feb593a6c23c1cc3d9510425c29b0a14d2b07b1e", licence="per model card"
)
"""WavLM-base-plus-sv at ``main``, which holds only ``pytorch_model.bin``: the revision the evidence loaded
(WP22 measured the same tensors as ``refs/pr/8``'s safetensors)."""
CTC_ALIGNER: Final = PinnedModel(
    repo=names.MODEL_ALIGNER, revision="54074b1c16f4de6a5ad59affb4caa8f2ea03a119", licence="apache-2.0"
)

PINNED: Final[dict[str, PinnedModel]] = {m.repo: m for m in (QWEN_BASE, QWEN_DESIGN, WHISPER, WAVLM_SV, CTC_ALIGNER)}
"""Every model the service loads, by repo."""


def pinned(repo: str) -> PinnedModel:
    """The pin of ``repo``; ``KeyError`` names the repos the service pins."""
    try:
        return PINNED[repo]
    except KeyError:
        raise KeyError(f"{repo} is not a model this service pins ({', '.join(sorted(PINNED))})") from None


__all__ = [
    "CTC_ALIGNER",
    "PINNED",
    "QWEN_BASE",
    "QWEN_DESIGN",
    "REVISION_PATTERN",
    "WAVLM_SV",
    "WHISPER",
    "PinnedModel",
    "pinned",
    "snapshot_dir",
]
