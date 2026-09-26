# ruff: noqa
# type: ignore
# fmt: off
#
# Vendored from openai/whisper, MIT licence (LICENSE beside this file; THIRD_PARTY_NOTICES):
#   https://github.com/openai/whisper/blob/31243bad24cc746f07d4c8bfdd2d974872cb1803/whisper/normalizers/__init__.py
#   release tag v20250625, commit 31243bad24cc746f07d4c8bfdd2d974872cb1803
#   upstream file sha256 f18fcdce4caee7f2e80e4c01ab0457d3a080a872a29489542c6d23c4b0bba572
# Kept as upstream wrote it, so it can be diffed against the source; the project's lint, format and
# type rules are switched off for this file only. Changes for narration-mcp (plan.md P2):
#   - none
#
from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizer
