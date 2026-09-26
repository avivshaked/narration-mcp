"""narration-mcp's Qwen3-TTS worker: Base (ICL clone) and VoiceDesign (design section 4, Appendix A).

- ``settings``: the audio-changing settings, checked (standard library only);
- ``engine``: loads a pinned snapshot and renders with those settings, recording how generation stopped;
- ``worker``: the ``qwen3`` role's protocol handler, registered as the entry point
  ``narration_worker.roles:qwen3``.

A model runner only (plan.md P1): it returns raw audio and generation facts; every verdict is the server's.
"""

__version__ = "0.1.0"
