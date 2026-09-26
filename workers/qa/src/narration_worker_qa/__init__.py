"""narration-mcp's QA worker, the ``qa`` role (design section 4; plan.md WP15 and WP22).

A model runner only (plan.md P1): it loads models, runs them and returns raw outputs. Every verdict,
threshold and flag is computed in the server package.

- ``worker``: the protocol handler, ``QaHandler``, registered as the ``qa`` role's entry point
  (``python -m narration_worker --role qa``);
- ``asr``: Whisper-large-v3 transcription with word times;
- ``sv``: WavLM-base-plus-sv speaker embeddings;
- ``align``: the CTC aligner (WP15), and the errors every module here raises;
- ``voice``: the voice profile's measures and pictures, and the ``f0`` pitch track;
- ``snapshots``: the checks on a ``load``'s model snapshots.
"""
