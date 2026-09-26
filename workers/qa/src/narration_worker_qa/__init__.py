"""narration-mcp's QA worker (design section 4; plan.md WP15 and WP22).

A model runner only (plan.md P1): it loads models, runs them and returns raw outputs. Every verdict,
threshold and flag is computed in the server package. ``align`` holds the CTC aligner (WP15).
"""
