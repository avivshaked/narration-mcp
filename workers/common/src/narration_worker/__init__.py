"""narration_worker: code shared by narration-mcp's model workers (design Appendix A; plan.md WP16).

Workers are model runners only (plan.md P1): they load models, run them and return raw outputs.
Every verdict, threshold and flag lives in the server package.
"""

__version__ = "0.1.0"
