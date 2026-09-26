"""The MCP front-end, ``narration-mcp`` (design sections 5 and 7; plan.md WP17 and WP36).

``build_front_end(backend)`` gives a ``FrontEnd`` whose ``server`` publishes the v1 tools, resources and
prompts over a ``narration.contracts.interfaces.Backend``; ``run_stdio()`` serves one client on stdio.
"""

from .server import FrontEnd, build_front_end

__all__ = ["FrontEnd", "build_front_end"]
