"""The real ``Backend`` of the MCP front-end (design sections 4, 7, 12, 14, 15, 17; plan.md WP36).

``NarrationBackend`` (``service``) is what ``narration-mcp`` serves: it checks and plans each request, commits
jobs to the store, starts the daemon (``launch``), and reads jobs, results and status back. The modules:

- ``service``: ``NarrationBackend``, one method per tool;
- ``requests``: a request's controls, limits, stored form and identity;
- ``clips``: a caller's voice clip, checked (section 17.3, 17.4) and copied into the store;
- ``planning``: the plan over the three cache layers, and its estimates;
- ``assemble``: ``get_results`` for a generation job, restamped for the request (section 7.5);
- ``views``: ``get_job`` and ``get_server_status``, with DC-2's numbers;
- ``resources``: the ``narration://`` resources (section 7.7);
- ``launch``: the daemon's autostart.
"""

from __future__ import annotations

from .launch import DaemonLauncher, DetachedLauncher
from .planning import AnalysisPins
from .service import NarrationBackend

__all__ = ["AnalysisPins", "DaemonLauncher", "DetachedLauncher", "NarrationBackend"]
