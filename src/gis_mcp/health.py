"""Health check tool for GIS MCP."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .mcp import gis_mcp


@gis_mcp.tool
def health_check() -> dict[str, Any]:
    """Health check for the GIS MCP server.

    Returns:
        dict[str, Any]: Health payload with a success flag and timestamp.
    """
    return {
        "success": True,
        "server": "gis_mcp",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

