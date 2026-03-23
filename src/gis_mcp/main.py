""" GIS MCP Server - Main entry point

This module implements an MCP server that connects LLMs to GIS operations using
Shapely and PyProj libraries, enabling AI assistants to perform geospatial operations
and transformations.
"""

import logging
import argparse
import sys
import os
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .mcp import gis_mcp
from .storage_config import initialize_storage
try:
    from .data import administrative_boundaries
except ImportError as e:
    administrative_boundaries = None
    import logging
    logging.warning(f"administrative_boundaries module could not be imported: {e}. Install with 'pip install gis-mcp[administrative-boundaries]' if you need this feature.")
try:
    from .data import climate
except ImportError as e:
    climate = None
    import logging
    logging.warning(f"climate module could not be imported: {e}. Install with 'pip install gis-mcp[climate]' if you need this feature.")
try:
    from .data import ecology
except ImportError as e:
    ecology = None
    import logging
    logging.warning(f"ecology module could not be imported: {e}. Install with 'pip install gis-mcp[ecology]' if you need this feature.")
try:
    from .data import movement
except ImportError as e:
    movement = None
    import logging
    logging.warning(f"movement module could not be imported: {e}. Install with 'pip install gis-mcp[movement]' if you need this feature.")

try:
    from .data import satellite_imagery
except ImportError as e:
    satellite_imagery = None
    import logging
    logging.warning(f"satellite_imagery module could not be imported: {e}. Install with 'pip install gis-mcp[satellite_imagery]' if you need this feature.")

try:
    from .data import land_cover
except ImportError as e:
    land_cover = None
    import logging
    logging.warning(f"land_cover module could not be imported: {e}. Install with 'pip install gis-mcp[land_cover]' if you need this feature.")

try:
    from .visualize import map_tool, web_map_tool
except ImportError as e:
    map_tool = None
    web_map_tool = None
    import logging
    logging.warning(f"Visualization modules could not be imported: {e}. Install with 'pip install gis-mcp[visualize]' if you need this feature.")


import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Import tool modules to register MCP tools via decorators
from . import (
    geopandas_functions,
    shapely_functions,
    rasterio_functions,
    pyproj_functions,
    pysal_functions,
)

# Import storage endpoints to register HTTP routes (for HTTP/SSE transport)
try:
    from . import storage_endpoints
except ImportError as e:
    import logging
    logging.warning(f"Storage endpoints could not be imported: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gis-mcp")

# Create FastMCP instance

def _load_env_file(path: str = ".env") -> None:
    """Load key/value pairs from .env into os.environ when missing."""
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        logger.warning(f"Failed to load .env file from {path}: {exc}")


def _is_enabled(value: str | None, default: bool = True) -> bool:
    """Parse a bool-like environment value."""
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def _register_with_orchestrator(
    transport: str,
    host: str,
    port: int,
    max_attempts: int = 5,
    timeout: float = 10.0,
) -> None:
    """Register this MCP server with Clairvoyint orchestrator when enabled."""
    if transport == "stdio":
        return

    if not _is_enabled(os.getenv("MCP_REGISTRATION_ENABLED"), default=True):
        logger.info("MCP registration disabled by MCP_REGISTRATION_ENABLED")
        return

    orchestrator_base_url = os.getenv("ORCHESTRATOR_BASE_URL", "").strip()
    if not orchestrator_base_url:
        orchestrator_ip = os.getenv("ORCHESTRATOR_IP", "").strip()
        orchestrator_port = os.getenv("ORCHESTRATOR_PORT", "").strip()
        if orchestrator_ip and orchestrator_port:
            orchestrator_base_url = f"http://{orchestrator_ip}:{orchestrator_port}"

    if not orchestrator_base_url:
        logger.info("No orchestrator address configured; skipping registration")
        return

    shared_secret = os.getenv("MCP_REGISTRATION_SECRET", "").strip()
    if not shared_secret:
        logger.warning("MCP_REGISTRATION_SECRET missing; skipping registration")
        return

    server_id = os.getenv("MCP_SERVER_ID", "gis_mcp").strip()
    description = os.getenv("MCP_DESCRIPTION", "GIS MCP server")
    mcp_base_url = os.getenv("MCP_BASE_URL", f"http://{host}:{port}").rstrip("/")
    capabilities = {"transport": transport}

    payload = {
        "server_id": server_id,
        "base_url": mcp_base_url,
        "shared_secret": shared_secret,
        "description": description,
        "capabilities": capabilities,
    }
    request = Request(
        f"{orchestrator_base_url.rstrip('/')}/mcp/register",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    logger.info(
                        "Registered with orchestrator at %s (attempt %d)",
                        orchestrator_base_url,
                        attempt,
                    )
                    return
        except HTTPError as exc:
            logger.warning(
                "Orchestrator registration HTTP error (attempt %d/%d): %s %s",
                attempt,
                max_attempts,
                exc.code,
                exc.reason,
            )
        except (URLError, OSError, TimeoutError) as exc:
            logger.warning(
                "Orchestrator registration connection error (attempt %d/%d): %s",
                attempt,
                max_attempts,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Orchestrator registration failed (attempt %d/%d): %s",
                attempt,
                max_attempts,
                exc,
            )

        if attempt < max_attempts:
            time.sleep(2.0)

    logger.error("Unable to register with orchestrator after %d attempts", max_attempts)


def main():
    """Main entry point for the GIS MCP server."""
    _load_env_file()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GIS MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--storage-path", type=str, default=None, 
                       help="Path to storage folder for file operations (default: ~/.gis_mcp/data/)")
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize storage configuration
    # Check environment variable first, then command-line argument
    storage_config = os.getenv('GIS_MCP_STORAGE_PATH', args.storage_path)
    storage_path = initialize_storage(storage_config)
    logger.info(f"Storage path initialized: {storage_path}")
    
    # Get transport configuration from environment variables
    transport = os.getenv('GIS_MCP_TRANSPORT', 'stdio').lower()
    
    try:
        if transport == 'stdio':
            # Default stdio transport
            print("Starting GIS MCP server with STDIO transport...")
            logger.info("STDIO transport enabled (default)")
            gis_mcp.run()
        else:
            # HTTP transport configuration
            host = os.getenv('GIS_MCP_HOST', '0.0.0.0')
            port = int(os.getenv('GIS_MCP_PORT', '8080'))
            _register_with_orchestrator(transport=transport, host=host, port=port)
            
            print(f"Starting GIS MCP server with {transport} transport on {host}:{port}")
            print(f"\nAvailable endpoints:")
            if transport == 'sse':
                print(f"  - MCP (SSE): http://{host}:{port}/sse")
            else:
                print(f"  - MCP (HTTP): http://{host}:{port}/mcp")
            print(f"  - Storage upload: http://{host}:{port}/storage/upload")
            print(f"  - Storage download: http://{host}:{port}/storage/download")
            print(f"  - Storage list: http://{host}:{port}/storage/list")
            print(f"\nFor detailed endpoint documentation, visit: https://gis-mcp.com/endpoints/")
            logger.info(f"{transport} transport enabled - {host}:{port}")
            
            gis_mcp.run(transport=transport, host=host, port=port)
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
