import json
import logging
import pprint

from fastapi import Request

from common.schemas import Identity
from utils.utils import truncate_strings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create a logger object
logger = logging.getLogger(__name__)


def log_pretty(message, obj):
    """Utility function to pretty print objects in logs."""
    logger.info(message + "\n%s", pprint.pformat(obj, indent=1, width=120, sort_dicts=False))


async def log_request(request: Request, identity: Identity):
    """Logs the request with identity and body if it's a POST request."""

    user_id = identity.user_id
    key_name = identity.key_name
    identity_str = f"{key_name} ({user_id})"

    if request.method == "POST":
        body_str = ""
        body_bytes = await request.body()
        request._body = body_bytes  # Preserve for route handler
        try:
            body_json = json.loads(body_bytes.decode("utf-8"))
            body_str = json.dumps(truncate_strings(body_json))
        except Exception:
            # If not valid JSON, log first 100 chars
            raw_body = body_bytes.decode("utf-8", errors="replace")
            body_str = truncate_strings(raw_body)

        logger.info(f"[Auth] {identity_str} -> {request.method} {request.url.path} body:{body_str}")
    else:
        logger.info(f"[Auth] {identity_str} -> {request.method} {request.url.path}")
