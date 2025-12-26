import hashlib
import os

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def get_rate_limit_key(request: Request) -> str:
    """
    Returns a hashed version of the API key for rate limiting.
    Falls back to IP address if no key is present.
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
        # Hash the token so we don't store raw keys in Redis
        return hashlib.sha256(token.encode()).hexdigest()

    return get_remote_address(request)


# Use Redis if available, otherwise memory (fallback)
redis_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
limiter = Limiter(key_func=get_rate_limit_key, storage_uri=redis_url)

CREATE_LIMIT = os.getenv("CREATE_LIMIT", "60/minute")
