import datetime
import hashlib
import os
import secrets
from typing import Dict, List, cast

import redis
from redis import Redis

# Connect to Redis
redis_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
redis_client = redis.from_url(redis_url, decode_responses=True)


class APIKeyManager:
    def __init__(self):
        self.client: Redis = redis_client
        self.prefix = "DDIFFUSION_API_KEY"

    def _get_redis_key(self, hashed_token: str) -> str:
        return f"{self.prefix}:{hashed_token}"

    def hash_token(self, token: str) -> str:
        """Hashes the token so the secret is never stored in plaintext."""
        return hashlib.sha256(token.encode()).hexdigest()

    def _name_exists(self, name: str) -> bool:
        for key in self.client.scan_iter(f"{self.prefix}:*"):
            if self.client.hget(key, "name") == name:
                return True
        return False

    def is_active(self, token: str) -> bool:
        hashed = self.hash_token(token)
        return bool(self.client.exists(self._get_redis_key(hashed)))

    def create_key(self, name: str) -> str:
        if self._name_exists(name):
            raise ValueError("Key name already exists")

        token = secrets.token_urlsafe(32)
        hashed = self.hash_token(token)
        key = self._get_redis_key(hashed)

        self.client.hset(
            key,
            mapping={
                "name": name,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "prefix": f"{token[:8]}...",
            },
        )
        return token

    def list_keys(self) -> List[Dict[str, str]]:
        keys = []
        for key in self.client.scan_iter(f"{self.prefix}:*"):
            data = self.client.hgetall(key)
            # The 'key_id' for management is the hash (the part after the prefix)
            key_hash = key.split(f"{self.prefix}:")[1]

            keys.append({"key_id": key_hash, **data})  # type: ignore
        return keys

    def delete_key(self, key_id: str) -> bool:
        """Permanently delete the key from Redis."""
        key = self._get_redis_key(key_id)
        return bool(self.client.delete(key))

    def get_name(self, key_hash: str) -> str:
        result = self.client.hget(self._get_redis_key(key_hash), "name")
        if result is None:
            return "unknown"

        return cast(str, result)

    def check_rate_limit(self, key_hash: str, limit: int = 60, window: int = 60) -> bool:
        """
        Checks if the token has exceeded the rate limit.
        :param limit: Max requests allowed.
        :param window: Time window in seconds.
        :return: True if request is allowed, False if limit exceeded.
        """
        # Create a specific key for rate limiting
        key = f"{self.prefix}_RATE_LIMIT:{key_hash}"

        # Atomic increment
        current_count = cast(int, self.client.incr(key))

        # If this is the first request, set the expiry
        if current_count == 1:
            self.client.expire(key, window)

        return current_count <= limit


key_manager = APIKeyManager()
