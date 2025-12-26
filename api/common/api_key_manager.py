import datetime
import secrets
from typing import Dict, List, Optional, cast

from redis import Redis

from common.redis_client import redis_client


class APIKeyManager:
    def __init__(self):
        self.client: Redis = redis_client
        self.prefix = "DDIFFUSION_API_KEY"

    def _get_redis_key(self, token: str) -> str:
        return f"{self.prefix}:{token}"

    def is_active(self, token: str) -> bool:
        result = cast(Optional[str], self.client.hget(self._get_redis_key(token), "active"))
        return result == "1"

    def get_name(self, token: str) -> str:
        result = cast(Optional[str], self.client.hget(self._get_redis_key(token), "name"))
        if result is None:
            return "unknown"
        return result

    def create_key(self, name: str) -> str:
        token = secrets.token_urlsafe(32)
        key = self._get_redis_key(token)
        self.client.hset(
            key,
            mapping={
                "name": name,
                "active": "1",
                "created_at": datetime.datetime.utcnow().isoformat(),
            },
        )
        return token

    def list_keys(self) -> List[Dict[str, str]]:
        keys = []
        for key in self.client.scan_iter(f"{self.prefix}:*"):
            data = self.client.hgetall(key)
            token = key.split(f"{self.prefix}:")[1]
            keys.append({"api_key": token, **data})  # type: ignore
        return keys

    def revoke_key(self, token: str) -> bool:
        key = self._get_redis_key(token)
        if self.client.exists(key):
            self.client.hset(key, "active", "0")
            return True
        return False


key_manager = APIKeyManager()
