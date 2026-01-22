import datetime
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, cast

import redis
from redis import Redis

from common.config import settings
from common.logger import logger
from common.schemas import APIKeyPublic, QueuePosition

_redis_client = redis.from_url(settings.celery_broker_url, decode_responses=True)


class RedisManager:
    def __init__(self):
        self.client: Redis = _redis_client
        self.prefix = "DDIFFUSION_API_KEY"

    def _get_redis_key(self, key_id: str) -> str:
        return f"{self.prefix}:{key_id}"

    def _name_exists(self, name: str) -> bool:
        for key in self.client.scan_iter(f"{self.prefix}:*"):
            if self.client.hget(key, "name") == name:
                return True
        return False

    def verify_token(self, token: str) -> Optional[APIKeyPublic]:
        """
        Verifies a token and returns its metadata if valid.
        Token format: dd_<key_id>_<secret>
        """
        if not token.startswith("dd_"):
            return None

        parts = token.split("_")
        if len(parts) != 3:
            return None

        key_id, secret = parts[1], parts[2]

        # Let RedisError bubble up to the global handler (500 error)
        data = self.client.hgetall(self._get_redis_key(key_id))
        if not data:
            return None

        key_data = cast(Dict, data)
        # Verify the secret against the stored hash and salt
        salt = key_data.get("salt")
        stored_hash = key_data.get("hash")
        name = key_data.get("name")

        if not salt or not stored_hash or not name:
            logger.error(f"Malformed API key data")
            return None

        # Use HMAC for constant-time comparison
        computed_hash = hashlib.sha256((secret + salt).encode()).hexdigest()

        if hmac.compare_digest(computed_hash, stored_hash):
            return APIKeyPublic(
                key_id=key_id,
                name=name,
                created_at=key_data.get("created_at", ""),
            )

        return None

    def create_key(self, name: str) -> str:
        if self._name_exists(name):
            raise ValueError("Key name already exists")

        # Replace underscores with hyphens so the dd_ID_SECRET format
        # can be parsed unambiguously using split("_")
        key_id = secrets.token_urlsafe(12).replace("_", "-")
        secret = secrets.token_urlsafe(32).replace("_", "-")
        salt = secrets.token_hex(16)

        hashed = hashlib.sha256((secret + salt).encode()).hexdigest()

        self.client.hset(
            self._get_redis_key(key_id),
            mapping={
                "name": name,
                "hash": hashed,
                "salt": salt,
                "created_at": datetime.datetime.utcnow().isoformat(),
            },
        )
        return f"dd_{key_id}_{secret}"

    def list_keys(self) -> List[APIKeyPublic]:
        keys = []
        for key in self.client.scan_iter(f"{self.prefix}:*"):
            data = self.client.hgetall(key)
            key_data = cast(Dict, data)

            key_id = key.split(f"{self.prefix}:")[1]

            keys.append(
                APIKeyPublic(
                    key_id=key_id,
                    name=key_data.get("name", "unknown"),
                    created_at=key_data.get("created_at", ""),
                )
            )
        return keys

    def delete_key(self, key_id: str) -> bool:
        """Permanently delete the key from Redis."""
        key = self._get_redis_key(key_id)
        return bool(self.client.delete(key))

    def waiting_tasks(self, queues=["gpu", "cpu", "comfy"]) -> int:
        """
        Returns the number of waiting tasks
        """

        waiting = sum(cast(int, self.client.llen(q)) for q in queues)
        return waiting

    def get_queue_position(self, task_id: str, queues=["gpu", "cpu", "comfy"]) -> Optional[QueuePosition]:
        """
        Finds the 1-based position of a task in the Redis queues.
        """
        for q in queues:
            # lrange is O(N), but we keep our task_backlog_limit small
            # so this is fast at our scale.
            tasks = cast(List[bytes], self.client.lrange(q, 0, -1))
            for index, task_payload in enumerate(tasks):
                # We check for the task_id string inside the byte payload
                if task_id in str(task_payload):
                    return QueuePosition(position=index + 1, queue=q, total=len(tasks))
        return None


redis_manager = RedisManager()
