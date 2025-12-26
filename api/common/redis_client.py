import os

import redis

# Connect to Redis
redis_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
redis_client = redis.from_url(redis_url, decode_responses=True)
