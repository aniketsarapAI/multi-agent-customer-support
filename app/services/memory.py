import json
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class MemoryService(Protocol):
    def load(self, conversation_id: str) -> dict:
        ...

    def save(self, conversation_id: str, state: dict) -> None:
        ...


class InMemoryMemoryService:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def load(self, conversation_id: str) -> dict:
        return self._store.get(conversation_id, {})

    def save(self, conversation_id: str, state: dict) -> None:
        self._store[conversation_id] = state


_KEY_PREFIX = "mem:conv:"


class RedisMemoryService:
    def __init__(self, redis_url: str, ttl_seconds: int = 86400):
        self._ttl = ttl_seconds
        self._redis = self._connect(redis_url)

    def _connect(self, redis_url: str):
        try:
            import redis as r
            client = r.from_url(redis_url, decode_responses=True, socket_connect_timeout=3)
            client.ping()
            logger.info("Connected to Redis for memory at %s", redis_url)
            return client
        except Exception as e:
            logger.warning("Redis memory connection failed (%s), falling back to in-memory", e)
            return None

    @property
    def available(self) -> bool:
        return self._redis is not None

    def _key(self, conversation_id: str) -> str:
        return f"{_KEY_PREFIX}{conversation_id}"

    def load(self, conversation_id: str) -> dict:
        if not self.available:
            return {}
        try:
            data = self._redis.get(self._key(conversation_id))
            if data is None:
                return {}
            return json.loads(data)
        except Exception as e:
            logger.warning("Redis memory load failed: %s", e)
            return {}

    def save(self, conversation_id: str, state: dict) -> None:
        if not self.available:
            return
        try:
            self._redis.setex(self._key(conversation_id), self._ttl, json.dumps(state))
        except Exception as e:
            logger.warning("Redis memory save failed: %s", e)
