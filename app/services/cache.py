import hashlib
import json
import time
import threading


class ResponseCache:
    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[str, float]] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def _key(self, question: str) -> str:
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, question: str) -> str | None:
        key = self._key(question)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            answer, expiry = entry
            if time.time() > expiry:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return answer

    def set(self, question: str, answer: str) -> None:
        key = self._key(question)
        expiry = time.time() + self._ttl
        with self._lock:
            self._store[key] = (answer, expiry)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "cached_entries": len(self._store),
                "hit_rate": round(self._hits / (self._hits + self._misses), 3) if (self._hits + self._misses) > 0 else 0.0,
            }
