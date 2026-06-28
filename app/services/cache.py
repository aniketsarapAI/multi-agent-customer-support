import hashlib
import json
import logging
import math
import time
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Legacy exact-match caches (kept for tests)
# ──────────────────────────────────────────────


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


class RedisCache:
    def __init__(self, redis_url: str, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
        self._redis = self._connect(redis_url)

    def _connect(self, redis_url: str):
        try:
            import redis as r
            client = r.from_url(redis_url, decode_responses=True, socket_connect_timeout=3)
            client.ping()
            logger.info("Connected to Redis at %s", redis_url)
            return client
        except Exception as e:
            logger.warning("Redis connection failed (%s), falling back to in-memory cache", e)
            return None

    @property
    def available(self) -> bool:
        return self._redis is not None

    def _key(self, question: str) -> str:
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, question: str) -> str | None:
        if not self.available:
            return None
        key = self._key(question)
        try:
            answer = self._redis.get(key)
            if answer is None:
                with self._lock:
                    self._misses += 1
                return None
            with self._lock:
                self._hits += 1
            return answer
        except Exception:
            with self._lock:
                self._misses += 1
            return None

    def set(self, question: str, answer: str) -> None:
        if not self.available:
            return
        key = self._key(question)
        try:
            self._redis.setex(key, self._ttl, answer)
        except Exception as e:
            logger.warning("Redis set failed: %s", e)

    @property
    def stats(self) -> dict:
        with self._lock:
            info = {}
            if self.available:
                try:
                    info = self._redis.info("keyspace")
                except Exception:
                    pass
            return {
                "hits": self._hits,
                "misses": self._misses,
                "cached_entries": self._redis.dbsize() if self.available else 0,
                "hit_rate": round(self._hits / (self._hits + self._misses), 3) if (self._hits + self._misses) > 0 else 0.0,
                "backend": "redis" if self.available else "none",
                "redis_info": info,
            }


# ──────────────────────────────────────────────
# Semantic cache (FAISS + embeddings + Redis)
# ──────────────────────────────────────────────

_ANS_PREFIX = "cache:ans:"
_EMB_PREFIX = "cache:emb:"


class SemanticCache:
    def __init__(
        self,
        redis_url: str = "",
        ttl_seconds: int = 600,
        similarity_threshold: float = 0.92,
        freq_questions_path: str = "",
    ):
        self._ttl = ttl_seconds
        self._threshold = similarity_threshold
        self._lock = threading.Lock()

        # hit/miss counters
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0

        # Embedding model (singleton via lru_cache)
        from app.infrastructure.llm import get_embeddings
        self._embedder = get_embeddings()

        # FAISS index
        import faiss
        self._dimension = 768
        self._index = faiss.IndexFlatIP(self._dimension)
        self._id_to_key: dict[int, str] = {}

        # Answer storage
        self._redis = self._connect_redis(redis_url)
        self._store: dict[str, tuple[str, float]] = {}

        # Pre-seed
        if freq_questions_path:
            self._load_freq_questions(freq_questions_path)

    # ── Redis connection ──

    def _connect_redis(self, redis_url: str):
        if not redis_url:
            return None
        try:
            import redis as r
            client = r.from_url(redis_url, decode_responses=True, socket_connect_timeout=3)
            client.ping()
            logger.info("SemanticCache: connected to Redis at %s", redis_url)
            return client
        except Exception as e:
            logger.warning("SemanticCache: Redis unavailable (%s), using in-memory store", e)
            return None

    @property
    def _redis_ok(self) -> bool:
        return self._redis is not None

    # ── Key helpers ──

    @staticmethod
    def _question_key(question: str) -> str:
        return hashlib.sha256(question.strip().lower().encode()).hexdigest()

    # ── Embedding ──

    def _embed(self, text: str):
        vec = self._embedder.embed_query(text)
        return [float(v) for v in vec]

    def _normalize(self, vec: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(v * v for v in vec))
        if magnitude == 0:
            return vec
        return [v / magnitude for v in vec]

    # ── Seed from freq_questions.json ──

    def _load_freq_questions(self, path: str):
        resolved = Path(path)
        if not resolved.is_absolute():
            from app.config import BASE_DIR
            resolved = BASE_DIR / path
        if not resolved.exists():
            logger.info("SemanticCache: no freq_questions file at %s", resolved)
            return
        try:
            with open(resolved) as f:
                entries = json.load(f)
        except Exception as e:
            logger.warning("SemanticCache: failed to load freq_questions: %s", e)
            return

        count = 0
        for entry in entries:
            q = entry.get("question", "").strip()
            a = entry.get("answer", "").strip()
            if not q or not a:
                continue
            self.set(q, a, pinned=True)
            count += 1
        logger.info("SemanticCache: pre-seeded %d entries from %s", count, resolved)

    # ── Public API ──

    def get(self, question: str) -> str | None:
        qkey = self._question_key(question)

        # 1. Try exact match (fast path)
        answer = self._get_by_key(qkey)
        if answer is not None:
            with self._lock:
                self._exact_hits += 1
            logger.debug("SemanticCache: exact hit for qkey=%s", qkey[:12])
            return answer

        # 2. Semantic match
        try:
            vec = self._embed(question)
            vec_norm = self._normalize(vec)
            import numpy as np
            query = np.array([vec_norm], dtype=np.float32)

            with self._lock:
                if self._index.ntotal == 0:
                    self._misses += 1
                    return None
                distances, indices = self._index.search(query, 1)
                distance = float(distances[0][0])
                idx = int(indices[0][0])

            if distance >= self._threshold and idx in self._id_to_key:
                matched_key = self._id_to_key[idx]
                answer = self._get_by_key(matched_key)
                if answer is not None:
                    with self._lock:
                        self._semantic_hits += 1
                    logger.debug(
                        "SemanticCache: semantic hit (dist=%.4f) for %s",
                        distance, qkey[:12],
                    )
                    return answer
        except Exception as e:
            logger.warning("SemanticCache: semantic lookup failed: %s", e)

        with self._lock:
            self._misses += 1
        return None

    def set(self, question: str, answer: str, pinned: bool = False) -> None:
        qkey = self._question_key(question)

        # Embed and add to FAISS
        try:
            vec = self._embed(question)
            vec_norm = self._normalize(vec)
            import numpy as np
            with self._lock:
                idx = self._index.ntotal
                self._index.add(np.array([vec_norm], dtype=np.float32))
                self._id_to_key[idx] = qkey
        except Exception as e:
            logger.warning("SemanticCache: failed to add to FAISS index: %s", e)

        # Store answer
        if pinned:
            self._set_by_key_pinned(qkey, answer)
        else:
            self._set_by_key(qkey, answer)

    @property
    def stats(self) -> dict:
        with self._lock:
            total_hits = self._exact_hits + self._semantic_hits
            total = total_hits + self._misses
            return {
                "exact_hits": self._exact_hits,
                "semantic_hits": self._semantic_hits,
                "misses": self._misses,
                "cached_entries": self._index.ntotal,
                "hit_rate": round(total_hits / total, 3) if total > 0 else 0.0,
                "backend": "redis" if self._redis_ok else "memory",
            }

    # ── Internal storage helpers ──

    def _get_by_key(self, qkey: str) -> str | None:
        if self._redis_ok:
            try:
                return self._redis.get(f"{_ANS_PREFIX}{qkey}")
            except Exception:
                return None
        entry = self._store.get(qkey)
        if entry is None:
            return None
        answer, expiry = entry
        if time.time() > expiry:
            del self._store[qkey]
            return None
        return answer

    def _set_by_key(self, qkey: str, answer: str) -> None:
        if self._redis_ok:
            try:
                self._redis.setex(f"{_ANS_PREFIX}{qkey}", self._ttl, answer)
                return
            except Exception as e:
                logger.warning("SemanticCache: Redis set failed: %s", e)
        expiry = time.time() + self._ttl
        self._store[qkey] = (answer, expiry)

    def _set_by_key_pinned(self, qkey: str, answer: str) -> None:
        if self._redis_ok:
            try:
                self._redis.set(f"{_ANS_PREFIX}{qkey}", answer)
                return
            except Exception as e:
                logger.warning("SemanticCache: Redis pinned set failed: %s", e)
        self._store[qkey] = (answer, float("inf"))
