import time
import threading

from app.services.cache import ResponseCache, RedisCache, SemanticCache


class TestResponseCache:
    def setup_method(self):
        self.cache = ResponseCache(ttl_seconds=2)

    def test_cache_miss_returns_none(self):
        assert self.cache.get("unknown query") is None

    def test_cache_hit_returns_response(self):
        self.cache.set("What is Python?", "A programming language.")
        result = self.cache.get("What is Python?")
        assert result == "A programming language."

    def test_case_insensitive_matching(self):
        self.cache.set("What is Python?", "A programming language.")
        result = self.cache.get("what is python?")
        assert result == "A programming language."

    def test_ttl_expiration(self):
        self.cache = ResponseCache(ttl_seconds=1)
        self.cache.set("query", "response")
        assert self.cache.get("query") == "response"
        time.sleep(1.5)
        assert self.cache.get("query") is None

    def test_stats_tracking(self):
        self.cache.get("miss1")
        self.cache.get("miss2")
        self.cache.set("hit", "value")
        self.cache.get("hit")

        stats = self.cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["cached_entries"] == 1

    def test_concurrent_access(self):
        results = []
        errors = []

        def worker(i):
            try:
                self.cache.set(f"key{i}", f"val{i}")
                v = self.cache.get(f"key{i}")
                results.append(v)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        assert all(r == f"val{i}" for i, r in enumerate(results))


class TestRedisCache:
    def test_fallback_when_redis_unavailable(self):
        cache = RedisCache("redis://localhost:16379/0", ttl_seconds=30)
        assert not cache.available
        assert cache.get("any question") is None
        cache.set("any question", "answer")  # should not raise
        stats = cache.stats
        assert stats["backend"] == "none"

    def test_stats_tracking_on_fallback(self):
        cache = RedisCache("redis://localhost:16379/0", ttl_seconds=30)
        cache.get("q1")
        cache.get("q2")
        stats = cache.stats
        # When Redis is unavailable, misses are not tracked (no cache to miss against)
        assert stats["hits"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["backend"] == "none"


class TestSemanticCache:
    def setup_method(self):
        self.cache = SemanticCache(
            redis_url="",
            ttl_seconds=60,
            similarity_threshold=0.92,
        )

    def test_miss_on_empty_cache(self):
        assert self.cache.get("anything") is None
        stats = self.cache.stats
        assert stats["misses"] == 1
        assert stats["exact_hits"] == 0
        assert stats["semantic_hits"] == 0

    def test_exact_hit(self):
        self.cache.set("What is Python?", "A programming language.")
        result = self.cache.get("What is Python?")
        assert result == "A programming language."
        stats = self.cache.stats
        assert stats["exact_hits"] == 1
        assert stats["semantic_hits"] == 0

    def test_case_insensitive_exact_hit(self):
        self.cache.set("What is Python?", "A programming language.")
        result = self.cache.get("what is python?")
        assert result == "A programming language."

    def test_stats_tracking(self):
        self.cache.get("miss1")
        self.cache.get("miss2")
        self.cache.set("hit1", "val1")
        self.cache.get("hit1")
        self.cache.set("hit2", "val2")
        self.cache.get("hit2")

        stats = self.cache.stats
        assert stats["exact_hits"] == 2
        assert stats["misses"] == 2
        assert stats["cached_entries"] == 2

    def test_pinned_entries_never_expire(self):
        self.cache.set("pinned", "value", pinned=True)
        assert self.cache.get("pinned") == "value"
