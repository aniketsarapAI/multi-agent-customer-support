import bisect
import json
import logging
import time


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


_loggers_created: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    if name in _loggers_created:
        return logging.getLogger(name)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    _loggers_created.add(name)
    return logger


class MetricsCollector:
    def __init__(self):
        self._latencies: list[float] = []
        self._sorted_latencies: list[float] = []
        self._total_requests = 0
        self._total_errors = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._tokens_input = 0
        self._tokens_output = 0

    def record_request(self, latency_ms: float = 0, error: bool = False, tokens_input: int = 0, tokens_output: int = 0, cache_hit: bool = False):
        self._total_requests += 1
        if error:
            self._total_errors += 1
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        if latency_ms > 0:
            self._latencies.append(latency_ms)
            bisect.insort(self._sorted_latencies, latency_ms)
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]
                self._sorted_latencies = sorted(self._latencies)
        self._tokens_input += tokens_input
        self._tokens_output += tokens_output

    @property
    def summary(self) -> dict:
        n = len(self._latencies)
        avg_latency = sum(self._latencies) / n if n > 0 else 0.0
        total = self._total_requests
        errors = self._total_errors
        cache_total = self._cache_hits + self._cache_misses
        return {
            "total_requests": total,
            "total_errors": errors,
            "error_rate": round(errors / total, 3) if total > 0 else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
            "p50_latency_ms": round(self._sorted_latencies[n // 2], 2) if n > 0 else 0.0,
            "p95_latency_ms": round(self._sorted_latencies[int(n * 0.95)], 2) if n > 0 else 0.0,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(self._cache_hits / cache_total, 3) if cache_total > 0 else 0.0,
            "total_tokens_input": self._tokens_input,
            "total_tokens_output": self._tokens_output,
        }


class RequestTimer:
    def __init__(self):
        self.elapsed_ms = 0.0
        self._start: float | None = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._start is not None:
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000
