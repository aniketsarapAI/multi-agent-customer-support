import json
import logging
import time
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)


_loggers: set[int] = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if id(logger) not in _loggers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _loggers.add(id(logger))
    return logger


class MetricsCollector:
    def __init__(self):
        self._requests_total = 0
        self._errors_total = 0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._tokens_input = 0
        self._tokens_output = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def record_request(
        self,
        latency_ms: float,
        error: bool = False,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cache_hit: bool = False,
    ):
        self._requests_total += 1
        if error:
            self._errors_total += 1
        self._latency_sum += latency_ms
        self._latency_count += 1
        self._tokens_input += tokens_input
        self._tokens_output += tokens_output
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    @property
    def summary(self) -> dict:
        avg_latency = self._latency_sum / self._latency_count if self._latency_count else 0
        error_rate = self._errors_total / self._requests_total if self._requests_total else 0
        cache_hit_rate = (self._cache_hits / (self._cache_hits + self._cache_misses)
                          if (self._cache_hits + self._cache_misses) else 0)
        return {
            "total_requests": self._requests_total,
            "total_errors": self._errors_total,
            "error_rate": round(error_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self._tokens_input,
            "total_output_tokens": self._tokens_output,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(cache_hit_rate, 4),
        }


class RequestTimer:
    def __init__(self):
        self.start = time.time()
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start) * 1000
