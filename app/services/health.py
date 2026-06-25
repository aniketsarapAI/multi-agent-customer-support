class HealthService:
    def __init__(self, registry=None, memory=None, cache=None, db=None, vector_store=None, llm=None):
        self._registry = registry
        self._memory = memory
        self._cache = cache
        self._db = db
        self._vector_store = vector_store
        self._llm = llm

    def check(self) -> dict[str, str]:
        results: dict[str, str] = {}

        if self._registry:
            results["agents"] = "healthy"
        else:
            results["agents"] = "unknown"

        if self._cache:
            results["cache"] = "healthy"
        else:
            results["cache"] = "unknown"

        if self._memory:
            results["memory"] = "healthy"
        else:
            results["memory"] = "unknown"

        results["vector_store"] = "healthy"
        results["database"] = "configured"

        # Determine overall status
        statuses = list(results.values())
        if "critical" in statuses:
            results["status"] = "critical"
        elif "degraded" in statuses:
            results["status"] = "degraded"
        else:
            results["status"] = "healthy"

        return results
