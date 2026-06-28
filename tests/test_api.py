import ast
import pytest


class TestApiStructure:
    """Verify the API module structure via AST (no imports, no downloads)."""

    def _parse_api(self):
        with open("app/api.py") as f:
            return ast.parse(f.read())

    def test_chat_request_class_exists(self):
        tree = self._parse_api()
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert "ChatRequest" in classes

    def test_chat_response_class_exists(self):
        tree = self._parse_api()
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert "ChatResponse" in classes

    def test_health_response_class_exists(self):
        tree = self._parse_api()
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert "HealthResponse" in classes

    def test_metrics_response_class_exists(self):
        tree = self._parse_api()
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert "MetricsResponse" in classes

    def _get_endpoints(self, tree, method: str) -> list[str]:
        endpoints = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in node.decorator_list:
                    if isinstance(d, ast.Call) and hasattr(d.func, "attr") and d.func.attr == method:
                        endpoints.append(node.name)
        return endpoints

    def test_health_endpoint_exists(self):
        tree = self._parse_api()
        endpoints = self._get_endpoints(tree, "get")
        assert "health" in endpoints

    def test_chat_post_endpoint_exists(self):
        tree = self._parse_api()
        endpoints = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for d in node.decorator_list:
                    if isinstance(d, ast.Call) and hasattr(d.func, "attr"):
                        endpoints.append((node.name, d.func.attr))
        post_endpoints = [n for n, v in endpoints if v == "post"]
        assert "chat" in post_endpoints

    def test_metrics_and_cache_endpoints_exist(self):
        tree = self._parse_api()
        get_endpoints = self._get_endpoints(tree, "get")
        assert "metrics" in get_endpoints
        assert "cache_stats" in get_endpoints

    def _parse_builder(self):
        with open("app/graph/builder.py") as f:
            return ast.parse(f.read())

    def test_security_pipeline_imported(self):
        tree = self._parse_builder()
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "security" in node.module:
                imports.append(node.module)
        assert any("security" in i for i in imports), "SecurityPipeline not imported by builder"

    def test_cache_module_imported(self):
        tree = self._parse_builder()
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "cache" in node.module:
                imports.append(node.module)
        assert any("cache" in i for i in imports), "ResponseCache not imported by builder"

    def test_monitoring_metrics_collector_imported(self):
        tree = self._parse_builder()
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "monitoring" in node.module:
                names = [a.name for a in node.names]
                imports.extend(names)
        assert "MetricsCollector" in imports, "MetricsCollector not imported by builder"

    def test_cors_middleware_registered(self):
        tree = self._parse_api()
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if hasattr(call.func, "attr") and call.func.attr == "add_middleware":
                    if hasattr(call.func, "value") and hasattr(call.func.value, "id"):
                        assert call.func.value.id == "app"
                        args = [ast.dump(a) for a in call.args]
                        assert any("CORSMiddleware" in a for a in args), "CORS not registered"
                        return
        pytest.fail("No add_middleware call found")

    def _has_decorator(self, tree, func_name: str, decorator_attr: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                for d in node.decorator_list:
                    if hasattr(d, "func") and hasattr(d.func, "attr") and d.func.attr == decorator_attr:
                        return True
        return False

    def test_rate_limiting_registered(self):
        tree = self._parse_api()
        assert self._has_decorator(tree, "chat", "limit"), "Rate limiter not registered on /chat"

    def test_response_model_on_chat(self):
        tree = self._parse_api()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "chat":
                for d in node.decorator_list:
                    if isinstance(d, ast.Call) and hasattr(d.func, "attr") and d.func.attr == "post":
                        kw_names = {kw.arg for kw in d.keywords if kw.arg is not None}
                        assert "response_model" in kw_names, "response_model should be on post decorator"
                        return
        pytest.fail("chat function with post decorator not found")


class TestApiResponseModels:
    """Validate Pydantic model schemas by constructing them directly."""

    def test_chat_response_has_all_fields(self):
        from pydantic import BaseModel
        import importlib.util
        spec = importlib.util.spec_from_file_location("api", "app/api.py")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pytest.skip("Cannot load api module without full dependency tree")

        resp = mod.ChatResponse(answer="test", query_type="document", logs=[], debug={})
        assert resp.security_notes == []
        assert resp.cached is False
        assert resp.processing_time_ms == 0
