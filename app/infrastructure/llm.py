import os
import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings, LLM_MODEL, GRADER_MODEL, LLM_TEMPERATURE, EMBEDDING_MODEL, OPENROUTER_BASE_URL, OPENROUTER_API_KEY_ENV

logger = logging.getLogger(__name__)


def _make_llm(model: str, **overrides) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=LLM_TEMPERATURE,
        base_url=OPENROUTER_BASE_URL,
        api_key=settings.openrouter_api_key or os.getenv(OPENROUTER_API_KEY_ENV),
        streaming=True,
        timeout=30,
        max_retries=0,
        **overrides,
    )


class LLMWithFallback:
    def __init__(self):
        self._primary = _make_llm(LLM_MODEL)
        self._fallback = _make_llm(settings.fallback_model)

    def invoke(self, messages, **kwargs):
        try:
            return self._primary.invoke(messages, **kwargs)
        except Exception as e:
            logger.warning("Primary LLM failed, trying fallback", extra={"extra_data": {"error": str(e)[:200]}})
            try:
                return self._fallback.invoke(messages, **kwargs)
            except Exception:
                logger.error("Both primary and fallback LLM failed")
                return AIMessage(
                    content="I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."
                )

    def ainvoke(self, messages, **kwargs):
        try:
            return self._primary.ainvoke(messages, **kwargs)
        except Exception as e:
            logger.warning("Primary LLM failed (async), trying fallback", extra={"extra_data": {"error": str(e)[:200]}})
            try:
                return self._fallback.ainvoke(messages, **kwargs)
            except Exception:
                logger.error("Both primary and fallback LLM failed (async)")
                return AIMessage(
                    content="I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."
                )

    def with_structured_output(self, schema, default=None, **kwargs):
        return _StructuredWithFallback(self._primary, self._fallback, schema, kwargs, default)

    @property
    def ChatOpenAI(self):
        return self._primary


class _StructuredWithFallback:
    def __init__(self, primary, fallback, schema, kwargs, default=None):
        self._primary = primary.with_structured_output(schema, **kwargs)
        self._fallback = fallback.with_structured_output(schema, **kwargs)
        self._default = default

    def invoke(self, messages, **kwargs):
        try:
            return self._primary.invoke(messages, **kwargs)
        except Exception as e:
            logger.warning("Primary structured LLM failed, trying fallback", extra={"extra_data": {"error": str(e)[:200]}})
            try:
                return self._fallback.invoke(messages, **kwargs)
            except Exception:
                logger.error("Both primary and fallback structured LLM failed")
                if self._default is not None:
                    if callable(self._default):
                        return self._default()
                    return self._default
                raise


def get_llm() -> LLMWithFallback:
    return LLMWithFallback()


class GraderLLM:
    """Lightweight LLM for classification/grader tasks (decide_retrieval, etc.).

    Uses a separate model (configurable via `grader_model`) to reduce cost
    and latency for simple classification prompts. Falls back to the primary
    model on failure.
    """

    def __init__(self):
        self._primary = _make_llm(GRADER_MODEL)
        self._fallback = _make_llm(LLM_MODEL)

    def invoke(self, messages, **kwargs):
        try:
            return self._primary.invoke(messages, **kwargs)
        except Exception as e:
            logger.warning("Grader LLM failed, falling back to primary model", extra={"extra_data": {"error": str(e)[:200]}})
            return self._fallback.invoke(messages, **kwargs)

    def with_structured_output(self, schema, default=None, **kwargs):
        return _StructuredWithFallback(self._primary, self._fallback, schema, kwargs, default)

    @property
    def ChatOpenAI(self):
        return self._primary


@lru_cache(maxsize=1)
def get_grader_llm() -> GraderLLM:
    return GraderLLM()


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
