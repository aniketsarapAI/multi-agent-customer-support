import os
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings, LLM_MODEL, LLM_TEMPERATURE, EMBEDDING_MODEL, OPENROUTER_BASE_URL, OPENROUTER_API_KEY_ENV

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

    def with_structured_output(self, schema, **kwargs):
        return _StructuredWithFallback(self._primary, self._fallback, schema, kwargs)

    @property
    def ChatOpenAI(self):
        return self._primary


class _StructuredWithFallback:
    def __init__(self, primary, fallback, schema, kwargs):
        self._primary = primary.with_structured_output(schema, **kwargs)
        self._fallback = fallback.with_structured_output(schema, **kwargs)

    def invoke(self, messages, **kwargs):
        try:
            return self._primary.invoke(messages, **kwargs)
        except Exception as e:
            logger.warning("Primary structured LLM failed, trying fallback", extra={"extra_data": {"error": str(e)[:200]}})
            try:
                return self._fallback.invoke(messages, **kwargs)
            except Exception:
                logger.error("Both primary and fallback structured LLM failed")
                raise


def get_llm() -> LLMWithFallback:
    return LLMWithFallback()


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
