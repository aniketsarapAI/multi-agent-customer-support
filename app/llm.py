import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    DOTENV_PATH,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY_ENV,
)

load_dotenv(DOTENV_PATH)

_API_KEY = os.getenv(OPENROUTER_API_KEY_ENV)


def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url=OPENROUTER_BASE_URL,
        api_key=_API_KEY,
        streaming=True,
    )


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
