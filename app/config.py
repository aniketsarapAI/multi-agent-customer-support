from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    # OpenRouter / LLM
    openrouter_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"
    grader_model: str = "openai/gpt-4o-mini"
    llm_temperature: float = 0
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    # TiDB / MySQL
    mysql_host: str = "gateway01.eu-central-1.prod.aws.tidbcloud.com"
    mysql_port: int = 4000
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_database: str = "ecommerce_v2"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "selfrag"

    # Gmail SMTP escalation
    gmail_user: str = ""
    gmail_app_password: str = ""
    support_email: str = ""

    # Models
    fallback_model: str = "openai/gpt-4o-mini"

    # RAG
    chunk_size: int = 600
    chunk_overlap: int = 150
    retriever_k: int = 4
    max_retries: int = 5
    max_rewrite_tries: int = 3
    recursion_limit: int = 80

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 600
    redis_memory_ttl: int = 86400

    # Semantic cache
    semantic_cache_threshold: float = 0.78
    freq_questions_path: str = "freq_questions.json"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "selfrag"

    # Checkpointing
    checkpoint_backend: str = "sqlite"  # "sqlite" | "redis"
    checkpoint_db_path: str = "checkpoints.db"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
RETRIEVER_K = settings.retriever_k
LLM_MODEL = settings.llm_model
GRADER_MODEL = settings.grader_model
LLM_TEMPERATURE = settings.llm_temperature
EMBEDDING_MODEL = settings.embedding_model
OPENROUTER_BASE_URL = settings.openai_base_url
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
MAX_RETRIES = settings.max_retries
MAX_REWRITE_TRIES = settings.max_rewrite_tries
RECURSION_LIMIT = settings.recursion_limit
