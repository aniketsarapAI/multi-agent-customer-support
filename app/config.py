from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
RETRIEVER_K = 4

LLM_MODEL = "openai/gpt-4o-mini"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

MAX_RETRIES = 10
MAX_REWRITE_TRIES = 3

RECURSION_LIMIT = 80

DOTENV_PATH = BASE_DIR / ".env"


