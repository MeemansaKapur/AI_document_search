"""Application settings, read from environment variables (or a local .env file)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MongoDB Atlas (or the mongodb-atlas-local Docker image for experiments)
    mongodb_uri: str = "mongodb://localhost:27017/?directConnection=true"
    mongodb_db: str = "docsearch"
    vector_index: str = "vector_index"
    text_index: str = "text_index"

    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    chat_model: str = "gpt-4o-mini"  # any small, cheap chat model; check the current model list
    answer_max_tokens: int = 300
    answer_temperature: float = 0.1

    # Retrieval
    num_candidates_multiplier: int = 15  # numCandidates = multiplier * limit
    rrf_k: int = 60
    w_vector: float = 1.0
    w_text: float = 1.0
    query_cache_size: int = 1024

    # /ask
    ask_top_k: int = 5
    # Atlas reports cosine similarity normalized to [0, 1] as (1 + cos) / 2.
    # Calibrate this on the dev split: python -m app.evaluate --calibrate
    abstain_threshold: float = 0.65
    ask_rate_limit_per_min: int = 10
    trust_forwarded_for: bool = False  # true only behind a proxy (Caddy) that sets X-Forwarded-For
    log_questions: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
