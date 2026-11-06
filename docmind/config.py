from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_provider: str = "huggingface"  # huggingface | openai | anthropic
    hf_llm_model: str = "google/flan-t5-small"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    top_k_retrieve: int = 20
    top_n_context: int = 4

    docs_dir: str = "data/documents"
    index_dir: str = "data/index"

    @property
    def docs_path(self) -> Path:
        return Path(self.docs_dir)

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)


settings = Settings()
