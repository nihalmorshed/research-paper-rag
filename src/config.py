"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Settings
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    default_llm: str = "ollama"  # "ollama" or "openai"
    ollama_model: str = "llama3.2"
    openai_model: str = "gpt-4o-mini"

    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    use_reranking: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
