from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    langchain_api_key: str = Field("", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field("arxiv-rag-research-assistant", alias="LANGCHAIN_PROJECT")
    langchain_tracing_v2: str = Field("false", alias="LANGCHAIN_TRACING_V2")

    vector_store_path: str = Field("vector_stores/faiss_index", alias="VECTOR_STORE_PATH")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")
    chunk_size: int = Field(500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(50, alias="CHUNK_OVERLAP")
    retrieval_k: int = Field(5, alias="RETRIEVAL_K")


settings = Settings()
