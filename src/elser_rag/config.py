from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI / MeshAPI
    openai_api_key: str
    openai_base_url: str = "https://api.meshapi.ai/v1"
    openai_model: str = "openai/gpt-5.4"
    openai_enrichment_model: str = "openai/gpt-5.4-mini"

    # Elasticsearch
    elasticsearch_url: str = "http://elasticsearch:9200"
    elasticsearch_docs_index: str = "elser_rag_docs"
    elasticsearch_chunks_index: str = "elser_rag_chunks"
    elser_model_id: str = ".elser-model-2-linux-x86_64"

    # Ingest
    pdf_upload_dir: str = "/data/uploads"
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # Retrieval
    bm25_top_k: int = 20

    # Logging
    log_level: str = "INFO"


settings = Settings()
