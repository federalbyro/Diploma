from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_env: Literal['dev', 'test', 'prod'] = 'dev'
    app_name: str = 'labse-portal'
    log_level: str = 'INFO'

    catalog_api_prefix: str = '/catalog'
    client_api_prefix: str = '/search'

    postgres_dsn: str = Field(default='postgresql+psycopg2://postgres:postgres@postgres:5432/diploma')

    qdrant_url: str = 'http://qdrant:6333'
    qdrant_api_key: str | None = None
    qdrant_products_collection: str = 'product_embeddings'
    qdrant_categories_collection: str = 'categories_embeddings'
    qdrant_search_collection: str = 'search_embeddings'
    embedding_size: int = 768
    qdrant_product_batch_size: int = 128
    qdrant_category_batch_size: int = 128

    redis_url: str = 'redis://redis:6379/0'
    celery_broker_url: str = 'redis://redis:6379/0'
    celery_result_backend: str = 'redis://redis:6379/1'

    model_name: str = 'sentence-transformers/LaBSE'
    model_device: str = 'cpu'
    model_batch_size: int = 16
    model_cache_dir: str = '/models'
    model_normalize_embeddings: bool = True
    model_max_text_length: int = 1024

    category_top_k: int = 5
    product_search_limit: int = 20
    supplier_top_k: int = 10
    supplier_min_similarity: float = 0.20

    upload_dir: str = '/tmp/uploads'
    allowed_extensions: tuple[str, ...] = ('.csv', '.xlsx', '.xls', '.txt')

    def ensure_dirs(self) -> None:
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)

    @validator("allowed_extensions", pre=True)
    def check_allowed_extensions(cls, value):
        """Ensure that allowed extensions are set correctly."""
        if not value:
            raise ValueError("allowed_extensions must not be empty")
        for ext in value:
            if not ext.startswith('.'):
                raise ValueError(f"Invalid extension format: {ext}")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings