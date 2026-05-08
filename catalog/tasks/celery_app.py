from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from celery import Celery

from shared.db.models import Category, Job, JobStatus
from shared.db.postgres import get_session
from shared.db.qdrant import QdrantDB
from shared.services.embedding import (
    ensure_qdrant_collections,
    upsert_category_embeddings,
    upsert_product_embeddings,
)
from catalog.services.ingestion import ingest_catalog
from shared.services.matcher import process_supplier_products


CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

celery_app = Celery(
    "catalog_service",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)


@celery_app.task(name="catalog_service.process_catalog_file")
def process_catalog_file(
    job_id: int,
    file_path: str,
    supplier_name: str,
    meta_json: str | None = None,
) -> dict:
    """
    Полный pipeline обработки каталога:
    1. ставим job в processing
    2. ingest товаров в PostgreSQL
    3. индексируем товары в Qdrant
    4. индексируем категории в Qdrant
    5. product -> categories
    6. агрегируем supplier_category_mapping
    7. ставим job в done / failed
    8. при успехе удаляем исходный файл
    """
    qdrant = QdrantDB(url=QDRANT_URL)
    ensure_qdrant_collections()

    try:
        with get_session() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")

            job.status = JobStatus.processing.value
            session.flush()

            products = ingest_catalog(
                session=session,
                file_path=file_path,
                supplier_name=supplier_name,
                meta_json=meta_json,
            )

            if products:
                job.supplier_id = products[0].supplier_id

            upsert_product_embeddings(products)

            categories = session.query(Category).all()
            if categories:
                upsert_category_embeddings(categories)

            process_supplier_products(
                session=session,
                qdrant=qdrant,
                products=products,
                top_k=5,
            )

            job.status = JobStatus.done.value
            job.finished_at = datetime.utcnow()
            session.flush()

            result = {
                "job_id": job_id,
                "status": job.status,
                "supplier_name": supplier_name,
                "products_processed": len(products),
            }

        # файл удаляем только после успешного завершения транзакции
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass

        return result

    except Exception as exc:
        with get_session() as session:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = JobStatus.failed.value
                job.error_message = str(exc)
                job.finished_at = datetime.utcnow()
                session.flush()
        raise


@celery_app.task(name="catalog_service.reindex_categories")
def reindex_categories() -> dict:
    """
    Переиндексация всех категорий в Qdrant.
    """
    ensure_qdrant_collections()

    with get_session() as session:
        categories = session.query(Category).all()
        upsert_category_embeddings(categories)

    return {
        "status": "ok",
        "categories_indexed": len(categories),
    }