from __future__ import annotations

import os
import time
from sqlalchemy import select

from shared.db.models import Supplier
from shared.db.postgres import get_session
from shared.db.qdrant import QdrantDB
from shared.services.embedding import upsert_search_embedding
from shared.services.matcher import search_suppliers_by_query
from shared.services.preprocessing import normalize_text


QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")


def handle_query(query_text: str, user_id: int | str | None = None) -> list[dict]:
    """
    Обработка клиентского запроса:
    1. нормализуем текст
    2. сохраняем search embedding в Qdrant
    3. ищем релевантных поставщиков через product_embeddings
    4. обогащаем результат данными из PostgreSQL
    """
    prepared = normalize_text(query_text)
    if not prepared:
        return []

    qdrant = QdrantDB(url=QDRANT_URL)

    # search_id пока делаем простым timestamp-based
    search_id = int(time.time() * 1000)
    upsert_search_embedding(
        search_id=search_id,
        query_text=prepared,
        user_id=user_id,
    )

    rows = search_suppliers_by_query(
        qdrant=qdrant,
        query_text=prepared,
        top_k_products=20,
    )

    supplier_ids = [row.supplier_id for row in rows]
    if not supplier_ids:
        return []

    with get_session() as session:
        suppliers = {
            s.supplier_id: s
            for s in session.execute(
                select(Supplier).where(Supplier.supplier_id.in_(supplier_ids))
            ).scalars().all()
        }

    result: list[dict] = []
    for row in rows:
        supplier = suppliers.get(row.supplier_id)

        result.append(
            {
                "supplier_id": row.supplier_id,
                "supplier_name": supplier.name if supplier else None,
                "score": row.score,
                "product_count": row.product_count,
                "avg_similarity": row.avg_similarity,
                "best_similarity": row.best_similarity,
                "matched_product_ids": row.matched_product_ids,
            }
        )

    return result