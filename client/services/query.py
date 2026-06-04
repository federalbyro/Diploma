from __future__ import annotations

import os
import time

from shared.core.config import get_settings
from shared.db.qdrant import QdrantDB
from shared.services.embedding import upsert_search_embedding
from shared.services.matcher import search_suppliers_by_query
from shared.services.preprocessing import normalize_text

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")


def handle_query(
    query_text: str,
    user_id: int | str | None = None,
    limit: int | None = None,
) -> list[int]:
    """
    Обработка клиентского запроса:
    1. нормализуем текст
    2. сохраняем search embedding в Qdrant (аналитика)
    3. ищем ближайшие товары → уникальные supplier_id по порядку
    """
    prepared = normalize_text(query_text)
    if not prepared:
        return []

    settings = get_settings()
    qdrant = QdrantDB(url=QDRANT_URL)

    search_id = int(time.time() * 1000)
    upsert_search_embedding(
        search_id=search_id,
        query_text=prepared,
        user_id=user_id,
    )

    return search_suppliers_by_query(
        qdrant=qdrant,
        query_text=prepared,
        limit=limit or settings.supplier_top_k,
        product_search_limit=settings.product_search_limit,
    )