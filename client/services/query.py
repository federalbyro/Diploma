from __future__ import annotations

import os
import time

from shared.core.config import get_settings
from shared.core.encoder import Encoder
from shared.db.qdrant import QdrantDB
from shared.services.embedding import upsert_search_embedding_vector
from shared.services.matcher import search_suppliers_by_vector
from shared.services.preprocessing import normalize_text

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")


def handle_query(
    query_text: str,
    user_id:    int | str | None = None,
    limit:      int | None = None,
) -> list[int]:
    prepared = normalize_text(query_text)
    if not prepared:
        return []

    settings = get_settings()
    encoder  = Encoder()

    vector = encoder.encode_single(prepared)

    search_id = int(time.time() * 1000)
    upsert_search_embedding_vector(
        search_id=search_id,
        query_text=prepared,
        vector=vector,
        user_id=user_id,
    )

    qdrant = QdrantDB(url=QDRANT_URL)
    return search_suppliers_by_vector(
        qdrant=qdrant,
        vector=vector,
        limit=limit or settings.supplier_top_k,
        product_search_limit=settings.product_search_limit,
    )