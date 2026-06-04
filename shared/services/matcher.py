from __future__ import annotations

import os
from sqlalchemy.orm import Session

from shared.core.config import get_settings
from shared.core.encoder import Encoder
from shared.db.models import ProductCategoryMatch, SupplierCategoryMapping
from shared.db.qdrant import QdrantDB

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")


def search_suppliers_by_query(
    qdrant: QdrantDB,
    query_text: str,
    limit: int = 10,
    product_search_limit: int = 50,
) -> list[int]:
    """
    Возвращает упорядоченный список supplier_id.

    Алгоритм:
    1. Кодируем запрос → вектор
    2. Ищем top-product_search_limit товаров по косинусному сходству
    3. Идём по результатам в порядке убывания сходства
    4. Извлекаем supplier_id из payload, дедуплицируем
    5. Останавливаемся когда набрали `limit` уникальных поставщиков
    """
    settings = get_settings()
    encoder = Encoder()
    vector = encoder.encode_single(query_text)

    hits = qdrant.search(
        collection_name=settings.qdrant_products_collection,
        query_vector=vector,
        limit=product_search_limit,
    )

    seen: set[int] = set()
    result: list[int] = []

    for hit in hits:
        supplier_id = (hit.payload or {}).get("supplier_id")
        if supplier_id is None:
            continue
        sid = int(supplier_id)
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
            if len(result) >= limit:
                break

    return result

def search_suppliers_by_vector(
    qdrant:               QdrantDB,
    vector:               list[float],
    limit:                int = 10,
    product_search_limit: int = 50,
) -> list[int]:
    """Поиск поставщиков по готовому вектору (без повторного encode)."""
    settings = get_settings()

    hits = qdrant.client.search(
        collection_name=settings.qdrant_products_collection,
        query_vector=vector,
        limit=product_search_limit,
        with_payload=True,
    )

    seen:   set[int]  = set()
    result: list[int] = []

    for hit in hits:
        supplier_id = (hit.payload or {}).get("supplier_id")
        if supplier_id is None:
            continue
        sid = int(supplier_id)
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
            if len(result) >= limit:
                break

    return result

def process_supplier_products(
    session: Session,
    qdrant: QdrantDB,
    products: list,
    top_k: int = 5,
) -> None:
    """
    Семантический матчинг товаров с категориями + агрегация на уровне поставщика.
    """
    settings = get_settings()

    if not products:
        return

    encoder = Encoder()

    cat_hits = qdrant.scroll_all(
        collection_name=settings.qdrant_categories_collection,
        with_vectors=True,
        with_payload=True,
    )

    if not cat_hits:
        return

    cat_vectors = {
        int(h.payload["category_id"]): h.vector
        for h in cat_hits
        if h.payload and "category_id" in h.payload
    }

    if not cat_vectors:
        return

    import numpy as np
    cat_ids = list(cat_vectors.keys())
    cat_matrix = np.array([cat_vectors[cid] for cid in cat_ids], dtype=np.float32)

    # Матчим каждый товар
    matches_to_add = []
    supplier_cat_scores: dict[tuple[int, int], list[float]] = {}

    for product in products:
        p_vector = encoder.encode_single(product.normalized_description)
        p_vec = np.array(p_vector, dtype=np.float32)

        scores = (cat_matrix @ p_vec).tolist()
        ranked = sorted(zip(scores, cat_ids), reverse=True)[:top_k]

        for rank, (score, cat_id) in enumerate(ranked, start=1):
            matches_to_add.append(
                ProductCategoryMatch(
                    product_id=product.product_id,
                    supplier_id=product.supplier_id,
                    category_id=cat_id,
                    rank=rank,
                    similarity_score=round(score, 6),
                )
            )
            key = (product.supplier_id, cat_id)
            supplier_cat_scores.setdefault(key, []).append(score)

    session.bulk_save_objects(matches_to_add)

    for (supplier_id, category_id), scores in supplier_cat_scores.items():
        avg_sim = round(sum(scores) / len(scores), 6)
        mapping = SupplierCategoryMapping(
            supplier_id=supplier_id,
            category_id=category_id,
            product_count=len(scores),
            avg_similarity=avg_sim,
            score=avg_sim,  
        )
        session.merge(mapping)

    session.flush()