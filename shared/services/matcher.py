from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from sqlalchemy import delete
from sqlalchemy.orm import Session

from shared.db.models import (
    Product,
    ProductCategoryMatch,
    SupplierCategoryMapping,
)
from shared.db.qdrant import CATEGORY_COLLECTION, PRODUCT_COLLECTION, QdrantDB
from shared.services.embedding import encoder
from shared.services.preprocessing import normalize_text


CATEGORY_DESCRIPTION_SEARCH_LIMIT = int(
    os.getenv("CATEGORY_DESCRIPTION_SEARCH_LIMIT", "100")
)


@dataclass
class CategoryHit:
    category_id: int
    rank: int
    similarity_score: float
    category_name_ru: str | None = None
    description_index: int | None = None
    description_text: str | None = None


@dataclass
class SupplierResult:
    supplier_id: int
    product_count: int
    avg_similarity: float
    best_similarity: float
    score: float
    matched_product_ids: list[int]


def _collapse_category_description_hits(hits) -> list[CategoryHit]:
    """
    Qdrant возвращает точки описаний категорий.

    Если у одной категории найдено несколько описаний,
    оставляем только лучшее описание этой категории.
    """
    best_by_category: dict[int, CategoryHit] = {}

    for hit in hits:
        payload = hit.payload or {}

        category_id = payload.get("category_id")
        if category_id is None:
            continue

        category_id = int(category_id)
        similarity_score = float(hit.score)

        current = best_by_category.get(category_id)

        if current is None or similarity_score > current.similarity_score:
            best_by_category[category_id] = CategoryHit(
                category_id=category_id,
                rank=0,
                similarity_score=similarity_score,
                category_name_ru=payload.get("category_name_ru"),
                description_index=payload.get("description_index"),
                description_text=payload.get("description_text"),
            )

    collapsed = sorted(
        best_by_category.values(),
        key=lambda item: item.similarity_score,
        reverse=True,
    )

    for rank, item in enumerate(collapsed, start=1):
        item.rank = rank

    return collapsed


def match_product_to_categories(
    qdrant: QdrantDB,
    product: Product,
    top_k: int = 5,
) -> list[CategoryHit]:
    """
    Для одного товара:
    1. строит эмбеддинг normalized_description;
    2. ищет ближайшие описания категорий в Qdrant;
    3. схлопывает найденные описания по category_id;
    4. возвращает top-k уникальных категорий.
    """
    vector = encoder.encode_single(product.normalized_description)

    # Ищем описаний больше, чем нужно категорий.
    # Это нужно, потому что несколько верхних точек могут относиться к одной категории.
    raw_limit = max(CATEGORY_DESCRIPTION_SEARCH_LIMIT, top_k * 5)

    hits = qdrant.search(
        collection_name=CATEGORY_COLLECTION,
        query_vector=vector,
        limit=raw_limit,
    )

    collapsed_hits = _collapse_category_description_hits(hits)

    return collapsed_hits[:top_k]


def save_product_category_matches(
    session: Session,
    product: Product,
    category_hits: Sequence[CategoryHit],
) -> list[ProductCategoryMatch]:
    """
    Сохраняет результаты мэтчинга product -> top-k categories
    в таблицу product_category_match.
    """
    created: list[ProductCategoryMatch] = []

    session.execute(
        delete(ProductCategoryMatch).where(
            ProductCategoryMatch.product_id == product.product_id
        )
    )

    for hit in category_hits:
        row = ProductCategoryMatch(
            product_id=product.product_id,
            supplier_id=product.supplier_id,
            category_id=hit.category_id,
            rank=hit.rank,
            similarity_score=hit.similarity_score,
        )
        session.add(row)
        created.append(row)

    session.flush()
    return created


def aggregate_supplier_category_mapping(
    session: Session,
    supplier_id: int,
) -> list[SupplierCategoryMapping]:
    """
    Агрегирует все product_category_match поставщика до уровня
    supplier_category_mapping.

    Базовая логика score:
        score = avg_similarity * product_count
    """
    matches = (
        session.query(ProductCategoryMatch)
        .filter(ProductCategoryMatch.supplier_id == supplier_id)
        .all()
    )

    session.execute(
        delete(SupplierCategoryMapping).where(
            SupplierCategoryMapping.supplier_id == supplier_id
        )
    )

    grouped: dict[int, list[ProductCategoryMatch]] = defaultdict(list)
    for match in matches:
        grouped[match.category_id].append(match)

    created: list[SupplierCategoryMapping] = []

    for category_id, rows in grouped.items():
        similarity_values = [row.similarity_score for row in rows]
        product_count = len({row.product_id for row in rows})
        avg_similarity = float(mean(similarity_values)) if similarity_values else 0.0
        score = avg_similarity * product_count

        mapping = SupplierCategoryMapping(
            supplier_id=supplier_id,
            category_id=category_id,
            product_count=product_count,
            avg_similarity=avg_similarity,
            score=score,
        )
        session.add(mapping)
        created.append(mapping)

    session.flush()
    return created


def process_supplier_products(
    session: Session,
    qdrant: QdrantDB,
    products: Sequence[Product],
    top_k: int = 5,
) -> None:
    """
    Полный цикл для пачки товаров одного поставщика:
    1. product -> categories;
    2. запись product_category_match;
    3. агрегация supplier_category_mapping.
    """
    if not products:
        return

    supplier_id = products[0].supplier_id

    for product in products:
        hits = match_product_to_categories(
            qdrant=qdrant,
            product=product,
            top_k=top_k,
        )
        save_product_category_matches(
            session=session,
            product=product,
            category_hits=hits,
        )

    aggregate_supplier_category_mapping(
        session=session,
        supplier_id=supplier_id,
    )


def search_suppliers_by_query(
    qdrant: QdrantDB,
    query_text: str,
    top_k_products: int = 20,
) -> list[SupplierResult]:
    """
    Поиск поставщиков по пользовательскому запросу.

    Логика:
    1. нормализуем запрос;
    2. кодируем запрос;
    3. ищем top-N товаров в product_embeddings;
    4. группируем по supplier_id из payload;
    5. считаем score поставщика.
    """
    normalized_query = normalize_text(query_text)
    query_vector = encoder.encode_single(normalized_query)

    hits = qdrant.search(
        collection_name=PRODUCT_COLLECTION,
        query_vector=query_vector,
        limit=top_k_products,
    )

    grouped_scores: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for hit in hits:
        payload = hit.payload or {}
        supplier_id = payload.get("supplier_id")
        product_id = payload.get("product_id")

        if supplier_id is None or product_id is None:
            continue

        grouped_scores[int(supplier_id)].append(
            (int(product_id), float(hit.score))
        )

    results: list[SupplierResult] = []

    for supplier_id, rows in grouped_scores.items():
        scores = [score for _, score in rows]
        product_ids = [product_id for product_id, _ in rows]

        avg_similarity = float(mean(scores)) if scores else 0.0
        best_similarity = max(scores) if scores else 0.0

        score = (0.7 * best_similarity) + (0.3 * avg_similarity)

        results.append(
            SupplierResult(
                supplier_id=supplier_id,
                product_count=len(set(product_ids)),
                avg_similarity=avg_similarity,
                best_similarity=best_similarity,
                score=score,
                matched_product_ids=list(dict.fromkeys(product_ids)),
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)
    return results