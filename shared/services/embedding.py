from __future__ import annotations

from datetime import datetime
import os
from typing import Any, Sequence

from qdrant_client.http import models as qdrant_models
from qdrant_client.models import PointStruct

from shared.core.config import get_settings
from shared.core.encoder import Encoder
from shared.db.models import Category, Product
from shared.db.qdrant import (
    CATEGORY_COLLECTION,
    PRODUCT_COLLECTION,
    SEARCH_COLLECTION,
    QdrantDB,
)
from shared.services.batching import iter_batches, flush_memory
from shared.services.preprocessing import build_category_text, normalize_text


QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "128"))
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))  # LaBSE = 768


encoder = Encoder()
qdrant = QdrantDB(url=QDRANT_URL)


def ensure_qdrant_collections() -> None:
    """
    Создаёт нужные коллекции в Qdrant, если они ещё не существуют.
    """
    qdrant.create_collection_if_not_exists(PRODUCT_COLLECTION, VECTOR_SIZE)
    qdrant.create_collection_if_not_exists(CATEGORY_COLLECTION, VECTOR_SIZE)
    qdrant.create_collection_if_not_exists(SEARCH_COLLECTION, VECTOR_SIZE)


def _upsert_in_batches(
    collection_name: str,
    ids: Sequence[int | str],
    vectors: Sequence[Sequence[float]],
    payloads: Sequence[dict],
) -> None:
    for batch_idx in iter_batches(list(range(len(ids))), QDRANT_UPSERT_BATCH_SIZE):
        qdrant.upsert_points(
            collection_name=collection_name,
            ids=[ids[i] for i in batch_idx],
            vectors=[vectors[i] for i in batch_idx],
            payloads=[payloads[i] for i in batch_idx],
        )


def _payload_by_category_name(
    category_payloads: Sequence[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for item in category_payloads or []:
        name = str(item.get("category_name_ru", "")).strip()
        if name:
            result[name] = item

    return result


def _category_description_texts(
    category: Category,
    payload: dict[str, Any] | None,
) -> list[str]:
    """
    Возвращает список текстов, по которым будут строиться эмбеддинги категории.

    Если пришёл массив category_descriptions, используем его.
    Если не пришёл — используем стандартное описание категории из PostgreSQL.
    """
    descriptions: list[str] = []

    if payload is not None:
        descriptions = [
            str(value).strip()
            for value in payload.get("category_descriptions") or []
            if str(value).strip()
        ]

    if not descriptions:
        descriptions = [category.category_description_ru]

    # В каждое описание добавляем название категории,
    # чтобы короткие описания не теряли контекст.
    return [
        build_category_text(category.category_name_ru, description)
        for description in descriptions
    ]


def _make_category_point_id(category_id: int, description_index: int) -> int:
    """
    Детерминированный id точки Qdrant для описания категории.

    category_id=12, description_index=3 -> 12000003
    """
    return (category_id * 1_000_000) + description_index


def _delete_category_points(category_ids: Sequence[int]) -> None:
    """
    Удаляет из Qdrant старые точки описаний для указанных категорий.
    Нужно, чтобы PATCH /categories не оставлял старые описания.
    """
    unique_ids = sorted(set(int(x) for x in category_ids))

    for category_id in unique_ids:
        qdrant.client.delete(
            collection_name=CATEGORY_COLLECTION,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="category_id",
                            match=qdrant_models.MatchValue(value=category_id),
                        )
                    ]
                )
            ),
            wait=True,
        )


def upsert_product_embeddings(products: Sequence[Product]) -> None:
    """
    Генерирует и сохраняет эмбеддинги товаров в коллекцию product_embeddings.
    """
    if not products:
        return

    ensure_qdrant_collections()

    all_ids: list[int] = []
    all_vectors: list[list[float]] = []
    all_payloads: list[dict] = []

    for batch in iter_batches(list(products), EMBEDDING_BATCH_SIZE):
        texts = [p.normalized_description for p in batch]
        vectors = encoder.encode_texts(texts, batch_size=EMBEDDING_BATCH_SIZE)

        for product, vector in zip(batch, vectors):
            all_ids.append(product.product_id)
            all_vectors.append(vector)
            all_payloads.append(
                {
                    "product_id": product.product_id,
                    "supplier_id": product.supplier_id,
                    "source_file": product.source_file,
                    "created_at": (
                        product.created_at.isoformat() if product.created_at else None
                    ),
                }
            )

        flush_memory()

    _upsert_in_batches(
        collection_name=PRODUCT_COLLECTION,
        ids=all_ids,
        vectors=all_vectors,
        payloads=all_payloads,
    )


def upsert_category_embeddings(
    categories: Sequence[Category],
    category_payloads: Sequence[dict[str, Any]] | None = None,
) -> None:
    """
    Генерирует и сохраняет эмбеддинги категорий в коллекцию categories_embeddings.

    Важная логика:
    - в PostgreSQL категория остаётся одной строкой;
    - в Qdrant у одной категории может быть несколько точек;
    - каждая точка соответствует отдельному описанию категории;
    - все точки одной категории имеют одинаковый category_id в payload.
    """
    if not categories:
        return

    ensure_qdrant_collections()

    payload_map = _payload_by_category_name(category_payloads)

    # Удаляем старые точки этих категорий, чтобы не оставались старые описания.
    _delete_category_points([category.category_id for category in categories])

    all_ids: list[int] = []
    all_vectors: list[list[float]] = []
    all_payloads: list[dict] = []

    category_items: list[tuple[Category, int, str]] = []

    for category in categories:
        payload = payload_map.get(category.category_name_ru)
        description_texts = _category_description_texts(category, payload)

        for description_index, text in enumerate(description_texts):
            category_items.append((category, description_index, text))

    for batch in iter_batches(category_items, EMBEDDING_BATCH_SIZE):
        texts = [item[2] for item in batch]
        vectors = encoder.encode_texts(texts, batch_size=EMBEDDING_BATCH_SIZE)

        for (category, description_index, text), vector in zip(batch, vectors):
            all_ids.append(
                _make_category_point_id(
                    category_id=category.category_id,
                    description_index=description_index,
                )
            )
            all_vectors.append(vector)
            all_payloads.append(
                {
                    "category_id": category.category_id,
                    "category_name_ru": category.category_name_ru,
                    "description_index": description_index,
                    "description_text": text,
                    "created_at": (
                        category.created_at.isoformat() if category.created_at else None
                    ),
                }
            )

        flush_memory()

    _upsert_in_batches(
        collection_name=CATEGORY_COLLECTION,
        ids=all_ids,
        vectors=all_vectors,
        payloads=all_payloads,
    )

def upsert_search_embedding_vector(
    search_id:  int,
    query_text: str,
    vector:     list[float],
    user_id:    int | str | None = None,
) -> None:
    """Сохраняет поисковый эмбеддинг в Qdrant. Вектор уже вычислен."""
    settings = get_settings()
    qdrant   = QdrantDB(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))

    ensure_qdrant_collections()

    qdrant.client.upsert(
        collection_name=settings.qdrant_search_collection,
        points=[
            PointStruct(
                id=search_id,
                vector=vector,
                payload={
                    "query_text": query_text,
                    "user_id":    str(user_id) if user_id is not None else None,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
        ],
    )

def upsert_search_embedding(
    search_id: int | str,
    query_text: str,
    user_id: int | str | None = None,
) -> list[float]:
    """
    Генерирует и сохраняет эмбеддинг поискового запроса в search_embeddings.
    Возвращает сам вектор, чтобы не кодировать запрос дважды.
    """
    ensure_qdrant_collections()

    normalized_query = normalize_text(query_text)
    vector = encoder.encode_single(normalized_query)

    payload = {
        "search_id": search_id,
        "user_id": user_id,
        "query_text": query_text,
        "normalized_query": normalized_query,
    }

    qdrant.upsert_points(
        collection_name=SEARCH_COLLECTION,
        ids=[search_id],
        vectors=[vector],
        payloads=[payload],
    )

    return vector