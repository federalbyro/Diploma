from __future__ import annotations

from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from shared.core.config import get_settings

settings = get_settings()

PRODUCT_COLLECTION = settings.qdrant_products_collection
CATEGORY_COLLECTION = settings.qdrant_categories_collection
SEARCH_COLLECTION = settings.qdrant_search_collection


class QdrantDB:
    def __init__(self, url: str | None = None):
        self.client = QdrantClient(url=url or settings.qdrant_url)

    def create_collection_if_not_exists(
        self,
        collection_name: str,
        vector_size: int,
    ) -> None:
        collections = self.client.get_collections().collections
        existing_names = {c.name for c in collections}
        if collection_name in existing_names:
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    
    def scroll_all(
        self,
        collection_name: str,
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> list:
        """Получить все точки из коллекции постранично."""
        all_points = []
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_vectors=with_vectors,
                with_payload=with_payload,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset
        return all_points
    
    def upsert_points(
        self,
        collection_name: str,
        ids: Sequence[int | str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")

        if payloads is not None and len(payloads) != len(vectors):
            raise ValueError("payloads and vectors must have the same length")

        points: list[PointStruct] = []
        for i, vector in enumerate(vectors):
            payload = payloads[i] if payloads is not None else {}
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=list(vector),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 5,
        query_filter: Any | None = None,
    ):
        return self.client.search(
            collection_name=collection_name,
            query_vector=list(query_vector),
            limit=limit,
            query_filter=query_filter,
        )

    def get_by_id(
        self,
        collection_name: str,
        point_id: int | str,
    ):
        result = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        return result[0] if result else None

    def delete_collection_if_exists(self, collection_name: str) -> None:
        collections = self.client.get_collections().collections
        existing_names = {c.name for c in collections}
        if collection_name in existing_names:
            self.client.delete_collection(collection_name=collection_name)