from __future__ import annotations

import os
from pathlib import Path

from catalog.services.ingestion import ingest_catalog, upsert_categories
from client.services.query import handle_query
from shared.db.models import Category
from shared.db.postgres import get_session
from shared.db.qdrant import QdrantDB
from shared.services.embedding import ensure_qdrant_collections, upsert_category_embeddings, upsert_product_embeddings
from shared.services.matcher import process_supplier_products

from test.benchmarks.data import load_categories


def test_e2e() -> None:
    QdrantDB(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    ensure_qdrant_collections()

    with get_session() as session:
        categories_payload = load_categories("categories_base.json")
        upsert_categories(session, categories_payload)

        categories = session.query(Category).all()
        upsert_category_embeddings(categories)

        products = ingest_catalog(
            session=session,
            file_path=str(Path("test/data/input/toys_simplified.csv")),
            supplier_name="E2E Supplier",
            meta_json='{"source":"e2e_test"}',
        )
        upsert_product_embeddings(products)

        process_supplier_products(
            session=session,
            qdrant=qdrant,
            products=products,
            top_k=5,
        )

    results = handle_query("игрушки для детей", user_id=12345)

    assert isinstance(results, list), "Поиск должен вернуть список"