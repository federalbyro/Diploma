from __future__ import annotations

import os
from pathlib import Path

from catalog.services.ingestion import ingest_catalog, upsert_categories
from shared.db.models import Category, Product, ProductCategoryMatch, Supplier, SupplierCategoryMapping
from shared.db.postgres import get_session
from shared.db.qdrant import CATEGORY_COLLECTION, PRODUCT_COLLECTION, QdrantDB
from shared.services.embedding import (
    ensure_qdrant_collections,
    upsert_category_embeddings,
    upsert_product_embeddings,
)
from shared.services.matcher import process_supplier_products
from test.benchmarks.data import load_categories


INPUT_ROOT = Path("test/data/input")
CATEGORY_FILE = "categories_extended.json"
TOP_K = 12


def _qdrant_count(qdrant: QdrantDB, collection_name: str) -> int:
    result = qdrant.client.count(
        collection_name=collection_name,
        exact=True,
    )
    return int(result.count)


def test_catalog_pipeline() -> None:
    qdrant = QdrantDB(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    ensure_qdrant_collections()

    catalog_files = sorted(
        file_path
        for file_path in INPUT_ROOT.glob("*")
        if file_path.suffix.lower() in {".csv", ".txt", ".xlsx", ".xls"}
    )

    assert catalog_files, "В test/data/input нет входных файлов каталогов"

    categories_payload = load_categories(CATEGORY_FILE)

    with get_session() as session:
        categories = upsert_categories(session, categories_payload)
        db_categories = session.query(Category).all()

        upsert_category_embeddings(
            db_categories,
            category_payloads=categories_payload,
        )

        total_loaded_products = 0

        for file_path in catalog_files:
            supplier_name = f"Integration Supplier {file_path.stem}"

            products = ingest_catalog(
                session=session,
                file_path=str(file_path),
                supplier_name=supplier_name,
                meta_json=f'{{"source_file":"{file_path.name}","source":"integration_test"}}',
            )

            assert len(products) > 0, f"Из файла {file_path.name} не загружены товары"

            total_loaded_products += len(products)

            upsert_product_embeddings(products)

            process_supplier_products(
                session=session,
                qdrant=qdrant,
                products=products,
                top_k=TOP_K,
            )

        supplier_count = session.query(Supplier).count()
        product_count = session.query(Product).count()
        category_count = session.query(Category).count()
        match_count = session.query(ProductCategoryMatch).count()
        mapping_count = session.query(SupplierCategoryMapping).count()

        assert supplier_count == len(catalog_files)
        assert product_count == total_loaded_products
        assert category_count == len(categories_payload)

        assert match_count >= total_loaded_products
        assert match_count <= total_loaded_products * TOP_K
        assert mapping_count > 0

        suppliers = session.query(Supplier).all()
        for supplier in suppliers:
            assert supplier.product_count > 0

    assert _qdrant_count(qdrant, PRODUCT_COLLECTION) == total_loaded_products
    
    expected_category_points = sum(
    max(1, len(item.get("category_descriptions", [])))
    for item in categories_payload
)

    assert _qdrant_count(qdrant, CATEGORY_COLLECTION) == expected_category_points