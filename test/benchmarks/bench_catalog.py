from __future__ import annotations
import os
from pathlib import Path

from catalog.services.ingestion import ingest_catalog, upsert_categories
from shared.db.models import Category, Product, ProductCategoryMatch, SupplierCategoryMapping
from shared.db.postgres import get_session
from shared.db.qdrant import QdrantDB
from shared.services.embedding import (
    ensure_qdrant_collections,
    upsert_category_embeddings,
    upsert_product_embeddings,
)
from shared.services.matcher import process_supplier_products
from test.benchmarks.data import load_categories
from test.benchmarks.timer import get_vram_stats, print_dict, print_stage, timed


def run_catalog_benchmark(
    catalog_path: str | Path,
    supplier_name: str,
    categories_file: str = "categories_base.json",
    meta_json: str | None = None,
) -> dict:
    catalog_path = Path(catalog_path)

    qdrant = QdrantDB(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    ensure_qdrant_collections()

    result: dict = {
        "file_name": catalog_path.name,
        "supplier_name": supplier_name,
        "rows": 0,
        "categories": 0,
        "ingest_sec": 0.0,
        "category_index_sec": 0.0,
        "product_index_sec": 0.0,
        "match_sec": 0.0,
        "total_sec": 0.0,
        "rows_per_sec": 0.0,
        "product_count": 0,
        "match_count": 0,
        "supplier_category_count": 0,
    }

    with timed("total") as total_timer:
        with get_session() as session:
            categories_payload = load_categories(categories_file)

            with timed("upsert_categories", rows=len(categories_payload)) as t:
                categories = upsert_categories(session, categories_payload)
            result["categories"] = len(categories)
            result["category_upsert_sec"] = t.seconds
            print_stage("upsert_categories", t.seconds, len(categories_payload))

            all_categories = session.query(Category).all()
            with timed("index_categories", rows=len(all_categories)) as t:
                upsert_category_embeddings(all_categories)
            result["category_index_sec"] = t.seconds
            print_stage("index_categories", t.seconds, len(all_categories))

            with timed("ingest_catalog") as t:
                products = ingest_catalog(
                    session=session,
                    file_path=str(catalog_path),
                    supplier_name=supplier_name,
                    meta_json=meta_json,
                )
            result["rows"] = len(products)
            result["product_count"] = len(products)
            result["ingest_sec"] = t.seconds
            print_stage("ingest_catalog", t.seconds, len(products))

            with timed("index_products", rows=len(products)) as t:
                upsert_product_embeddings(products)
            result["product_index_sec"] = t.seconds
            print_stage("index_products", t.seconds, len(products))

            with timed("match_products", rows=len(products)) as t:
                process_supplier_products(
                    session=session,
                    qdrant=qdrant,
                    products=products,
                    top_k=5,
                )
            result["match_sec"] = t.seconds
            print_stage("match_products", t.seconds, len(products))

            match_count = session.query(ProductCategoryMatch).count()
            supplier_category_count = session.query(SupplierCategoryMapping).count()

            result["match_count"] = match_count
            result["supplier_category_count"] = supplier_category_count

        result["total_sec"] = total_timer.seconds
        result["rows_per_sec"] = (
            result["rows"] / result["total_sec"] if result["total_sec"] > 0 else 0.0
        )

    result.update(get_vram_stats())
    print_dict("\n[bench] Catalog benchmark summary", result)
    return result


if __name__ == "__main__":
    sample = Path("test/data/input/toys_simplified.csv")
    run_catalog_benchmark(
        catalog_path=sample,
        supplier_name="Benchmark Supplier",
        categories_file="categories_base.json",
    )