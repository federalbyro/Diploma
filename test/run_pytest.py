from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pytest
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[1]


METRICS_ROOT = ROOT / "metrics"
TEST_ROOT = ROOT / "test"



TEST_DB_NAME = os.getenv("TEST_DB_NAME", "diploma_test")
TEST_DB_HOST = os.getenv("TEST_DB_HOST", "postgres")
TEST_DB_PORT = os.getenv("TEST_DB_PORT", "5432")
TEST_DB_USER = os.getenv("TEST_DB_USER", "postgres")
TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD", "postgres")

TEST_POSTGRES_DSN = (
    f"postgresql+psycopg2://{TEST_DB_USER}:{TEST_DB_PASSWORD}"
    f"@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
)

ADMIN_POSTGRES_DSN = (
    f"postgresql+psycopg2://{TEST_DB_USER}:{TEST_DB_PASSWORD}"
    f"@{TEST_DB_HOST}:{TEST_DB_PORT}/postgres"
)

TEST_QDRANT_URL = os.getenv("TEST_QDRANT_URL", "http://qdrant:6333")

TEST_PRODUCT_COLLECTION = "test_product_embeddings"
TEST_CATEGORY_COLLECTION = "test_categories_embeddings"
TEST_SEARCH_COLLECTION = "test_search_embeddings"

CATALOG_GT_PATH = METRICS_ROOT / "catalog_ground_truth_by_file.json"
QUERY_GT_PATH = METRICS_ROOT / "query_ground_truth_extended.json"
CLIENT_QUERIES_PATH = TEST_ROOT / "data" / "client" / "client_queries_extended.json"

_CLIENT_SEARCH_CACHE: list[dict[str, Any]] | None = None


def configure_env() -> None:
    os.environ["APP_ENV"] = "test"
    os.environ["POSTGRES_DSN"] = TEST_POSTGRES_DSN
    os.environ["QDRANT_URL"] = TEST_QDRANT_URL
    os.environ["QDRANT_PRODUCTS_COLLECTION"] = TEST_PRODUCT_COLLECTION
    os.environ["QDRANT_CATEGORIES_COLLECTION"] = TEST_CATEGORY_COLLECTION
    os.environ["QDRANT_SEARCH_COLLECTION"] = TEST_SEARCH_COLLECTION
    os.environ.setdefault("MODEL_CACHE_DIR", "/models")
    os.environ.setdefault("MODEL_DEVICE", "cpu")
    os.environ.setdefault("ENCODER_LOG", "0")


def ensure_test_database() -> None:
    engine = create_engine(ADMIN_POSTGRES_DSN, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
            {"db_name": TEST_DB_NAME},
        ).scalar()

        if not exists:
            conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))

    engine.dispose()


def reset_test_database() -> None:
    engine = create_engine(TEST_POSTGRES_DSN, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO postgres"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))

    engine.dispose()


def run_migrations() -> None:
    env = os.environ.copy()
    env["POSTGRES_DSN"] = TEST_POSTGRES_DSN
    env["PYTHONPATH"] = str(ROOT)

    subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=True,
    )


def reset_qdrant() -> None:
    from shared.db.qdrant import QdrantDB

    qdrant = QdrantDB(url=TEST_QDRANT_URL)

    qdrant.delete_collection_if_exists(TEST_PRODUCT_COLLECTION)
    qdrant.delete_collection_if_exists(TEST_CATEGORY_COLLECTION)
    qdrant.delete_collection_if_exists(TEST_SEARCH_COLLECTION)


def print_section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("Нет данных")
        return

    columns = list(rows[0].keys())
    widths = {
        col: max(len(str(col)), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }

    header = " | ".join(str(col).ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)

    print(header)
    print(sep)

    for row in rows:
        print(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_all(sql: str) -> list[dict[str, Any]]:
    engine = create_engine(TEST_POSTGRES_DSN)

    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = [dict(row._mapping) for row in result]

    engine.dispose()
    return rows


def qdrant_count(collection_name: str) -> int:
    from shared.db.qdrant import QdrantDB

    qdrant = QdrantDB(url=TEST_QDRANT_URL)

    try:
        result = qdrant.client.count(
            collection_name=collection_name,
            exact=True,
        )
        return int(result.count)
    except Exception:
        return 0


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 6)


def top_k_accuracy(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0

    return round(
        sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks),
        6,
    )


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0

    total = 0.0

    for rank in ranks:
        if rank is not None:
            total += 1.0 / rank

    return round(total / len(ranks), 6)


def rank_min(ranks: list[int | None]) -> int | str:
    existing = [rank for rank in ranks if rank is not None]
    return min(existing) if existing else "-"


def rank_max(ranks: list[int | None]) -> int | str:
    existing = [rank for rank in ranks if rank is not None]
    return max(existing) if existing else "-"


def value_for_sort(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "-":
        return float("-inf")
    return float(value)


def print_best_worst(
    title: str,
    rows: list[dict[str, Any]],
    group_key: str,
    metrics: list[str],
) -> None:
    print_section(title)

    if not rows:
        print("Нет данных")
        return

    result_rows: list[dict[str, Any]] = []

    for metric in metrics:
        valid_rows = [
            row for row in rows
            if row.get(metric) is not None and row.get(metric) != "-"
        ]

        if not valid_rows:
            continue

        best = max(valid_rows, key=lambda row: value_for_sort(row, metric))
        worst = min(valid_rows, key=lambda row: value_for_sort(row, metric))

        result_rows.append(
            {
                "metric": metric,
                "best_group": best[group_key],
                "best_value": best[metric],
                "worst_group": worst[group_key],
                "worst_value": worst[metric],
            }
        )

    print_table(result_rows)


def calculate_catalog_metrics_by_file() -> list[dict[str, Any]]:
    if not CATALOG_GT_PATH.exists():
        print(f"[metrics] file not found: {CATALOG_GT_PATH}")
        return []

    expected_by_file: dict[str, list[str]] = load_json(CATALOG_GT_PATH)

    rows = fetch_all(
        """
        SELECT
            p.product_id,
            p.source_file,
            c.category_name_ru,
            m.rank,
            m.similarity_score
        FROM product_category_match m
        JOIN products p ON p.product_id = m.product_id
        JOIN categories c ON c.category_id = m.category_id
        ORDER BY p.product_id, m.rank;
        """
    )

    grouped_by_product: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped_by_product[int(row["product_id"])].append(row)

    product_groups_by_file: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)

    for product_matches in grouped_by_product.values():
        if not product_matches:
            continue

        source_file = str(product_matches[0]["source_file"])
        product_groups_by_file[source_file].append(product_matches)

    metric_rows: list[dict[str, Any]] = []

    all_ranks: list[int | None] = []
    all_relevant_scores: list[float] = []
    all_non_relevant_scores: list[float] = []

    for source_file, products_matches in sorted(product_groups_by_file.items()):
        expected_categories = set(expected_by_file.get(source_file, []))

        ranks: list[int | None] = []
        relevant_scores: list[float] = []
        non_relevant_scores: list[float] = []

        for matches in products_matches:
            first_relevant_rank: int | None = None

            for match in matches:
                category_name = str(match["category_name_ru"])
                rank = int(match["rank"])
                score = float(match["similarity_score"])

                if category_name in expected_categories:
                    relevant_scores.append(score)

                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                else:
                    non_relevant_scores.append(score)

            ranks.append(first_relevant_rank)

        avg_relevant = safe_mean(relevant_scores)
        avg_non_relevant = safe_mean(non_relevant_scores)

        cosmet_gap = None
        if avg_relevant is not None and avg_non_relevant is not None:
            cosmet_gap = round(avg_relevant - avg_non_relevant, 6)

        metric_rows.append(
            {
                "source_file": source_file,
                "products": len(ranks),
                "found": sum(1 for rank in ranks if rank is not None),
                "top_1_accuracy": top_k_accuracy(ranks, 1),
                "top_3_accuracy": top_k_accuracy(ranks, 3),
                "top_5_accuracy": top_k_accuracy(ranks, 5),
                "mrr": mean_reciprocal_rank(ranks),
                "avg_relevant_cos": avg_relevant,
                "avg_non_relevant_cos": avg_non_relevant,
                "cosmet_gap": cosmet_gap,
                "best_rank": rank_min(ranks),
                "worst_rank": rank_max(ranks),
            }
        )

        all_ranks.extend(ranks)
        all_relevant_scores.extend(relevant_scores)
        all_non_relevant_scores.extend(non_relevant_scores)

    overall_relevant = safe_mean(all_relevant_scores)
    overall_non_relevant = safe_mean(all_non_relevant_scores)

    overall_gap = None
    if overall_relevant is not None and overall_non_relevant is not None:
        overall_gap = round(overall_relevant - overall_non_relevant, 6)

    metric_rows.insert(
        0,
        {
            "source_file": "OVERALL",
            "products": len(all_ranks),
            "found": sum(1 for rank in all_ranks if rank is not None),
            "top_1_accuracy": top_k_accuracy(all_ranks, 1),
            "top_3_accuracy": top_k_accuracy(all_ranks, 3),
            "top_5_accuracy": top_k_accuracy(all_ranks, 5),
            "mrr": mean_reciprocal_rank(all_ranks),
            "avg_relevant_cos": overall_relevant,
            "avg_non_relevant_cos": overall_non_relevant,
            "cosmet_gap": overall_gap,
            "best_rank": rank_min(all_ranks),
            "worst_rank": rank_max(all_ranks),
        },
    )

    return metric_rows


def print_category_ranking_metrics() -> None:
    print_section("CATEGORY RANKING METRICS")

    metric_rows = calculate_catalog_metrics_by_file()
    print_table(metric_rows)

    per_file_rows = [
        row for row in metric_rows
        if row.get("source_file") != "OVERALL"
    ]

    print_best_worst(
        title="CATEGORY RANKING BEST / WORST",
        rows=per_file_rows,
        group_key="source_file",
        metrics=[
            "top_1_accuracy",
            "top_3_accuracy",
            "top_5_accuracy",
            "mrr",
            "avg_relevant_cos",
            "cosmet_gap",
        ],
    )


def load_client_queries() -> list[dict[str, Any]]:
    if CLIENT_QUERIES_PATH.exists():
        return load_json(CLIENT_QUERIES_PATH)

    return [
        {"user_id": 999, "query_text": "детские игрушки"},
        {"user_id": 999, "query_text": "авиационные детали"},
        {"user_id": 999, "query_text": "сантехнические товары"},
    ]


def get_client_search_runs() -> list[dict[str, Any]]:
    global _CLIENT_SEARCH_CACHE

    if _CLIENT_SEARCH_CACHE is not None:
        return _CLIENT_SEARCH_CACHE

    from client.services.query import handle_query

    runs: list[dict[str, Any]] = []

    for item in load_client_queries():
        query_text = item["query_text"]
        user_id = item.get("user_id", 999)

        results = handle_query(
            query_text=query_text,
            user_id=user_id,
        )

        runs.append(
            {
                "query_text": query_text,
                "user_id": user_id,
                "results": results,
            }
        )

    _CLIENT_SEARCH_CACHE = runs
    return runs


def calculate_client_search_metrics() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not QUERY_GT_PATH.exists():
        print(f"[metrics] file not found: {QUERY_GT_PATH}")
        return [], []

    expected_items = load_json(QUERY_GT_PATH)

    expected_by_query = {
        str(item["query_text"]): set(item["expected_supplier_names"])
        for item in expected_items
    }

    detail_rows: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    relevant_scores: list[float] = []
    non_relevant_scores: list[float] = []

    for run in get_client_search_runs():
        query_text = run["query_text"]
        results = run["results"]

        expected_suppliers = expected_by_query.get(query_text, set())

        first_expected_rank: int | None = None
        relevant_score: float | None = None
        best_non_relevant_score: float | None = None

        top_supplier = None
        top_score = None

        if results:
            top_supplier = results[0].get("supplier_name")
            top_score = round(float(results[0].get("score", 0)), 6)

        for rank, result in enumerate(results, start=1):
            supplier_name = result.get("supplier_name")
            score = float(result.get("score", 0))

            if supplier_name in expected_suppliers:
                relevant_scores.append(score)

                if first_expected_rank is None:
                    first_expected_rank = rank
                    relevant_score = score
            else:
                non_relevant_scores.append(score)

                if best_non_relevant_score is None or score > best_non_relevant_score:
                    best_non_relevant_score = score

        ranks.append(first_expected_rank)

        reciprocal_rank = 0.0
        if first_expected_rank is not None:
            reciprocal_rank = round(1.0 / first_expected_rank, 6)

        cosmet_gap = None
        if relevant_score is not None and best_non_relevant_score is not None:
            cosmet_gap = round(relevant_score - best_non_relevant_score, 6)

        detail_rows.append(
            {
                "query": query_text[:60],
                "expected_rank": first_expected_rank if first_expected_rank is not None else "-",
                "reciprocal_rank": reciprocal_rank,
                "top_supplier": top_supplier,
                "top_score": top_score,
                "relevant_score": round(relevant_score, 6) if relevant_score is not None else "-",
                "best_non_relevant": (
                    round(best_non_relevant_score, 6)
                    if best_non_relevant_score is not None
                    else "-"
                ),
                "cosmet_gap": cosmet_gap if cosmet_gap is not None else "-",
            }
        )

    avg_relevant = safe_mean(relevant_scores)
    avg_non_relevant = safe_mean(non_relevant_scores)

    overall_gap = None
    if avg_relevant is not None and avg_non_relevant is not None:
        overall_gap = round(avg_relevant - avg_non_relevant, 6)

    summary_rows = [
        {
            "metric": "queries_count",
            "value": len(ranks),
        },
        {
            "metric": "top_1_accuracy",
            "value": top_k_accuracy(ranks, 1),
        },
        {
            "metric": "top_3_accuracy",
            "value": top_k_accuracy(ranks, 3),
        },
        {
            "metric": "top_5_accuracy",
            "value": top_k_accuracy(ranks, 5),
        },
        {
            "metric": "mrr",
            "value": mean_reciprocal_rank(ranks),
        },
        {
            "metric": "avg_relevant_score",
            "value": avg_relevant,
        },
        {
            "metric": "avg_non_relevant_score",
            "value": avg_non_relevant,
        },
        {
            "metric": "cosmet_gap",
            "value": overall_gap,
        },
    ]

    return summary_rows, detail_rows


def print_client_search_metrics() -> None:
    print_section("CLIENT SEARCH METRICS")

    summary_rows, detail_rows = calculate_client_search_metrics()

    print_table(summary_rows)

    print_best_worst(
        title="CLIENT SEARCH BEST / WORST",
        rows=detail_rows,
        group_key="query",
        metrics=[
            "reciprocal_rank",
            "relevant_score",
            "cosmet_gap",
        ],
    )

    print_section("CLIENT SEARCH DETAILS")
    print_table(detail_rows)


def print_database_summary() -> None:
    print_section("POSTGRES SUMMARY")

    rows = fetch_all(
        """
        SELECT 'suppliers' AS table_name, COUNT(*) AS rows FROM suppliers
        UNION ALL
        SELECT 'products', COUNT(*) FROM products
        UNION ALL
        SELECT 'categories', COUNT(*) FROM categories
        UNION ALL
        SELECT 'product_category_match', COUNT(*) FROM product_category_match
        UNION ALL
        SELECT 'supplier_category_mapping', COUNT(*) FROM supplier_category_mapping
        ORDER BY table_name;
        """
    )

    print_table(rows)


def print_qdrant_summary() -> None:
    print_section("QDRANT SUMMARY")

    rows = [
        {
            "collection": TEST_PRODUCT_COLLECTION,
            "points": qdrant_count(TEST_PRODUCT_COLLECTION),
        },
        {
            "collection": TEST_CATEGORY_COLLECTION,
            "points": qdrant_count(TEST_CATEGORY_COLLECTION),
        },
        {
            "collection": TEST_SEARCH_COLLECTION,
            "points": qdrant_count(TEST_SEARCH_COLLECTION),
        },
    ]

    print_table(rows)


def print_catalog_file_summary() -> None:
    print_section("CATALOG FILE RESULTS")

    rows = fetch_all(
        """
        SELECT
            p.source_file,
            COUNT(DISTINCT p.product_id) AS products,
            COUNT(m.match_id) AS matches,
            ROUND(AVG(m.similarity_score)::numeric, 6) AS avg_similarity,
            ROUND(MIN(m.similarity_score)::numeric, 6) AS min_similarity,
            ROUND(MAX(m.similarity_score)::numeric, 6) AS max_similarity
        FROM products p
        LEFT JOIN product_category_match m ON m.product_id = p.product_id
        GROUP BY p.source_file
        ORDER BY p.source_file;
        """
    )

    print_table(rows)


def print_supplier_summary() -> None:
    print_section("SUPPLIER RESULTS")

    rows = fetch_all(
        """
        SELECT
            s.supplier_id,
            s.name AS supplier_name,
            s.product_count,
            COUNT(DISTINCT m.match_id) AS matches,
            ROUND(AVG(m.similarity_score)::numeric, 6) AS avg_similarity,
            ROUND(MAX(m.similarity_score)::numeric, 6) AS best_similarity
        FROM suppliers s
        LEFT JOIN product_category_match m ON m.supplier_id = s.supplier_id
        GROUP BY s.supplier_id, s.name, s.product_count
        ORDER BY s.supplier_id;
        """
    )

    print_table(rows)


def print_similarity_by_rank() -> None:
    print_section("COSINE SIMILARITY BY RANK")

    rows = fetch_all(
        """
        SELECT
            rank,
            COUNT(*) AS rows,
            ROUND(AVG(similarity_score)::numeric, 6) AS avg_similarity,
            ROUND(MIN(similarity_score)::numeric, 6) AS min_similarity,
            ROUND(MAX(similarity_score)::numeric, 6) AS max_similarity
        FROM product_category_match
        GROUP BY rank
        ORDER BY rank;
        """
    )

    print_table(rows)


def print_similarity_by_supplier_category() -> None:
    print_section("COSINE SIMILARITY BY SUPPLIER AND CATEGORY")

    rows = fetch_all(
        """
        SELECT
            s.name AS supplier_name,
            c.category_name_ru AS category,
            COUNT(*) AS matches,
            ROUND(AVG(m.similarity_score)::numeric, 6) AS avg_similarity,
            ROUND(MAX(m.similarity_score)::numeric, 6) AS best_similarity
        FROM product_category_match m
        JOIN suppliers s ON s.supplier_id = m.supplier_id
        JOIN categories c ON c.category_id = m.category_id
        GROUP BY s.name, c.category_name_ru
        ORDER BY s.name, avg_similarity DESC;
        """
    )

    print_table(rows)


def print_top1_samples() -> None:
    print_section("TOP-1 MATCH SAMPLES")

    rows = fetch_all(
        """
        SELECT
            p.source_file,
            LEFT(p.original_description, 80) AS product_text,
            c.category_name_ru AS top_category,
            ROUND(m.similarity_score::numeric, 6) AS cosine_similarity
        FROM product_category_match m
        JOIN products p ON p.product_id = m.product_id
        JOIN categories c ON c.category_id = m.category_id
        WHERE m.rank = 1
        ORDER BY p.product_id
        LIMIT 20;
        """
    )

    print_table(rows)


def print_search_results() -> None:
    print_section("CLIENT SEARCH RESULTS")

    for run in get_client_search_runs():
        query = run["query_text"]
        results = run["results"]

        print()
        print(f"QUERY: {query}")

        rows = []

        for idx, item in enumerate(results[:5], start=1):
            rows.append(
                {
                    "rank": idx,
                    "supplier_name": item.get("supplier_name"),
                    "score": round(float(item.get("score", 0)), 6),
                    "avg_similarity": round(float(item.get("avg_similarity", 0)), 6),
                    "best_similarity": round(float(item.get("best_similarity", 0)), 6),
                    "matched_products": len(item.get("matched_product_ids", [])),
                }
            )

        print_table(rows)


def print_final_report() -> None:
    # Сначала метрики для главы 3.
    print_category_ranking_metrics()
    print_client_search_metrics()

    # Потом технические таблицы, подтверждающие состояние данных.
    print_database_summary()
    print_qdrant_summary()
    print_catalog_file_summary()
    print_supplier_summary()
    print_similarity_by_rank()
    print_similarity_by_supplier_category()
    print_top1_samples()
    print_search_results()


def main() -> int:
    configure_env()

    print("[test-runner] using database:", TEST_POSTGRES_DSN, flush=True)
    print("[test-runner] using qdrant:", TEST_QDRANT_URL, flush=True)

    ensure_test_database()
    reset_test_database()
    reset_qdrant()
    run_migrations()

    exit_code = pytest.main(
        [
            "-q",
            "-s",
            "--disable-warnings",
            "test/integration/test_catalog_pipeline.py",
            "test/integration/test_search_pipeline.py",
            "test/integration/test_client_service_api.py",
        ]
    )

    print_final_report()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())