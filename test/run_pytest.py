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

ROOT       = Path(__file__).resolve().parents[1]
METRICS_ROOT = ROOT / "metrics"
TEST_ROOT    = ROOT / "test"

TEST_DB_NAME     = os.getenv("TEST_DB_NAME",     "diploma_test")
TEST_DB_HOST     = os.getenv("TEST_DB_HOST",     "postgres")
TEST_DB_PORT     = os.getenv("TEST_DB_PORT",     "5432")
TEST_DB_USER     = os.getenv("TEST_DB_USER",     "postgres")
TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD", "postgres")

TEST_POSTGRES_DSN = (
    f"postgresql+psycopg2://{TEST_DB_USER}:{TEST_DB_PASSWORD}"
    f"@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
)
ADMIN_POSTGRES_DSN = (
    f"postgresql+psycopg2://{TEST_DB_USER}:{TEST_DB_PASSWORD}"
    f"@{TEST_DB_HOST}:{TEST_DB_PORT}/postgres"
)
TEST_QDRANT_URL         = os.getenv("TEST_QDRANT_URL", "http://qdrant:6333")
TEST_PRODUCT_COLLECTION  = "test_product_embeddings"
TEST_CATEGORY_COLLECTION = "test_categories_embeddings"
TEST_SEARCH_COLLECTION   = "test_search_embeddings"

CATALOG_GT_PATH       = METRICS_ROOT / "labels/catalog_ground_truth_by_file.json"
QUERY_GT_PATH         = METRICS_ROOT / "labels/query_ground_truth_extended.json"
CLIENT_QUERIES_PATH   = TEST_ROOT / "data" / "client" / "client_queries_extended.json"

_CLIENT_SEARCH_CACHE: list[dict[str, Any]] | None = None


def configure_env() -> None:
    os.environ["APP_ENV"]                    = "test"
    os.environ["POSTGRES_DSN"]               = TEST_POSTGRES_DSN
    os.environ["QDRANT_URL"]                 = TEST_QDRANT_URL
    os.environ["QDRANT_PRODUCTS_COLLECTION"] = TEST_PRODUCT_COLLECTION
    os.environ["QDRANT_CATEGORIES_COLLECTION"] = TEST_CATEGORY_COLLECTION
    os.environ["QDRANT_SEARCH_COLLECTION"]   = TEST_SEARCH_COLLECTION
    os.environ.setdefault("MODEL_CACHE_DIR", "/models")
    os.environ.setdefault("MODEL_DEVICE",    "cpu")
    os.environ.setdefault("ENCODER_LOG",     "0")


def ensure_test_database() -> None:
    engine = create_engine(ADMIN_POSTGRES_DSN, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db"),
            {"db": TEST_DB_NAME},
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
    env = {**os.environ, "POSTGRES_DSN": TEST_POSTGRES_DSN, "PYTHONPATH": str(ROOT)}
    subprocess.run(["alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True)


def reset_qdrant() -> None:
    from shared.db.qdrant import QdrantDB
    q = QdrantDB(url=TEST_QDRANT_URL)
    for col in (TEST_PRODUCT_COLLECTION, TEST_CATEGORY_COLLECTION, TEST_SEARCH_COLLECTION):
        q.delete_collection_if_exists(col)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_all(sql: str) -> list[dict[str, Any]]:
    engine = create_engine(TEST_POSTGRES_DSN)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(text(sql))]
    engine.dispose()
    return rows


def safe_mean(values: list[float]) -> float | None:
    return round(float(mean(values)), 4) if values else None


def top_k_acc(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return round(sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 4)


def mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return round(sum(1.0 / r for r in ranks if r is not None) / len(ranks), 4)


# ── Table printer ─────────────────────────────────────────────────────────────
def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("  (нет данных)")
        return
    cols   = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep    = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


# ── Table 1: основные метрики (Top-K, MRR) ───────────────────────────────────
def build_catalog_metrics() -> list[dict[str, Any]]:
    if not CATALOG_GT_PATH.exists():
        return []

    expected_by_file: dict[str, list[str]] = load_json(CATALOG_GT_PATH)

    rows = fetch_all("""
        SELECT p.product_id, p.source_file,
               c.category_name_ru, m.rank, m.similarity_score
        FROM product_category_match m
        JOIN products  p ON p.product_id  = m.product_id
        JOIN categories c ON c.category_id = m.category_id
        ORDER BY p.product_id, m.rank
    """)

    by_product: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_product[int(row["product_id"])].append(row)

    by_file: dict[str, list[list[dict]]] = defaultdict(list)
    for matches in by_product.values():
        by_file[str(matches[0]["source_file"])].append(matches)

    result_rows: list[dict] = []
    all_ranks: list[int | None] = []

    for src_file, products in sorted(by_file.items()):
        expected = set(expected_by_file.get(src_file, []))
        ranks: list[int | None] = []

        for matches in products:
            first = next(
                (int(m["rank"]) for m in matches if str(m["category_name_ru"]) in expected),
                None,
            )
            ranks.append(first)

        all_ranks.extend(ranks)
        result_rows.append({
            "source_file": src_file,
            "products":    len(ranks),
            "Top-1":       top_k_acc(ranks, 1),
            "Top-3":       top_k_acc(ranks, 3),
            "Top-5":       top_k_acc(ranks, 5),
            "MRR":         mrr(ranks),
        })

    result_rows.insert(0, {
        "source_file": "OVERALL",
        "products":    len(all_ranks),
        "Top-1":       top_k_acc(all_ranks, 1),
        "Top-3":       top_k_acc(all_ranks, 3),
        "Top-5":       top_k_acc(all_ranks, 5),
        "MRR":         mrr(all_ranks),
    })
    return result_rows


def build_cosine_catalog_metrics() -> list[dict[str, Any]]:
    if not CATALOG_GT_PATH.exists():
        return []

    expected_by_file: dict[str, list[str]] = load_json(CATALOG_GT_PATH)

    rows = fetch_all("""
        SELECT p.source_file, c.category_name_ru, m.similarity_score
        FROM product_category_match m
        JOIN products   p ON p.product_id  = m.product_id
        JOIN categories c ON c.category_id = m.category_id
    """)

    by_file: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"rel": [], "non": []})
    all_rel:  list[float] = []
    all_non:  list[float] = []

    for row in rows:
        src  = str(row["source_file"])
        cat  = str(row["category_name_ru"])
        score = float(row["similarity_score"])
        expected = set(expected_by_file.get(src, []))
        bucket = "rel" if cat in expected else "non"
        by_file[src][bucket].append(score)
        (all_rel if bucket == "rel" else all_non).append(score)

    result_rows: list[dict] = []
    for src_file, buckets in sorted(by_file.items()):
        avg_rel = safe_mean(buckets["rel"])
        max_rel = round(max(buckets["rel"]), 4) if buckets["rel"] else None  
        avg_non = safe_mean(buckets["non"])
        result_rows.append({
            "source_file":      src_file,
            "avg_relevant":     avg_rel,
            "max_relevant":     round(max(buckets["rel"]), 4) if buckets["rel"] else None,
            "avg_non_relevant": avg_non,
            "Δcos":             round(avg_rel - avg_non, 4) if avg_rel and avg_non else None,
        })

    ovr_rel = safe_mean(all_rel)
    ovr_max = round(max(all_rel), 4) if all_rel else None
    ovr_non = safe_mean(all_non)
    result_rows.insert(0, {
        "source_file":      "OVERALL",
        "avg_relevant":     ovr_rel,
        "max_relevant":     round(max(all_rel), 4) if all_rel else None,
        "avg_non_relevant": ovr_non,
        "Δcos":             round(ovr_rel - ovr_non, 4) if ovr_rel and ovr_non else None,
    })
    return result_rows


def get_client_runs() -> list[dict[str, Any]]:
    global _CLIENT_SEARCH_CACHE
    if _CLIENT_SEARCH_CACHE is not None:
        return _CLIENT_SEARCH_CACHE

    gt_items = load_json(QUERY_GT_PATH) if QUERY_GT_PATH.exists() else []

    runs = []
    for item in gt_items:
        query_text = item["query_text"]
        scored     = _search_with_scores(query_text)   # [(supplier_id, score)]
        runs.append({
            "query_text":   query_text,
            "user_id":      item.get("user_id"),
            "scored":       scored,
            "supplier_ids": [sid for sid, _ in scored],
        })

    _CLIENT_SEARCH_CACHE = runs
    return runs


def resolve_supplier_ids_by_name() -> dict[str, int]:
    """Возвращает {supplier_name: supplier_id} из тестовой БД."""
    rows = fetch_all("SELECT supplier_id, name FROM suppliers")
    return {str(r["name"]): int(r["supplier_id"]) for r in rows}


def _search_with_scores(query_text: str) -> list[tuple[int, float]]:
    """(supplier_id, best_score) — напрямую из Qdrant, для метрик."""
    from shared.core.encoder import Encoder
    from shared.db.qdrant import QdrantDB
    from shared.services.preprocessing import normalize_text

    encoder = Encoder()
    vector  = encoder.encode_single(normalize_text(query_text))
    qdrant  = QdrantDB(url=TEST_QDRANT_URL)

    hits = qdrant.client.search(
        collection_name=TEST_PRODUCT_COLLECTION,
        query_vector=vector,
        limit=50,
        with_payload=True,
    )

    best: dict[int, float] = {}
    for hit in hits:
        sid = int((hit.payload or {}).get("supplier_id", 0))
        if sid and hit.score > best.get(sid, -1.0):
            best[sid] = round(float(hit.score), 4)

    return sorted(best.items(), key=lambda x: -x[1])

def build_cosine_query_metrics() -> list[dict[str, Any]]:
    if not QUERY_GT_PATH.exists():
        return []

    gt_items   = load_json(QUERY_GT_PATH)
    name_to_id = resolve_supplier_ids_by_name()

    gt_by_query: dict[str, set[int]] = {
        item["query_text"]: {name_to_id[n] for n in item.get("expected_supplier_names", []) if n in name_to_id}
        for item in gt_items
    }

    rows: list[dict] = []
    for run in get_client_runs():
        query    = run["query_text"]
        scored   = run["scored"]          
        expected = gt_by_query.get(query, set())

        rank            = None
        max_relevant    = None
        max_non_relevant = None

        for i, (sid, score) in enumerate(scored):
            if sid in expected:
                if rank is None:
                    rank = i + 1
                max_relevant = max(max_relevant or -1, score)
            else:
                max_non_relevant = max(max_non_relevant or -1, score)

        rows.append({
            "query":            query[:50],
            "expected_rank":    rank if rank is not None else "-",
            "reciprocal_rank":  round(1.0 / rank, 4) if rank else 0.0,
            "max_relevant":     max_relevant if max_relevant is not None else "-",
            "max_non_relevant": max_non_relevant if max_non_relevant is not None else "-",
            "found_in_top5":    any(sid in expected for sid, _ in scored[:5]),
        })

    return rows


def print_final_report() -> None:
    print_section("1 / 3  |  CATEGORY RANKING  —  Top-K Accuracy & MRR")
    print_table(build_catalog_metrics())

    print_section("2 / 3  |  COSINE SIMILARITY  —  по категориям")
    print_table(build_cosine_catalog_metrics())

    print_section("3 / 3  |  COSINE SIMILARITY  —  по клиентским запросам")
    print_table(build_cosine_query_metrics())


def main() -> int:
    configure_env()

    print(f"[runner] db:     {TEST_POSTGRES_DSN}")
    print(f"[runner] qdrant: {TEST_QDRANT_URL}")

    ensure_test_database()
    reset_test_database()
    reset_qdrant()
    run_migrations()

    exit_code = pytest.main([
        "-q", "-s", "--disable-warnings",
        "test/integration/test_catalog_pipeline.py",
        "test/integration/test_search_pipeline.py",
        "test/integration/test_client_service_api.py",
    ])

    print_final_report()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())