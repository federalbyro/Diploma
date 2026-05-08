from __future__ import annotations

import json
from pathlib import Path

from client.services.query import handle_query


QUERIES_FILE = Path("test/data/client/client_queries_extended.json")


def test_search_pipeline() -> None:
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))

    assert queries, "Файл пользовательских запросов пуст"

    for item in queries:
        query = item["query_text"]

        results = handle_query(query_text=query, user_id=item.get("user_id"))

        assert isinstance(results, list), "Результат поиска должен быть списком"
        assert len(results) > 0, f"По запросу {query!r} поиск ничего не вернул"

        previous_score = None

        for row in results:
            assert "supplier_id" in row
            assert "supplier_name" in row
            assert "score" in row
            assert "avg_similarity" in row
            assert "best_similarity" in row
            assert "matched_product_ids" in row

            assert row["supplier_id"] is not None
            assert row["supplier_name"] is not None
            assert isinstance(row["matched_product_ids"], list)
            assert len(row["matched_product_ids"]) > 0

            assert 0 <= float(row["score"]) <= 1
            assert 0 <= float(row["avg_similarity"]) <= 1
            assert 0 <= float(row["best_similarity"]) <= 1

            if previous_score is not None:
                assert float(row["score"]) <= float(previous_score)

            previous_score = row["score"]