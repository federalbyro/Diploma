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

        assert isinstance(results, list), \
            f"Результат поиска должен быть списком, получили: {type(results)}"
        assert len(results) > 0, \
            f"По запросу {query!r} поиск ничего не вернул"

        # results — list[int] (supplier_id), порядок = релевантность
        for supplier_id in results:
            assert isinstance(supplier_id, int), \
                f"Каждый элемент должен быть int (supplier_id), получили: {type(supplier_id)}"