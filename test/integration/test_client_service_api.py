from __future__ import annotations

from fastapi.testclient import TestClient


def test_client_service_search_api() -> None:
    from client.main import app

    payload = {
        "query": "детские игрушки",
        "user_id": 999,
    }

    with TestClient(app) as client:
        response = client.post("/search", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == payload["query"]
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0

    first = data["results"][0]

    assert "supplier_id" in first
    assert "supplier_name" in first
    assert "score" in first
    assert "matched_product_ids" in first