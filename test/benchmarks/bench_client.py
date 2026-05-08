from __future__ import annotations

from statistics import mean

import httpx

from test.benchmarks.data import load_client_queries
from test.benchmarks.timer import print_dict, print_stage, timed


def run_client_api_benchmark(
    base_url: str = "http://127.0.0.1:8001",
    queries_file: str = "client_queries_small.json",
) -> dict:
    queries = load_client_queries(queries_file)

    per_query: list[dict] = []

    with httpx.Client(timeout=60.0) as client:
        with timed("client_api_all", rows=len(queries)) as total_timer:
            for item in queries:
                payload = {
                    "query": item["query_text"],
                    "user_id": item.get("user_id"),
                }

                with timed("client_api_query") as t:
                    response = client.post(f"{base_url}/search", json=payload)
                    response.raise_for_status()
                    data = response.json()

                results = data.get("results", [])
                per_query.append(
                    {
                        "query_text": item["query_text"],
                        "seconds": t.seconds,
                        "status_code": response.status_code,
                        "result_count": len(results),
                    }
                )
                print_stage("client_api_query", t.seconds)

    avg_sec = mean(x["seconds"] for x in per_query) if per_query else 0.0

    result = {
        "queries_file": queries_file,
        "queries_count": len(queries),
        "total_sec": total_timer.seconds,
        "avg_query_sec": round(avg_sec, 4),
        "queries_per_sec": round(len(queries) / total_timer.seconds, 2) if total_timer.seconds > 0 else 0.0,
        "per_query": per_query,
    }

    print_dict("\n[bench] Client API benchmark summary", {
        k: v for k, v in result.items() if k != "per_query"
    })
    return result


if __name__ == "__main__":
    run_client_api_benchmark()