from __future__ import annotations

from statistics import mean

from client.services.query import handle_query
from test.benchmarks.data import load_client_queries
from test.benchmarks.timer import print_dict, print_stage, timed


def run_search_benchmark(
    queries_file: str = "client_queries_small.json",
) -> dict:
    queries = load_client_queries(queries_file)

    per_query: list[dict] = []

    with timed("all_queries", rows=len(queries)) as total_timer:
        for item in queries:
            query_text = item["query_text"]

            with timed(f"query::{query_text}") as t:
                results = handle_query(query_text=query_text, user_id=item.get("user_id"))

            per_query.append(
                {
                    "query_text": query_text,
                    "seconds": t.seconds,
                    "result_count": len(results),
                }
            )
            print_stage("search_query", t.seconds)

    avg_sec = mean(x["seconds"] for x in per_query) if per_query else 0.0
    avg_result_count = mean(x["result_count"] for x in per_query) if per_query else 0.0

    result = {
        "queries_file": queries_file,
        "queries_count": len(queries),
        "total_sec": total_timer.seconds,
        "avg_query_sec": round(avg_sec, 4),
        "queries_per_sec": round(len(queries) / total_timer.seconds, 2) if total_timer.seconds > 0 else 0.0,
        "avg_result_count": round(avg_result_count, 2),
        "per_query": per_query,
    }

    print_dict("\n[bench] Search benchmark summary", {
        k: v for k, v in result.items() if k != "per_query"
    })
    return result


if __name__ == "__main__":
    run_search_benchmark()