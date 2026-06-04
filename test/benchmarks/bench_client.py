"""
Нагрузочный тест POST /search.
Запуск: python test/benchmarks/bench_client.py --url http://localhost:8001 --rps 20 --duration 30
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any

import httpx

QUERIES = [
    "детские игрушки",
    "авиационные комплектующие",
    "сантехнические товары",
    "промышленное оборудование",
    "текстиль и ткани",
]


async def single_request(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
) -> tuple[float, int]:
    """Возвращает (latency_ms, status_code)."""
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/search",
            json={"query": query, "limit": 10},
            timeout=10.0,
        )
        return (time.perf_counter() - t0) * 1000, resp.status_code
    except Exception:
        return (time.perf_counter() - t0) * 1000, 0


async def run_bench(
    base_url: str,
    target_rps: int,
    duration_sec: int,
    concurrency: int,
) -> None:
    latencies: list[float] = []
    errors = 0
    total  = 0

    interval = 1.0 / target_rps  # секунд между запросами
    deadline = time.perf_counter() + duration_sec

    sem = asyncio.Semaphore(concurrency)

    async def bounded(query: str) -> None:
        nonlocal errors, total
        async with sem:
            lat, status = await single_request(client, base_url, query)
            latencies.append(lat)
            total += 1
            if status != 200:
                errors += 1

    print(f"[bench] target={target_rps} rps | duration={duration_sec}s "
          f"| concurrency={concurrency} | url={base_url}")
    print("[bench] warming up...")

    async with httpx.AsyncClient() as client:
        # Warm-up (1 запрос)
        await single_request(client, base_url, QUERIES[0])
        print("[bench] starting...")

        tasks: list[asyncio.Task] = []
        q_idx = 0
        t_start = time.perf_counter()

        while time.perf_counter() < deadline:
            query = QUERIES[q_idx % len(QUERIES)]
            tasks.append(asyncio.create_task(bounded(query)))
            q_idx += 1
            await asyncio.sleep(interval)

        await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t_start

    if not latencies:
        print("[bench] no results")
        return

    latencies.sort()

    def pct(p: float) -> float:
        idx = int(len(latencies) * p / 100)
        return round(latencies[min(idx, len(latencies) - 1)], 1)

    print(f"\n{'─' * 50}")
    print(f"  Requests      : {total}")
    print(f"  Errors        : {errors}  ({errors/total*100:.1f}%)")
    print(f"  Elapsed       : {elapsed:.1f}s")
    print(f"  Actual RPS    : {total / elapsed:.1f}")
    print(f"  Latency  p50  : {pct(50)} ms")
    print(f"  Latency  p95  : {pct(95)} ms")
    print(f"  Latency  p99  : {pct(99)} ms")
    print(f"  Latency  max  : {round(max(latencies), 1)} ms")
    print(f"  Latency  mean : {round(statistics.mean(latencies), 1)} ms")
    print(f"{'─' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test POST /search")
    parser.add_argument("--url",         default="http://localhost:8001")
    parser.add_argument("--rps",         type=int, default=10)
    parser.add_argument("--duration",    type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    asyncio.run(run_bench(args.url, args.rps, args.duration, args.concurrency))


if __name__ == "__main__":
    main()