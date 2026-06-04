"""
Нагрузочный тест POST /search — два режима:
  1. Стабильный тест: фиксированный RPS, заданная длительность
  2. Поиск потолка: ступенчатый рост RPS до деградации

Запуск:
  python bench_client.py                        # стабильный 10 rps + поиск потолка
  python bench_client.py --rps 20 --no-rampup  # только стабильный
  python bench_client.py --only-rampup          # только потолок
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

import httpx

QUERIES = [
    "детские игрушки",
    "авиационные комплектующие",
    "сантехнические товары",
    "промышленное оборудование",
    "текстиль и ткани",
]

P95_THRESHOLD_MS   = 3_000
ERROR_THRESHOLD_PCT = 5.0


@dataclass
class BenchResult:
    target_rps:  int
    actual_rps:  float
    total:       int
    errors:      int
    error_rate:  float   # %
    p50:         float
    p95:         float
    p99:         float
    p_max:       float
    mean:        float

    @property
    def degraded(self) -> bool:
        return self.p95 > P95_THRESHOLD_MS or self.error_rate > ERROR_THRESHOLD_PCT

    def status(self) -> str:
        if self.error_rate > ERROR_THRESHOLD_PCT:
            return "🔴 ERRORS"
        if self.p95 > P95_THRESHOLD_MS:
            return "🟡 SLOW"
        return "🟢 OK"


async def single_request(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
) -> tuple[float, int]:
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
) -> BenchResult:
    latencies: list[float] = []
    errors = 0
    total  = 0

    interval = 1.0 / target_rps
    deadline = time.perf_counter() + duration_sec
    sem      = asyncio.Semaphore(concurrency)

    async def bounded(client: httpx.AsyncClient, query: str) -> None:
        nonlocal errors, total
        async with sem:
            lat, status = await single_request(client, base_url, query)
            latencies.append(lat)
            total += 1
            if status != 200:
                errors += 1

    async with httpx.AsyncClient() as client:
        tasks: list[asyncio.Task] = []
        q_idx   = 0
        t_start = time.perf_counter()

        while time.perf_counter() < deadline:
            query = QUERIES[q_idx % len(QUERIES)]
            tasks.append(asyncio.create_task(bounded(client, query)))
            q_idx += 1
            await asyncio.sleep(interval)

        await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t_start

    if not latencies:
        return BenchResult(target_rps, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    latencies.sort()

    def pct(p: float) -> float:
        idx = int(len(latencies) * p / 100)
        return round(latencies[min(idx, len(latencies) - 1)], 1)

    return BenchResult(
        target_rps  = target_rps,
        actual_rps  = round(total / elapsed, 1),
        total       = total,
        errors      = errors,
        error_rate  = round(errors / total * 100, 1) if total else 0.0,
        p50         = pct(50),
        p95         = pct(95),
        p99         = pct(99),
        p_max       = round(max(latencies), 1),
        mean        = round(statistics.mean(latencies), 1),
    )


def print_result(r: BenchResult, label: str = "") -> None:
    title = f"  {label}  " if label else ""
    print(f"\n{'─' * 52}")
    if title:
        print(f"{title.center(52)}")
        print(f"{'─' * 52}")
    print(f"  Status        : {r.status()}")
    print(f"  Target RPS    : {r.target_rps}")
    print(f"  Actual RPS    : {r.actual_rps}")
    print(f"  Requests      : {r.total}")
    print(f"  Errors        : {r.errors}  ({r.error_rate}%)")
    print(f"  Latency  p50  : {r.p50} ms")
    print(f"  Latency  p95  : {r.p95} ms")
    print(f"  Latency  p99  : {r.p99} ms")
    print(f"  Latency  max  : {r.p_max} ms")
    print(f"  Latency  mean : {r.mean} ms")
    print(f"{'─' * 52}")


async def find_max_rps(
    base_url:    str,
    start_rps:   int,
    step_sec:    int,
    concurrency: int,
) -> None:
    """
    Ступенчато удваиваем RPS: start → start*2 → start*4 → ...
    Останавливаемся при первой деградации и сообщаем потолок.
    """
    print(f"\n{'═' * 52}")
    print("  RAMP-UP: поиск максимального RPS".center(52))
    print(f"  Критерии деградации:".center(52))
    print(f"  p95 > {P95_THRESHOLD_MS} ms  ИЛИ  error_rate > {ERROR_THRESHOLD_PCT}%".center(52))
    print(f"  Шаг: {step_sec}s на каждый уровень".center(52))
    print(f"{'═' * 52}")

    results: list[BenchResult] = []
    rps = start_rps

    # Warm-up
    async with httpx.AsyncClient() as client:
        await single_request(client, base_url, QUERIES[0])

    while True:
        print(f"\n[ramp] пробуем {rps} rps...", flush=True)
        r = await run_bench(base_url, rps, step_sec, concurrency)
        results.append(r)
        print_result(r, label=f"RPS = {rps}")

        if r.degraded:
            break

        rps *= 2
        if rps > 2000:
            print("[ramp] достигнут лимит 2000 rps, останавливаемся")
            break

    good = [r for r in results if not r.degraded]
    print(f"\n{'═' * 52}")
    if good:
        best = good[-1]
        print(f"  ✅ Максимальный устойчивый RPS : {best.target_rps}".center(52))
        print(f"     p95 = {best.p95} ms | errors = {best.error_rate}%".center(52))
    else:
        print("  ⚠️  Система деградировала с первого шага".center(52))
    if results[-1].degraded:
        d = results[-1]
        print(f"  🔴 Деградация при RPS = {d.target_rps}".center(52))
        print(f"     p95 = {d.p95} ms | errors = {d.error_rate}%".center(52))
    print(f"{'═' * 52}\n")


async def main_async(args: argparse.Namespace) -> None:
    base_url    = args.url
    concurrency = args.concurrency
    print("[bench] warm-up...", flush=True)
    async with httpx.AsyncClient() as client:
        await single_request(client, base_url, QUERIES[0])

    if not args.only_rampup:
        print(f"\n{'═' * 52}")
        print(f"  ФАЗА 1: стабильный тест {args.rps} rps × {args.duration}s".center(52))
        print(f"{'═' * 52}")
        r = await run_bench(base_url, args.rps, args.duration, concurrency)
        print_result(r, label=f"Стабильный тест — {args.rps} rps")

    # ── Фаза 2: поиск потолка ─────────────────────────────────────────────
    if not args.no_rampup:
        await find_max_rps(
            base_url    = base_url,
            start_rps   = args.rps,
            step_sec    = args.step,
            concurrency = concurrency,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test POST /search")
    parser.add_argument("--url",         default="http://localhost:8001")
    parser.add_argument("--rps",         type=int, default=10,
                        help="RPS для стабильного теста и стартовый для ramp-up")
    parser.add_argument("--duration",    type=int, default=30,
                        help="Длительность стабильного теста (сек)")
    parser.add_argument("--step",        type=int, default=15,
                        help="Длительность каждого шага ramp-up (сек)")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Макс. параллельных запросов")
    parser.add_argument("--no-rampup",   action="store_true",
                        help="Только стабильный тест, без поиска потолка")
    parser.add_argument("--only-rampup", action="store_true",
                        help="Только поиск потолка, без стабильного теста")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()