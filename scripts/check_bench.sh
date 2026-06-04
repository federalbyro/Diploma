#!/usr/bin/env bash
set -e

echo "[2/2] Load test  POST /search  (30s, 10 rps)"
python test/benchmarks/bench_client.py \
    --url http://localhost:8001 \
    --rps 10 \
    --duration 30 \
    --concurrency 20