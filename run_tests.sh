#!/usr/bin/env bash
set -e

echo "[tests] running inside catalog_service..."
docker-compose exec catalog_service python test/run_pytest.py


echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/2] Load test  POST /search  (30s, 10 rps)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python test/benchmarks/bench_client.py \
    --url http://localhost:8001 \
    --rps 10 \
    --duration 30 \
    --concurrency 20