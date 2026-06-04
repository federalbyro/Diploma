#!/usr/bin/env bash
set -e

echo "[tests] running inside catalog_service..."
docker-compose exec catalog_service python test/run_pytest.py