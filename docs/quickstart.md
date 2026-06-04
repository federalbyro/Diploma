# Быстрый старт

## Требования

- Docker ≥ 24
- Docker Compose ≥ 2.20

## Запуск

```bash
git clone https://github.com/federalbyro/Diploma.git
cd Diploma

cp .env.example .env      # при необходимости отредактируй

docker-compose up --build
```

При первом запуске:

1. `model_downloader` скачивает LaBSE (~1.8 ГБ) в Docker volume `model_cache`
2. `migrations` применяет схему БД
3. Сервисы стартуют после готовности модели

Повторные запуски — быстрые, модель уже в volume.

## Проверка

| Сервис          | URL                               |
|-----------------|-----------------------------------|
| Catalog API     | http://localhost:8000/docs        |
| Client API      | http://localhost:8001/docs        |
| Qdrant UI       | http://localhost:6333/dashboard   |

## Тесты

```bash
./run_tests.sh
```

## Остановка

```bash
docker-compose down          # сохраняет volumes (БД, модель)
docker-compose down -v       # полный сброс
```