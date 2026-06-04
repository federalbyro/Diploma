# Diploma — Multilingual Supplier Catalog

Система автоматической обработки товарных каталогов китайских поставщиков
и семантического поиска релевантных поставщиков по текстовым запросам.

## Что делает система

- Принимает каталоги поставщиков в форматах **CSV / XLSX / TXT**
- Нормализует и индексирует товарные описания (в том числе на китайском)
- Сопоставляет товары с категориями через **косинусное сходство** векторов LaBSE
- Отвечает на клиентские запросы списком релевантных `supplier_id`

## Стек

| Компонент       | Решение                          |
|-----------------|----------------------------------|
| API             | FastAPI + Uvicorn                |
| Очередь         | Celery + Redis                   |
| Реляционная БД  | PostgreSQL + SQLAlchemy + Alembic|
| Векторная БД    | Qdrant                           |
| ML-модель       | LaBSE (sentence-transformers)    |
| Контейнеризация | Docker + Docker Compose          |

## Архитектура

```
catalog_service  ──POST /catalog/upload──►  Celery Worker
                                                │
                                    ┌───────────▼───────────┐
                                    │  1. parse file        │
                                    │  2. normalize text    │
                                    │  3. LaBSE encode      │
                                    │  4. Qdrant index      │
                                    │  5. top-k match       │
                                    │  6. aggregate         │
                                    └───────────────────────┘

client_service   ──POST /search──►  normalize ► LaBSE ► Qdrant ► [supplier_ids]
```