
## Основные функции

```text
1. Приём файла каталога.
2. Создание задачи обработки.
3. Чтение CSV/TXT/XLS/XLSX.
4. Нормализация товарных описаний.
5. Сохранение товаров в PostgreSQL.
6. Построение эмбеддингов товаров.
7. Сохранение эмбеддингов в Qdrant.
8. Сопоставление товаров с категориями.
9. Агрегация результатов на уровне поставщика.
```

---

## Структура

```text
catalog/
├── api/
│   ├── catalog.py      # загрузка каталогов и работа с задачами
│   └── categories.py   # обновление категорий
│
├── services/
│   └── ingestion.py    # чтение файлов и загрузка товаров в PostgreSQL
│
├── tasks/
│   └── celery_app.py   # Celery-приложение и фоновые задачи
│
└── main.py             # FastAPI-приложение catalog_service
```

---

## Основные API-ручки

```text
POST /catalog/upload
```

Загружает каталог поставщика и запускает фоновую обработку.

```text
GET /catalog/jobs/{job_id}
```

Возвращает статус фоновой задачи.

```text
PATCH /categories/{category_id}
```

Обновляет категорию и переиндексирует её эмбеддинг в Qdrant.

---

## Поддерживаемые форматы каталогов

```text
.csv
.txt
.xlsx
.xls
```

Внутренняя логика обработки использует принцип:

```text
одна строка = одна товарная позиция
```

---

## Связанные модули

```text
shared.services.preprocessing  # нормализация текста
shared.services.embedding      # генерация эмбеддингов
shared.services.matcher        # семантическое сопоставление
shared.db.models               # ORM-модели
shared.db.postgres             # подключение к PostgreSQL
shared.db.qdrant               # подключение к Qdrant
```