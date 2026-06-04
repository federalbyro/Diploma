

## Назначение

```text
1. единая конфигурация;
2. единая модель LaBSE;
3. единые ORM-модели;
4. единая нормализация текста;
5. единый механизм генерации эмбеддингов;
6. единый механизм поиска ближайших соседей.
```

---

## Структура

```text
shared/
├── core/
│   ├── config.py        # переменные окружения и настройки приложения
│   └── encoder.py       # загрузка и использование LaBSE
│
├── db/
│   ├── models.py        # ORM-модели PostgreSQL
│   ├── postgres.py      # engine, session, подключение к PostgreSQL
│   └── qdrant.py        # клиент Qdrant и работа с коллекциями
│
└── services/
    ├── preprocessing.py # очистка и нормализация текста
    ├── embedding.py     # генерация и сохранение эмбеддингов
    ├── matcher.py       # поиск категорий и поставщиков
    └── batching.py      # батчевая обработка
```

---

## Основные сущности

### PostgreSQL

```text
Supplier
Product
Category
ProductCategoryMatch
SupplierCategoryMapping
Job
```

### Qdrant

```text
PRODUCT_COLLECTION
CATEGORY_COLLECTION
SEARCH_COLLECTION
```

---

## Нормализация текста

Модуль нормализации приводит входные строки к единому виду, но не выполняет агрессивную лингвистическую обработку.

В проекте не используются:

```text
стемминг
лемматизация
промежуточный машинный перевод
```

Это сделано для сохранения технических параметров, артикулов, стандартов, размеров и смешанных китайско-латинских фрагментов.


# Использование дообученной модели
# После обучения сохраняем:
model.save('/models/local')

# Либо через MODEL_LOCAL_PATH в .env:
MODEL_LOCAL_PATH=/path/to/my_model

# Приоритет при загрузке:
# 1. MODEL_LOCAL_PATH (если задан и валиден)
# 2. /models/local/ (если существует)
# 3. HuggingFace (MODEL_NAME)