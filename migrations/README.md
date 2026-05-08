

## Назначение

Alembic используется для воспроизводимого создания и обновления структуры базы данных. 
Это позволяет разворачивать проект в новой среде без ручного создания таблиц.

---
## Важно - миграции накатываются автоматически, в ручную их не надо закидывать!

## Основные команды на будущее

Применить все миграции:

```bash
alembic upgrade head
```

Посмотреть текущую миграцию:

```bash
alembic current
```

Создать новую миграцию:

```bash
alembic revision -m "migration_name"
```

Создать миграцию с автогенерацией:

```bash
alembic revision --autogenerate -m "migration_name"
```

Откатить последнюю миграцию:

```bash
alembic downgrade -1
```

---

## Что создаётся миграциями

```text
suppliers
products
categories
product_category_match
supplier_category_mapping
jobs
```

---

