from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.models import Category, Product, Supplier
from shared.services.preprocessing import build_category_text, build_product_text


def _read_text_lines(path: Path) -> list[str]:
    """
    Читает CSV/TXT как обычный текстовый файл:
    одна непустая строка = один товар.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if set(line) <= {"-", "—", "_", "=", "*", " "}:
            continue
        lines.append(line)

    return lines


def _read_excel_lines(path: Path) -> list[str]:
    """
    Читает XLSX/XLS:
    одна строка Excel = один товар.

    Если в строке несколько непустых ячеек, склеиваем их.
    Если у тебя XLSX всегда строго в одну колонку, это тоже работает.
    """
    df = pd.read_excel(path, header=None).fillna("")
    lines: list[str] = []
    for _, row in df.iterrows():
        parts: list[str] = []
        for value in row.tolist():
            value_str = str(value).strip()
            if not value_str:
                continue
            if value_str.lower() in {"nan", "none", "null"}:
                continue
            parts.append(value_str)
        line = " ; ".join(parts).strip()
        if line:
            lines.append(line)

    return lines


def _read_catalog_lines(file_path: str) -> list[str]:
    """
    Универсальное чтение входного каталога:
    CSV/TXT/XLSX/XLS -> list[str],
    где каждый элемент списка — отдельное описание товара.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return _read_text_lines(path)

    if suffix in {".xlsx", ".xls"}:
        return _read_excel_lines(path)

    raise ValueError(f"Unsupported file extension: {suffix}")


def get_or_create_supplier(
    session: Session,
    supplier_name: str,
    meta_json: str | None = None,
) -> Supplier:
    supplier = session.execute(
        select(Supplier).where(Supplier.name == supplier_name)
    ).scalar_one_or_none()

    if supplier is not None:
        if meta_json and not supplier.meta_json:
            supplier.meta_json = meta_json
        session.flush()
        return supplier

    supplier = Supplier(
        name=supplier_name,
        meta_json=meta_json,
    )
    session.add(supplier)
    session.flush()
    return supplier


def ingest_catalog(
    session: Session,
    file_path: str,
    supplier_name: str,
    meta_json: str | None = None,
) -> list[Product]:
    """
    Загружает товары из CSV/XLSX/XLS/TXT в PostgreSQL.

    Формат входа:
    одна строка = один товар.
    """
    lines = _read_catalog_lines(file_path)

    supplier = get_or_create_supplier(
        session=session,
        supplier_name=supplier_name,
        meta_json=meta_json,
    )

    created: list[Product] = []

    for original_text in lines:
        original_text = original_text.strip()

        if not original_text:
            continue

        normalized_text = build_product_text(original_text)

        if not normalized_text:
            continue

        product = Product(
            supplier_id=supplier.supplier_id,
            original_description=original_text,
            normalized_description=normalized_text,
            source_file=Path(file_path).name,
        )

        session.add(product)
        created.append(product)

    session.flush()

    supplier.product_count = (supplier.product_count or 0) + len(created)
    session.flush()

    return created


def upsert_categories(
    session: Session,
    categories: list[dict[str, Any]],
) -> list[Category]:
    """
    Добавляет или обновляет категории.
    Ожидает:
    - category_name_ru
    - category_description_ru
    """
    created_or_updated: list[Category] = []

    for row in categories:
        category_name = str(row["category_name_ru"]).strip()

        if not category_name:
            continue

        category_description = str(
            row.get("category_description_ru") or category_name
        ).strip()

        normalized_text = build_category_text(
            category_name,
            category_description,
        )

        existing = session.execute(
            select(Category).where(Category.category_name_ru == category_name)
        ).scalar_one_or_none()

        if existing is not None:
            existing.category_description_ru = category_description
            existing.normalized_text_ru = normalized_text
            created_or_updated.append(existing)
            continue

        category = Category(
            category_name_ru=category_name,
            category_description_ru=category_description,
            normalized_text_ru=normalized_text,
        )

        session.add(category)
        created_or_updated.append(category)

    session.flush()
    return created_or_updated