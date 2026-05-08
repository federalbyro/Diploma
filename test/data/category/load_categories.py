from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CATEGORY_ROOT = Path("test/data/category")


def load_categories(filename: str) -> list[dict[str, Any]]:
    path = CATEGORY_ROOT / filename
    items: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    result: list[dict[str, Any]] = []

    for item in items:
        name = str(item["category_name_ru"]).strip()

        descriptions = [
            str(value).strip()
            for value in item.get("category_descriptions", [])
            if str(value).strip()
        ]

        if item.get("category_description_ru"):
            summary = str(item["category_description_ru"]).strip()
        elif item.get("category_summary_ru"):
            summary = str(item["category_summary_ru"]).strip()
        elif descriptions:
            summary = descriptions[0]
        else:
            summary = name

        if not descriptions:
            old_parts: list[str] = []

            base_description = item.get("base_description")
            if base_description:
                old_parts.append(str(base_description).strip())

            for value in item.get("synonyms") or []:
                value = str(value).strip()
                if value:
                    old_parts.append(value)

            for value in item.get("examples") or []:
                value = str(value).strip()
                if value:
                    old_parts.append(value)

            if old_parts:
                descriptions = old_parts

        result.append(
            {
                "category_name_ru": name,
                "category_description_ru": summary,
                "category_descriptions": descriptions,
            }
        )

    return result