from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


TEST_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = TEST_ROOT / "data"
CATEGORY_ROOT = DATA_ROOT / "category"
CLIENT_ROOT = DATA_ROOT / "client"
INPUT_ROOT = DATA_ROOT / "input"


def load_json(path: str | Path) -> Any:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_categories(filename: str) -> list[dict[str, Any]]:
    """
    Загружает категории из JSON.

    В PostgreSQL уходит одна категория:
    - category_name_ru
    - category_description_ru

    В Qdrant дополнительно уйдут несколько прототипов:
    - category_descriptions
    """
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

        summary = str(
            item.get("category_summary_ru")
            or item.get("category_description_ru")
            or (descriptions[0] if descriptions else name)
        ).strip()

        result.append(
            {
                "category_name_ru": name,
                "category_description_ru": summary,
                "category_descriptions": descriptions,
            }
        )

    return result


def load_client_queries(file_name: str) -> list[dict]:
    return load_json(CLIENT_ROOT / file_name)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return pd.DataFrame({"raw_text": [x.strip() for x in lines if x.strip()]})

    raise ValueError(f"Unsupported file type: {suffix}")


def list_input_files() -> list[Path]:
    files: list[Path] = []
    for ext in ("*.csv", "*.xlsx", "*.xls", "*.txt"):
        files.extend(INPUT_ROOT.glob(ext))
    return sorted(files)