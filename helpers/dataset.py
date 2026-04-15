from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def list_input_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.glob("*.csv") if p.is_file())


def resolve_label_file(input_file: Path, label_dir: Path) -> Path | None:
    candidate = label_dir / input_file.name
    return candidate if candidate.exists() else None


def make_run_output_dir(output_dir: Path, timestamp: str | None = None) -> Path:
    run_id = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_output_file(run_dir: Path, input_file: Path) -> Path:
    return run_dir / f"{input_file.stem}_russian.csv"


def _read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise last_error


def load_input_dataframe(input_file: Path, text_column: str = "source") -> pd.DataFrame:
    # Читаем максимально безопасно для одно-колоночных CSV без заголовка.
    df = _read_csv_with_fallback(input_file, header=None, names=[text_column])
    if not df.empty:
        first = str(df.iloc[0, 0]).strip().lower()
        if first == text_column.lower():
            df = df.iloc[1:].reset_index(drop=True)

    df[text_column] = df[text_column].fillna("").astype(str).str.strip()
    return df


def load_reference_texts(reference_file: Path, ref_column: str = "reference_ru") -> list[str]:
    # Для label поддерживаем и вариант с заголовком, и вариант с одной колонкой без заголовка.
    try:
        df = _read_csv_with_fallback(reference_file)
        if ref_column in df.columns:
            series = df[ref_column]
        elif len(df.columns) == 1:
            series = df.iloc[:, 0]
        else:
            raise ValueError(
                f"Не найден столбец '{ref_column}' в {reference_file.name}. "
                f"Доступные столбцы: {list(df.columns)}"
            )
    except Exception:
        df = _read_csv_with_fallback(reference_file, header=None)
        if df.empty:
            return []
        series = df.iloc[:, 0]

    return series.fillna("").astype(str).str.strip().tolist()
