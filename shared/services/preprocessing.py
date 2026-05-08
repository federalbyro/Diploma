from __future__ import annotations

import re
import html
import unicodedata


MULTISPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)

# Оставляем буквы/цифры/пробелы/базовую тех. пунктуацию
DISALLOWED_SYMBOLS_RE = re.compile(r"[^\w\s\-\.,;:/%\(\)\[\]\+#&×x]", re.UNICODE)

# Пробелы вокруг пунктуации
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:%\)\]])")
SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[])\s+")
DOUBLE_PUNCT_RE = re.compile(r"([,.;:])\1+")

# Числа + единицы измерения
UNIT_GLUE_RE = re.compile(
    r"(\d)\s+(mm|cm|m|km|kg|g|mg|l|ml|w|kw|v|a|mah|hz|khz|mhz|ghz|°c|℃|%)\b",
    re.IGNORECASE,
)

DIMENSION_RE_2D = re.compile(r"(\d)\s*[x×]\s*(\d)", re.IGNORECASE)
DIMENSION_RE_3D = re.compile(r"(\d)\s*[x×]\s*(\d)\s*[x×]\s*(\d)", re.IGNORECASE)


def _basic_normalize(text: str) -> str:
    """
    Базовая нормализация:
    - HTML unescape
    - Unicode normalization
    - удаление URL / email
    - схлопывание пробелов
    """
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def _clean_symbols(text: str) -> str:
    """
    Удаляет явный мусор, сохраняя тех. пунктуацию.
    """
    text = DISALLOWED_SYMBOLS_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def _normalize_punctuation(text: str) -> str:
    """
    Подчищает пробелы и пунктуацию, не ломая технические описания.
    """
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = DOUBLE_PUNCT_RE.sub(r"\1", text)

    text = UNIT_GLUE_RE.sub(r"\1\2", text)
    text = DIMENSION_RE_3D.sub(r"\1x\2x\3", text)
    text = DIMENSION_RE_2D.sub(r"\1x\2", text)

    text = MULTISPACE_RE.sub(" ", text)
    return text.strip(" ,;.")


def normalize_text(text: str) -> str:
    """
    Универсальная нормализация текста для LaBSE.
    """
    text = _basic_normalize(text)
    text = _clean_symbols(text)
    text = _normalize_punctuation(text)
    return text


def build_product_text(*parts: str) -> str:
    """
    Собирает итоговое текстовое описание товара из одной или нескольких частей.
    Используется ingestion-слоем для любых форматов входа.
    """
    cleaned_parts: list[str] = []

    for part in parts:
        part_norm = normalize_text(part)
        if part_norm:
            cleaned_parts.append(part_norm)

    # убираем дубли с сохранением порядка
    seen = set()
    unique_parts: list[str] = []
    for part in cleaned_parts:
        if part not in seen:
            seen.add(part)
            unique_parts.append(part)

    return " ; ".join(unique_parts).strip()


def build_category_text(
    category_name_ru: str,
    category_description_ru: str | None = None,
) -> str:
    """
    Формирует текст категории для эмбеддинга.
    Категория должна быть богаче по смыслу, чем просто одно название.
    """
    name = normalize_text(category_name_ru)
    description = normalize_text(category_description_ru or "")

    if description and description != name:
        return f"{name}. {description}".strip()

    return name