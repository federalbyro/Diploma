import re
from langdetect import detect, LangDetectException

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
NUMBER_RE  = re.compile(r"\d+\.?\d*")


def lang_is_correct(text: str, expected_lang: str) -> bool:
    """
    Проверяет что langdetect определяет нужный язык.
    expected_lang: 'ru', 'en', etc.
    """
    try:
        return detect(text) == expected_lang
    except LangDetectException:
        return False


def length_ratio(source: str, hypothesis: str) -> float:
    """
    Отношение длины перевода к длине исходника (в символах).
    Норма: 0.5–2.0. Сильное отклонение = потеря содержания или мусор.
    """
    src_len = max(len(source), 1)
    return round(len(hypothesis) / src_len, 3)


def repetition_score(text: str) -> float:
    """
    Доля уникальных слов в тексте.
    1.0 = нет повторов. < 0.6 = вероятно мусорный перевод.
    """
    words = text.lower().split()
    if len(words) < 3:
        return 1.0
    return round(len(set(words)) / len(words), 3)


def numbers_preserved(source: str, hypothesis: str) -> float:
    """
    Доля числовых значений из source, сохранившихся в hypothesis.
    1.0 = все числа на месте. Критично для технических описаний.
    """
    src_nums = set(NUMBER_RE.findall(source))
    hyp_nums = set(NUMBER_RE.findall(hypothesis))
    if not src_nums:
        return 1.0
    return round(len(src_nums & hyp_nums) / len(src_nums), 3)


def no_chinese_leak(text: str) -> bool:
    """Проверяет что в переводе нет иероглифов."""
    return len(CHINESE_RE.findall(text)) == 0



def score_row(source: str, english: str, russian: str) -> dict:
    """
    Считает все эвристики для одной строки.
    Возвращает словарь — удобно собирать в DataFrame.
    """
    return {
        "en_lang_ok":         lang_is_correct(english, "en"),
        "ru_lang_ok":         lang_is_correct(russian, "ru"),
        "en_length_ratio":    length_ratio(source, english),
        "ru_length_ratio":    length_ratio(source, russian),
        "en_repetition":      repetition_score(english),
        "ru_repetition":      repetition_score(russian),
        "numbers_preserved":  numbers_preserved(source, russian),
        "en_no_zh_leak":      no_chinese_leak(english),
        "ru_no_zh_leak":      no_chinese_leak(russian),
    }


def score_dataframe(
    df,
    source_col:  str,
    english_col: str,
    russian_col: str,
) -> dict:
    """
    Считает эвристики для всего датафрейма и печатает сводку.

    Возвращает:
        {
            "per_row": [{"en_lang_ok": True, ...}, ...],
            "summary": {"ru_lang_ok_pct": 95.0, ...}
        }
    """
    import pandas as pd

    rows = [
        score_row(str(src), str(en), str(ru))
        for src, en, ru in zip(df[source_col], df[english_col], df[russian_col])
    ]
    metrics_df = pd.DataFrame(rows)

    summary = {
        "en_lang_ok_pct":       round(metrics_df["en_lang_ok"].mean() * 100, 1),
        "ru_lang_ok_pct":       round(metrics_df["ru_lang_ok"].mean() * 100, 1),
        "avg_en_length_ratio":  round(metrics_df["en_length_ratio"].mean(), 3),
        "avg_ru_length_ratio":  round(metrics_df["ru_length_ratio"].mean(), 3),
        "avg_en_repetition":    round(metrics_df["en_repetition"].mean(), 3),
        "avg_ru_repetition":    round(metrics_df["ru_repetition"].mean(), 3),
        "avg_numbers_preserved":round(metrics_df["numbers_preserved"].mean(), 3),
        "en_no_zh_leak_pct":    round(metrics_df["en_no_zh_leak"].mean() * 100, 1),
        "ru_no_zh_leak_pct":    round(metrics_df["ru_no_zh_leak"].mean() * 100, 1),
    }

    print("\n=== ЭВРИСТИКИ ===")
    for k, v in summary.items():
        print(f"  {k:<28} {v}")

    return {"per_row": rows, "summary": summary}