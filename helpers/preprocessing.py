import re
from opencc import OpenCC

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE   = re.compile(r"[A-Za-z]")
NUMBER_RE  = re.compile(r"\d+\.?\d*")

_cc_t2s = OpenCC("t2s")  # traditional → simplified
_cc_s2t = OpenCC("s2t")  # simplified  → traditional


DUPLICATE_PHRASE_RE = re.compile(r'\b(.+?)(?:\s*,\s*|\s+)\1\b', re.IGNORECASE)
MULTISPACE_RE = re.compile(r'\s+')
DUP_WORD_RE = re.compile(r'\b(\w+)(\s+\1\b)+', re.IGNORECASE)

def clean_english_pivot(text: str) -> str:
    text = str(text).strip()

    text = MULTISPACE_RE.sub(" ", text)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", text)

    text = re.sub(r"([,.;:!?])\1+", r"\1", text)

    text = DUP_WORD_RE.sub(r"\1", text)

    text = re.sub(r"(\d)\s+(mm|cm|m|kg|g|mpa|bar|v|w|kw|hz|l|ml|°c)\b", r"\1\2", text, flags=re.IGNORECASE)

    text = re.sub(r"(\d)\s*[x×]\s*(\d)", r"\1x\2", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s*[x×]\s*(\d)\s*[x×]\s*(\d)", r"\1x\2x\3", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+([)\]])", r"\1", text)
    text = re.sub(r"([(\\[])\s+", r"\1", text)

    text = text.strip(" ,;.")

    return text

def normalize_zh_punct(text: str) -> str:
    """
    Заменяет китайские знаки препинания на западные аналоги.
    Модель лучше работает с западной пунктуацией на входе.
    """
    mapping = {
        "，": ", ",
        "。": ". ",
        "！": "! ",
        "？": "? ",
        "；": "; ",
        "：": ": ",
        "「": '"',
        "」": '"',
        "『": "'",
        "』": "'",
        "（": "(",
        "）": ")",
        "\u3000": " ",  # идеографический пробел
    }
    for zh, en in mapping.items():
        text = text.replace(zh, en)
    return text.strip()


def normalize_en_punct(text: str) -> str:
    """
    Убирает задвоенные пробелы и артефакты после перевода.
    Применяется к английскому и русскому тексту после model.generate().
    """
    text = re.sub(r" {2,}", " ", text)        # задвоенные пробелы
    text = re.sub(r" ([,\.!?;:])", r"\1", text)  # пробел перед пунктуацией
    return text.strip()


def extract_numbers(text: str) -> list[str]:
    """
    Извлекает все числовые значения из текста.
    Используется для проверки сохранности чисел после перевода.
    """
    return NUMBER_RE.findall(text)


def check_numbers_preserved(source: str, hypothesis: str) -> float:
    """
    Возвращает долю чисел из source, которые сохранились в hypothesis.
    1.0 = все числа на месте, 0.0 = все числа потеряны.
    """
    src_nums  = set(extract_numbers(source))
    hyp_nums  = set(extract_numbers(hypothesis))
    if not src_nums:
        return 1.0
    preserved = src_nums & hyp_nums
    return len(preserved) / len(src_nums)



def detect_row_lang(text: str) -> str:
    """
    Возвращает 'zh', 'en' или 'other'.
    Логика: считаем иероглифы vs латинские буквы.
    """
    text     = str(text)
    zh_count = len(CHINESE_RE.findall(text))
    en_count = len(LATIN_RE.findall(text))

    if zh_count == 0 and en_count == 0:
        return "other"
    if zh_count > en_count:
        return "zh"
    return "en"



def _diff_score(a: str, b: str) -> int:
    """Количество позиций где строки различаются."""
    max_len = max(len(a), len(b))
    a = a.ljust(max_len)
    b = b.ljust(max_len)
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def auto_select_chinese_code(
    text: str,
    chinese_codes: list[str] | None = None,
    default_code: str = "zho_Hans",
) -> str:
    """
    Автоматически выбирает zho_Hans (упрощённый) или zho_Hant (традиционный).
    Сравнивает исходный текст с его конвертированными версиями через OpenCC.
    """
    if chinese_codes is None:
        chinese_codes = ["zho_Hans", "zho_Hant"]

    chinese_only = "".join(CHINESE_RE.findall(str(text)))
    if not chinese_only:
        return default_code

    simp_version = _cc_t2s.convert(chinese_only)
    trad_version = _cc_s2t.convert(chinese_only)

    diff_to_simp = _diff_score(chinese_only, simp_version)
    diff_to_trad = _diff_score(chinese_only, trad_version)

    if "zho_Hans" in chinese_codes and "zho_Hant" in chinese_codes:
        if diff_to_simp < diff_to_trad:
            return "zho_Hans"
        if diff_to_trad < diff_to_simp:
            return "zho_Hant"
        return default_code

    return chinese_codes[0]



def split_long_text(
    text: str,
    lang: str = "zh",
    max_chunk_len: int = 180,
) -> list[str]:
    """
    Разбивает длинный текст на чанки до max_chunk_len символов.
    Старается резать по знакам препинания, не по середине слова.

    lang='zh' — режет по китайским/западным разделителям
    lang='en' — режет по пробелам и западной пунктуации
    """
    text = text.strip()
    if len(text) <= max_chunk_len:
        return [text]

    if lang == "zh":
        split_pattern = re.compile(r"(?<=[。！？；,\.!?;])\s*")
    else:
        split_pattern = re.compile(r"(?<=[\.!?;])\s+")

    segments = split_pattern.split(text)

    chunks, current = [], ""
    for seg in segments:
        if len(current) + len(seg) <= max_chunk_len:
            current += seg
        else:
            if current:
                chunks.append(current.strip())
            # если один сегмент сам по себе длиннее лимита — берём как есть
            current = seg

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]