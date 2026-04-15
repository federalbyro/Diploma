from __future__ import annotations

import pandas as pd
import torch

from config.model import BATCH_SIZE, CHINESE_CODES, ENGLISH_CODE, GENERATE_PARAMS, MAX_INPUT_LEN, RUSSIAN_CODE
from helpers.batching import flush_gpu_cache, group_by_code, iter_batches
from helpers.preprocessing import (
    auto_select_chinese_code,
    detect_row_lang,
    normalize_en_punct,
    normalize_zh_punct,
)


def _is_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def translate_batch_nllb(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    tokenizer,
    model,
    device: str,
    max_input_length: int = MAX_INPUT_LEN,
    generate_params: dict | None = None,
) -> list[str]:
    if not texts:
        return []

    params = {**GENERATE_PARAMS, **(generate_params or {})}

    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding=True,
    ).to(device)

    with torch.inference_mode():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            **params,
        )

    decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return [normalize_en_punct(x) for x in decoded]


def safe_translate_batch(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    tokenizer,
    model,
    device: str,
    max_input_length: int = MAX_INPUT_LEN,
    generate_params: dict | None = None,
    depth: int = 0,
) -> list[str]:
    """
    Пытается перевести батч целиком.
    Если словили OOM или битую строку — рекурсивно делит батч пополам,
    чтобы не уронить весь run.
    """
    if not texts:
        return []

    try:
        return translate_batch_nllb(
            texts=texts,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_input_length=max_input_length,
            generate_params=generate_params,
        )
    except Exception as exc:
        flush_gpu_cache()

        if len(texts) == 1:
            print(
                f"[translate] ⚠️ Не удалось перевести одну строку "
                f"({src_lang}→{tgt_lang}): {exc}"
            )
            # Не роняем прогон: возвращаем исходный текст как fallback.
            return [texts[0]]

        if _is_oom_error(exc):
            print(
                f"[translate] CUDA OOM на батче из {len(texts)} строк, "
                "делю батч пополам"
            )
        else:
            print(
                f"[translate] Ошибка на батче из {len(texts)} строк, "
                "пытаюсь локализовать проблемную строку"
            )

        mid = max(1, len(texts) // 2)
        left = safe_translate_batch(
            texts[:mid], src_lang, tgt_lang,
            tokenizer, model, device,
            max_input_length=max_input_length,
            generate_params=generate_params,
            depth=depth + 1,
        )
        right = safe_translate_batch(
            texts[mid:], src_lang, tgt_lang,
            tokenizer, model, device,
            max_input_length=max_input_length,
            generate_params=generate_params,
            depth=depth + 1,
        )
        return left + right


def translate_full_dataframe(
    df: pd.DataFrame,
    text_column: str,
    tokenizer,
    model,
    device: str,
    batch_size: int = BATCH_SIZE,
    chinese_codes: list[str] | None = None,
    generate_params: dict | None = None,
) -> pd.DataFrame:
    """
    Полный пайплайн: zh/en → en → ru.

    Возвращает df с колонками:
        english_text, russian_text, selected_chinese_code
    """
    if chinese_codes is None:
        chinese_codes = CHINESE_CODES

    if text_column not in df.columns:
        raise ValueError(f"Столбец '{text_column}' не найден.")

    texts: list[str] = df[text_column].fillna("").astype(str).str.strip().tolist()
    n = len(texts)

    zh_indices: list[int] = []
    en_indices: list[int] = []
    selected_codes: list[str] = [""] * n

    for i, text in enumerate(texts):
        lang = detect_row_lang(text)
        if lang == "zh":
            zh_indices.append(i)
            selected_codes[i] = auto_select_chinese_code(text, chinese_codes)
        elif lang == "en":
            en_indices.append(i)

    english_texts: list[str] = list(texts)
    russian_texts: list[str] = list(texts)

    for batch_idx in iter_batches(zh_indices, batch_size):
        grouped = group_by_code(batch_idx, selected_codes, chinese_codes)

        for code, sub_idx in grouped.items():
            if not sub_idx:
                continue

            sub_texts = [normalize_zh_punct(texts[i]) for i in sub_idx]
            translated = safe_translate_batch(
                texts=sub_texts,
                src_lang=code,
                tgt_lang=ENGLISH_CODE,
                tokenizer=tokenizer,
                model=model,
                device=device,
                generate_params=generate_params,
            )
            for i, en in zip(sub_idx, translated):
                english_texts[i] = en

        flush_gpu_cache()

    all_to_ru = zh_indices + en_indices
    for batch_idx in iter_batches(all_to_ru, batch_size):
        batch_en = [normalize_en_punct(english_texts[i]) for i in batch_idx]
        translated = safe_translate_batch(
            texts=batch_en,
            src_lang=ENGLISH_CODE,
            tgt_lang=RUSSIAN_CODE,
            tokenizer=tokenizer,
            model=model,
            device=device,
            generate_params=generate_params,
        )
        for i, ru in zip(batch_idx, translated):
            russian_texts[i] = ru

        flush_gpu_cache()

    result_df = df.copy()
    result_df["english_text"] = english_texts
    result_df["russian_text"] = russian_texts
    result_df["selected_chinese_code"] = selected_codes
    return result_df
