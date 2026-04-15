from __future__ import annotations
import math
from typing import Iterator
import torch

def iter_batches(indices: list[int], batch_size: int) -> Iterator[list[int]]:
    """
    Генерирует батчи индексов размером batch_size.

    Пример:
        for batch in iter_batches([0,1,2,3,4], batch_size=2):
            # batch = [0,1], [2,3], [4]
    """
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def group_by_code(
    batch_idx: list[int],
    selected_codes: list[str],
    codes: list[str],
) -> dict[str, list[int]]:
    """
    Группирует индексы батча по языковому коду.
    Нужно потому что NLLB не может мешать языки в одном батче.

    Возвращает:
        {"zho_Hans": [0, 3], "zho_Hant": [1, 2]}
    """
    grouped: dict[str, list[int]] = {code: [] for code in codes}
    for i in batch_idx:
        code = selected_codes[i]
        if code in grouped:
            grouped[code].append(i)
    return grouped



def estimate_vram_usage_gb(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    num_beams: int,
    dtype_bytes: int = 2,  # float16 = 2 байта, float32 = 4
) -> float:
    """
    Грубая оценка пикового потребления VRAM в GB.

    Формула:
        активации = batch_size × seq_len × hidden_size × num_layers × 2 (enc+dec)
        beam буфер = активации × num_beams
        итого = веса модели + beam буфер

    Использование:
        estimate_vram_usage_gb(
            batch_size=4, seq_len=512,
            hidden_size=2048, num_layers=24,  # NLLB-3.3B
            num_beams=4
        )
    """
    activations_bytes = (
        batch_size * seq_len * hidden_size * num_layers * 2 * dtype_bytes
    )
    beam_buffer_bytes = activations_bytes * num_beams
    total_bytes       = activations_bytes + beam_buffer_bytes
    return round(total_bytes / (1024 ** 3), 2)


# Нужны для estimate_vram_usage_gb

MODEL_ARCH = {
    "facebook/nllb-200-3.3B": {
        "hidden_size": 2048, "num_layers": 24, "weights_gb": 13.0
    },
    "facebook/nllb-200-54b": {
        "hidden_size": 8192, "num_layers": 48, "weights_gb": 108.0
    },
}


def check_vram_fits(model_name: str, batch_size: int, num_beams: int) -> dict:
    """
    Проверяет влезет ли конфигурация в доступную VRAM.

    Возвращает:
        {
            "available_gb": 15.7,
            "estimated_gb": 12.3,
            "fits": True,
            "warning": None
        }
    """
    if not torch.cuda.is_available():
        return {"available_gb": 0, "estimated_gb": 0, "fits": False,
                "warning": "CUDA недоступна"}

    available_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    arch = MODEL_ARCH.get(model_name)

    if arch is None:
        return {"available_gb": round(available_gb, 1), "estimated_gb": None,
                "fits": None, "warning": f"Модель {model_name} неизвестна"}

    activation_gb = estimate_vram_usage_gb(
        batch_size=batch_size,
        seq_len=512,
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        num_beams=num_beams,
    )
    estimated_gb = round(arch["weights_gb"] + activation_gb, 1)
    fits         = estimated_gb < available_gb * 0.90  # оставляем 10% буфер

    warning = None
    if not fits:
        warning = (
            f"Оценочное потребление {estimated_gb} GB превышает "
            f"доступные {round(available_gb, 1)} GB. "
            f"Попробуй уменьшить batch_size или num_beams."
        )

    return {
        "available_gb": round(available_gb, 1),
        "estimated_gb": estimated_gb,
        "fits":         fits,
        "warning":      warning,
    }


def flush_gpu_cache() -> None:
    """Освобождает неиспользуемую VRAM после батча."""
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()