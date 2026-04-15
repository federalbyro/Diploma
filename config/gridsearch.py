from itertools import product
from typing import Iterator

PARAM_GRID = {
    "num_beams":              [2, 6, 8],
    "repetition_penalty":     [1.0, 1.1, 1.3],
    "no_repeat_ngram_size":   [3, 4],
    "length_penalty":         [0.8, 1.0, 1.2],
}


def iter_param_combinations() -> Iterator[dict]:
    """
    Генерирует все комбинации параметров из PARAM_GRID.

    Использование:
        for params in iter_param_combinations():
            score = run_translation_with_params(params)
    """
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    for combo in product(*values):
        yield dict(zip(keys, combo))


def total_combinations() -> int:
    """Количество комбинаций для прогресс-бара."""
    result = 1
    for v in PARAM_GRID.values():
        result *= len(v)
    return result