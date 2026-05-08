from __future__ import annotations

from typing import Iterable, Iterator, Sequence, TypeVar
import gc

import torch

T = TypeVar("T")


def iter_batches(items: Sequence[T], batch_size: int) -> Iterator[list[T]]:
    """
    Разбивает последовательность на батчи фиксированного размера.

    Пример:
        list(iter_batches([1, 2, 3, 4, 5], 2))
        -> [[1, 2], [3, 4], [5]]
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def iter_index_batches(size: int, batch_size: int) -> Iterator[list[int]]:
    """
    Возвращает батчи индексов [0..size-1].
    Удобно, когда не хочется копировать сами объекты.
    """
    if size < 0:
        raise ValueError("size must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    for start in range(0, size, batch_size):
        yield list(range(start, min(start + batch_size, size)))


def chunk_iterable(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """
    Батчинг для произвольного итерируемого объекта.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def flush_memory() -> None:
    """
    Освобождает Python heap и CUDA cache после тяжёлого батча.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def safe_batch_size(
    requested_batch_size: int,
    hard_max: int | None = None,
) -> int:
    """
    Возвращает безопасный размер батча.
    Можно использовать как простой guardrail в конфиге.
    """
    if requested_batch_size <= 0:
        raise ValueError("requested_batch_size must be > 0")

    if hard_max is None:
        return requested_batch_size

    return max(1, min(requested_batch_size, hard_max))