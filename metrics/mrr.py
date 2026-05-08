from __future__ import annotations


def reciprocal_rank(rank: int | None) -> float:
    """
    1 / rank, если rank найден.
    Если правильного ответа нет, возвращает 0.0
    """
    if rank is None:
        return 0.0
    if rank <= 0:
        raise ValueError("rank must be >= 1")
    return 1.0 / rank


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """
    Mean Reciprocal Rank.
    """
    if not ranks:
        return 0.0

    total = sum(reciprocal_rank(rank) for rank in ranks)
    return total / len(ranks)