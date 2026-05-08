from __future__ import annotations


def top_k_accuracy(ranks: list[int | None], k: int) -> float:
    """
    Доля объектов, у которых правильный ответ попал в top-k.
    rank = 1 означает первое место.
    rank = None означает, что правильный ответ не найден.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if not ranks:
        return 0.0

    hits = 0
    for rank in ranks:
        if rank is not None and rank <= k:
            hits += 1

    return hits / len(ranks)


def top_k_report(ranks: list[int | None], ks: list[int] | tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    """
    Возвращает словарь вида:
    {
        "top_1": 0.72,
        "top_3": 0.91,
        "top_5": 0.96
    }
    """
    report: dict[str, float] = {}
    for k in ks:
        report[f"top_{k}"] = round(top_k_accuracy(ranks, k), 6)
    return report