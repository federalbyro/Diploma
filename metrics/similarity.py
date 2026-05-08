from __future__ import annotations

from statistics import mean


def mean_similarity(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def similarity_gap(
    positive_scores: list[float],
    negative_scores: list[float],
) -> dict[str, float | None]:
    """
    Считает среднее similarity для релевантных и нерелевантных пар
    и разницу между ними.
    """
    pos_mean = mean_similarity(positive_scores)
    neg_mean = mean_similarity(negative_scores)

    if pos_mean is None or neg_mean is None:
        gap = None
    else:
        gap = pos_mean - neg_mean

    return {
        "avg_positive_similarity": round(pos_mean, 6) if pos_mean is not None else None,
        "avg_negative_similarity": round(neg_mean, 6) if neg_mean is not None else None,
        "similarity_gap": round(gap, 6) if gap is not None else None,
    }