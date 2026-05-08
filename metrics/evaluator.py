from __future__ import annotations

from dataclasses import dataclass

from test.metrics.mrr import mean_reciprocal_rank
from test.metrics.similarity import similarity_gap
from test.metrics.topk import top_k_report


@dataclass
class MatchCandidate:
    category_id: int
    score: float


@dataclass
class MatchCase:
    item_id: int | str
    true_category_id: int
    predicted: list[MatchCandidate]


def find_true_rank(case: MatchCase) -> int | None:
    """
    Возвращает позицию правильной категории в списке predicted.
    Нумерация с 1.
    """
    for idx, candidate in enumerate(case.predicted, start=1):
        if candidate.category_id == case.true_category_id:
            return idx
    return None


def collect_positive_negative_scores(case: MatchCase) -> tuple[list[float], list[float]]:
    positive: list[float] = []
    negative: list[float] = []

    for candidate in case.predicted:
        if candidate.category_id == case.true_category_id:
            positive.append(candidate.score)
        else:
            negative.append(candidate.score)

    return positive, negative


def evaluate_matches(cases: list[MatchCase]) -> dict:
    """
    Общий расчёт метрик качества категоризации.
    """
    if not cases:
        return {
            "count": 0,
            "top_1": 0.0,
            "top_3": 0.0,
            "top_5": 0.0,
            "mrr": 0.0,
            "avg_positive_similarity": None,
            "avg_negative_similarity": None,
            "similarity_gap": None,
        }

    ranks: list[int | None] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for case in cases:
        rank = find_true_rank(case)
        ranks.append(rank)

        pos, neg = collect_positive_negative_scores(case)
        positive_scores.extend(pos)
        negative_scores.extend(neg)

    result = {
        "count": len(cases),
        **top_k_report(ranks, ks=(1, 3, 5)),
        "mrr": round(mean_reciprocal_rank(ranks), 6),
        **similarity_gap(positive_scores, negative_scores),
    }

    return result