from sacrebleu.metrics import CHRF


def compute_chrf(
    hypotheses: list[str],
    references: list[str],
    char_order: int = 6,
    beta:       int = 2,
) -> dict:
    """
    Считает chrF для списка гипотез и эталонов.

    char_order=6  — стандарт, длина символьных n-грамм
    beta=2        — полнота весит вдвое больше точности (стандарт chrF)

    Возвращает:
        {
            "chrf":       54.32,   # системный score
            "per_row":    [51.2, 58.1, ...]  # по каждой строке
        }
    """
    metric = CHRF(char_order=char_order, beta=beta)

    system_score = metric.corpus_score(hypotheses, [references]).score

    per_row = [
        metric.sentence_score(hyp, [ref]).score
        for hyp, ref in zip(hypotheses, references)
    ]

    return {
        "chrf":    round(system_score, 2),
        "per_row": [round(s, 2) for s in per_row],
    }