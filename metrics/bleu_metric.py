from sacrebleu.metrics import BLEU


def compute_bleu(
    hypotheses: list[str],
    references: list[str],
    tokenize:   str = "char",
) -> dict:
    """
    Считает BLEU для списка гипотез и эталонов.

    tokenize="char" — символьный уровень, лучше для русского.
    tokenize="13a"  — стандартный токенизатор (для сравнения с другими работами).

    Возвращает:
        {
            "bleu":    32.14,
            "details": "BLEU = 32.14 ..."  # полная строка sacrebleu
        }
    """
    metric = BLEU(tokenize=tokenize)
    result = metric.corpus_score(hypotheses, [references])

    return {
        "bleu":    round(result.score, 2),
        "details": str(result),
    }