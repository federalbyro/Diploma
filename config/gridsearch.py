import csv
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.model import MODEL_NAME, BATCH_SIZE
from config.paths import INPUT_DIR, LABEL_DIR
from helpers.dataset import list_input_files, load_input_dataframe, load_reference_texts, resolve_label_path
from helpers.loader import load_model
from helpers.translator import translate_full_dataframe
from metrics.bleu_metric import compute_bleu
from metrics.chrf_metric import compute_chrf
from benchmarks.bench_model import timed
from itertools import product

TEXT_COLUMN = "source"

PARAM_GRID = {
    "num_beams": [2, 6],
    "repetition_penalty": [1.05, 1.1, 1.3],
    "no_repeat_ngram_size": [3, 4],
    "length_penalty": [0.8, 1.0, 1.2],
}

TARGET_FILES = [
    "nllb_middle_trad.csv",
    "sanitary_goods.csv",
]

RESULTS_FILE = ROOT / "gridsearch_results.csv"


def iter_param_combinations():
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))


def total_combinations() -> int:
    result = 1
    for v in PARAM_GRID.values():
        result *= len(v)
    return result


def score_run(chrf_score: float, bleu_score: float, rows_per_sec: float) -> float:
    # Простой составной скор
    return round(chrf_score * 0.45 + bleu_score * 0.45 + rows_per_sec * 10 * 0.10, 4)


def run_gridsearch():
    input_files = [p for p in list_input_files(INPUT_DIR) if p.name in TARGET_FILES]
    if not input_files:
        print("[grid] Нет целевых файлов для gridsearch")
        return

    print(f"[grid] Файлов: {len(input_files)}")
    print(f"[grid] Комбинаций: {total_combinations()}")
    print(f"[grid] Модель: {MODEL_NAME}")

    with timed("model_load") as t:
        tokenizer, model, device = load_model(MODEL_NAME)
    print(f"[grid] Модель загружена за {t.seconds:.2f} sec")

    rows_out = []
    best = None

    combos = list(iter_param_combinations())
    total = len(combos)

    for idx, params in enumerate(combos, start=1):
        print("=" * 78)
        print(f"[grid] {idx}/{total}  params={params}")

        bleu_scores = []
        chrf_scores = []
        speed_scores = []

        ok = True

        for input_file in input_files:
            label_file = resolve_label_path(input_file, LABEL_DIR)
            if label_file is None:
                print(f"[grid] Нет label для {input_file.name}, пропуск")
                ok = False
                break

            df = load_input_dataframe(input_file, text_column=TEXT_COLUMN)

            with timed("translate") as t_translate:
                translated_df = translate_full_dataframe(
                    df=df,
                    text_column=TEXT_COLUMN,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=BATCH_SIZE,
                    generation_params=params,
                )

            references = load_reference_texts(label_file)
            hypotheses = translated_df["russian_text"].fillna("").astype(str).tolist()

            if len(references) != len(hypotheses):
                raise ValueError(
                    f"Длины не совпали для {input_file.name}: refs={len(references)}, hyps={len(hypotheses)}"
                )

            chrf_score = compute_chrf(hypotheses, references)["chrf"]
            bleu_score = compute_bleu(hypotheses, references)["bleu"]
            rows_per_sec = len(df) / t_translate.seconds if t_translate.seconds else 0.0

            print(
                f"[grid] {input_file.name}: "
                f"chrF={chrf_score:.2f}  BLEU={bleu_score:.2f}  rps={rows_per_sec:.2f}"
            )

            bleu_scores.append(bleu_score)
            chrf_scores.append(chrf_score)
            speed_scores.append(rows_per_sec)

        if not ok:
            continue

        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_chrf = sum(chrf_scores) / len(chrf_scores)
        avg_rps = sum(speed_scores) / len(speed_scores)
        final_score = score_run(avg_chrf, avg_bleu, avg_rps)

        row = {
            **params,
            "avg_bleu": round(avg_bleu, 4),
            "avg_chrf": round(avg_chrf, 4),
            "avg_rows_per_sec": round(avg_rps, 4),
            "final_score": final_score,
        }
        rows_out.append(row)

        if best is None or row["final_score"] > best["final_score"]:
            best = row

        print(
            f"[grid] AVG -> chrF={avg_chrf:.2f}  BLEU={avg_bleu:.2f}  "
            f"rps={avg_rps:.2f}  final={final_score:.4f}"
        )

    if rows_out:
        with open(RESULTS_FILE, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

        print("=" * 78)
        print(f"[grid] Результаты сохранены: {RESULTS_FILE}")
        print("[grid] Лучший пресет:")
        print(best)
    else:
        print("[grid] Нет результатов")


if __name__ == "__main__":
    run_gridsearch()