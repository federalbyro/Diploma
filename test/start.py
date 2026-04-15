import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.bench_model import (
    get_vram_stats,
    print_file_header,
    print_file_summary,
    print_final_summary,
    print_run_header,
    reset_peak_vram,
    timed,
)
from config.model import BATCH_SIZE, MODEL_NAME
from config.paths import INPUT_DIR, LABEL_DIR, OUTPUT_DIR
from helpers.dataset import (
    build_output_file,
    list_input_files,
    load_input_dataframe,
    load_reference_texts,
    make_run_output_dir,
    resolve_label_file,
)
from helpers.loader import load_model
from helpers.translator import translate_full_dataframe
from metrics.bleu_metric import compute_bleu
from metrics.chrf_metric import compute_chrf
from metrics.heuristics import score_dataframe

TEXT_COLUMN = "source"
REF_COLUMN = "reference_ru"


def run_tests() -> None:
    input_files = list_input_files(INPUT_DIR)
    if not input_files:
        print(f"[test] Нет CSV файлов в {INPUT_DIR}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = make_run_output_dir(OUTPUT_DIR, timestamp=timestamp)

    print_run_header(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        files_count=len(input_files),
        output_dir=run_output_dir,
    )

    with timed("model_load") as t_load:
        tokenizer, model, device = load_model(MODEL_NAME)
    print(f"[bench] Загрузка модели: {t_load.seconds:.2f} sec")

    results: list[dict] = []

    for input_file in input_files:
        label_file = resolve_label_file(input_file, LABEL_DIR)
        df = load_input_dataframe(input_file, text_column=TEXT_COLUMN)

        print_file_header(
            file_name=input_file.name,
            rows=len(df),
            label_found=label_file is not None,
        )

        reset_peak_vram()
        with timed("translate") as t_translate:
            translated_df = translate_full_dataframe(
                df=df,
                text_column=TEXT_COLUMN,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=BATCH_SIZE,
            )

        heuristics = score_dataframe(
            translated_df,
            TEXT_COLUMN,
            "english_text",
            "russian_text",
        )

        chrf_score = None
        bleu_score = None
        if label_file is not None:
            try:
                references = load_reference_texts(label_file, ref_column=REF_COLUMN)
                hypotheses = translated_df["russian_text"].fillna("").astype(str).tolist()

                if not references:
                    print(f"[test] Эталон пустой: {label_file.name}")
                else:
                    if len(references) != len(hypotheses):
                        limit = min(len(references), len(hypotheses))
                        print(
                            f"[test] ⚠️ Длины не совпали: refs={len(references)}, hyps={len(hypotheses)}. "
                            f"Считаю по первым {limit} строкам."
                        )
                        references = references[:limit]
                        hypotheses = hypotheses[:limit]

                    chrf_score = compute_chrf(hypotheses, references)["chrf"]
                    bleu_score = compute_bleu(hypotheses, references)["bleu"]
                    print(f"[metric] chrF: {chrf_score:.2f}")
                    print(f"[metric] BLEU: {bleu_score:.2f}")
            except Exception as exc:
                print(f"[test] ⚠️ Не удалось посчитать метрики по эталону: {exc}")
        else:
            print("[test] Эталон не найден — BLEU/chrF пропущены")

        output_file = build_output_file(run_output_dir, input_file)
        translated_df[["russian_text"]].to_csv(
            output_file,
            index=False,
            header=False,
            encoding="utf-8",
        )

        vram = get_vram_stats()
        rows_per_sec = len(df) / t_translate.seconds if t_translate.seconds else 0.0
        file_result = {
            "file_name": input_file.name,
            "rows": len(df),
            "translate_sec": round(t_translate.seconds, 2),
            "rows_per_sec": round(rows_per_sec, 2),
            "chrf": chrf_score,
            "bleu": bleu_score,
            "heuristics": heuristics["summary"],
            "output_file": str(output_file),
            **vram,
        }
        results.append(file_result)
        print_file_summary(file_result)

    print_final_summary(results)


if __name__ == "__main__":
    run_tests()
