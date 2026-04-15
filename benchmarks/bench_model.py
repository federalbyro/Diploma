from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class TimerResult:
    label: str
    seconds: float = 0.0

    @property
    def rows_per_second(self) -> float | None:
        return None


@contextmanager
def timed(label: str) -> Iterator[TimerResult]:
    result = TimerResult(label=label)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result.seconds = time.perf_counter() - start


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)


def get_vram_stats() -> dict:
    if not torch.cuda.is_available():
        return {
            "vram_used_gb": None,
            "vram_reserved_gb": None,
            "vram_peak_gb": None,
            "vram_total_gb": None,
        }

    props = torch.cuda.get_device_properties(0)
    return {
        "vram_used_gb": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2),
        "vram_total_gb": round(props.total_memory / (1024 ** 3), 2),
    }


def print_run_header(model_name: str, batch_size: int, files_count: int, output_dir) -> None:
    print("\n" + "=" * 78)
    print("NLLB TEST RUN")
    print("=" * 78)
    print(f"Модель:      {model_name}")
    print(f"Batch size:  {batch_size}")
    print(f"Файлов:      {files_count}")
    print(f"Output dir:  {output_dir}")
    if torch.cuda.is_available():
        stats = get_vram_stats()
        print(f"GPU VRAM:    {stats['vram_total_gb']} GB")
    else:
        print("GPU VRAM:    CUDA недоступна")
    print("=" * 78)


def print_file_header(file_name: str, rows: int, label_found: bool) -> None:
    print("\n" + "─" * 78)
    print(f"Файл:        {file_name}")
    print(f"Строк:       {rows}")
    print(f"Эталон:      {'да' if label_found else 'нет'}")
    print("─" * 78)


def print_file_summary(result: dict) -> None:
    print("[bench] Итог по файлу")
    print(f"  translate_sec           {result['translate_sec']:.2f}")
    print(f"  rows_per_sec            {result['rows_per_sec']:.2f}")
    if result.get("vram_peak_gb") is not None:
        print(f"  vram_peak_gb            {result['vram_peak_gb']:.2f}")
    if result.get("chrf") is not None:
        print(f"  chrF                    {result['chrf']:.2f}")
    if result.get("bleu") is not None:
        print(f"  BLEU                    {result['bleu']:.2f}")

    heur = result.get("heuristics") or {}
    for key in (
        "ru_lang_ok_pct",
        "avg_numbers_preserved",
        "ru_no_zh_leak_pct",
        "avg_ru_repetition",
    ):
        if key in heur:
            print(f"  {key:<24} {heur[key]}")

    print(f"  output                  {result['output_file']}")


def print_final_summary(results: list[dict]) -> None:
    if not results:
        return

    print("\n" + "=" * 78)
    print("ОБЩАЯ СВОДКА")
    print("=" * 78)

    total_rows = sum(x["rows"] for x in results)
    total_time = sum(x["translate_sec"] for x in results)
    overall_rps = total_rows / total_time if total_time else 0.0

    print(f"Всего файлов:            {len(results)}")
    print(f"Всего строк:             {total_rows}")
    print(f"Суммарное время:         {total_time:.2f} sec")
    print(f"Средняя скорость:        {overall_rps:.2f} строк/сек")

    with_refs = [x for x in results if x.get("chrf") is not None]
    if with_refs:
        avg_chrf = sum(x["chrf"] for x in with_refs) / len(with_refs)
        avg_bleu = sum(x["bleu"] for x in with_refs) / len(with_refs)
        print(f"Средний chrF:            {avg_chrf:.2f}")
        print(f"Средний BLEU:            {avg_bleu:.2f}")

    ru_lang = [x["heuristics"].get("ru_lang_ok_pct") for x in results if x.get("heuristics")]
    if ru_lang:
        print(f"Средний ru_lang_ok_pct:  {sum(ru_lang) / len(ru_lang):.2f}")

    print("=" * 78)

    print("\nПо файлам:")
    for item in results:
        chrf = "-" if item.get("chrf") is None else f"{item['chrf']:.2f}"
        bleu = "-" if item.get("bleu") is None else f"{item['bleu']:.2f}"
        print(
            f"  {item['file_name']:<28} "
            f"rows={item['rows']:<4} "
            f"sec={item['translate_sec']:<7.2f} "
            f"rps={item['rows_per_sec']:<7.2f} "
            f"chrF={chrf:<6} BLEU={bleu:<6}"
        )
