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
    rows: int | None = None

    @property
    def rows_per_second(self) -> float | None:
        if not self.rows or self.seconds <= 0:
            return None
        return self.rows / self.seconds


@contextmanager
def timed(label: str, rows: int | None = None) -> Iterator[TimerResult]:
    result = TimerResult(label=label, rows=rows)

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


def print_stage(label: str, seconds: float, rows: int | None = None) -> None:
    line = f"[bench] {label:<24} {seconds:.4f} sec"
    if rows:
        rps = rows / seconds if seconds > 0 else 0.0
        line += f" | rows={rows} | rows/sec={rps:.2f}"
    print(line)


def print_dict(title: str, data: dict) -> None:
    print(f"\n{title}")
    for key, value in data.items():
        print(f"  {key:<24} {value}")