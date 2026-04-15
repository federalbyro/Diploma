from pathlib import Path

ROOT = Path(__file__).parent.parent


TEST_DIR        = ROOT / "test"
INPUT_DIR       = TEST_DIR / "data" / "input"    
LABEL_DIR       = TEST_DIR / "data" / "label"    # эталонные переводы для BLEU/chrF
OUTPUT_DIR      = TEST_DIR / "data" / "output"   # результаты перевода


# Чтобы не качать модель каждый раз заново
MODEL_CACHE_DIR = ROOT / ".model_cache"

# ── Результаты бенчмарков ────────────────────────────────────
BENCH_RESULTS_DIR = ROOT / "benchmarks" / "results"

# ── Создаём папки если не существуют ────────────────────────
for _dir in [INPUT_DIR, LABEL_DIR, OUTPUT_DIR, MODEL_CACHE_DIR, BENCH_RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)