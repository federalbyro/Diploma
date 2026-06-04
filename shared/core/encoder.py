from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME      = os.getenv("MODEL_NAME",      "sentence-transformers/LaBSE")
DEFAULT_MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models")
DEFAULT_MODEL_DEVICE    = os.getenv("MODEL_DEVICE",    "cpu")
MODEL_LOCAL_PATH        = os.getenv("MODEL_LOCAL_PATH", "").strip()
ENCODER_LOG             = os.getenv("ENCODER_LOG", "1") == "1"


def _log(msg: str) -> None:
    """Выводит логирующее сообщение с временем, если логирование включено."""
    if ENCODER_LOG:
        print(f"[encoder] {time.strftime('%H:%M:%S')} | {msg}", flush=True)


def _resolve_model_path() -> tuple[str, bool]:
    """Определяет путь к модели, проверяя локальные директории или используя HuggingFace."""
    # 1. Explicit local path
    if MODEL_LOCAL_PATH:
        local = Path(MODEL_LOCAL_PATH)
        if (local / "modules.json").exists():
            return str(local), True
        _log(f"WARN: MODEL_LOCAL_PATH={MODEL_LOCAL_PATH} set but modules.json not found, "
             "falling back to HuggingFace")

    # 2. Default local slot
    default_local = Path(DEFAULT_MODEL_CACHE_DIR) / "local"
    if (default_local / "modules.json").exists():
        return str(default_local), True

    # 3. HuggingFace
    return DEFAULT_MODEL_NAME, False


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Загружает и возвращает закэшированную модель SentenceTransformer."""
    path, is_local = _resolve_model_path()

    if is_local:
        _log(f"Loading local model: {path}")
        model = SentenceTransformer(path, device=DEFAULT_MODEL_DEVICE)
        _log("Local model loaded.")
    else:
        _log(f"Downloading from HuggingFace: {path}")
        _log(f"cache_folder: {DEFAULT_MODEL_CACHE_DIR}, device: {DEFAULT_MODEL_DEVICE}")
        model = SentenceTransformer(
            path,
            cache_folder=DEFAULT_MODEL_CACHE_DIR,
            device=DEFAULT_MODEL_DEVICE,
        )
        _log("HuggingFace model loaded.")

    return model


class Encoder:
    def __init__(self) -> None:
        """Инициализирует энкодер загрузкой модели SentenceTransformer."""
        self.model = get_model()

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Кодирует список текстов в нормализованные векторы эмбеддингов."""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def encode_single(self, text: str) -> list[float]:
        """Кодирует одиночный текст в вектор эмбеддинга."""
        return self.encode_texts([text], batch_size=1)[0]