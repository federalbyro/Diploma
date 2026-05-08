from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/LaBSE")
DEFAULT_MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models")
DEFAULT_MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

# 0 — тихий режим, 1 — логировать загрузку модели
ENCODER_LOG = os.getenv("ENCODER_LOG", "1") == "1"


def log(message: str) -> None:
    if ENCODER_LOG:
        print(f"[encoder] {time.strftime('%H:%M:%S')} | {message}", flush=True)


@lru_cache(maxsize=1)
def get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    log(f"START loading model: {model_name}")
    log(f"cache_folder: {DEFAULT_MODEL_CACHE_DIR}")
    log(f"device: {DEFAULT_MODEL_DEVICE}")

    model = SentenceTransformer(
        model_name,
        cache_folder=DEFAULT_MODEL_CACHE_DIR,
        device=DEFAULT_MODEL_DEVICE,
    )

    log(f"END loading model: {model_name}")
    return model


class Encoder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model = get_model(model_name)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
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
        vectors = self.encode_texts([text], batch_size=1)
        return vectors[0]