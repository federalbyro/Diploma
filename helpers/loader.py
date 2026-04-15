import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config.model  import MODEL_NAME, MODEL_LOAD_CONFIG
from config.paths  import MODEL_CACHE_DIR
from helpers.batching import check_vram_fits, MODEL_ARCH


def load_model(
    model_name: str = MODEL_NAME,
    verbose: bool = True,
) -> tuple:
    """
    Загружает токенизатор и модель.

    Перед загрузкой проверяет оценочное потребление VRAM
    и предупреждает если может не влезть.

    Возвращает:
        (tokenizer, model, device)
    """
    from config.model import BATCH_SIZE, GENERATE_PARAMS

    if verbose:
        print(f"[loader] Модель:  {model_name}")
        print(f"[loader] Кэш:     {MODEL_CACHE_DIR}")

    vram_check = check_vram_fits(
        model_name=model_name,
        batch_size=BATCH_SIZE,
        num_beams=GENERATE_PARAMS["num_beams"],
    )

    if verbose:
        print(f"[loader] VRAM доступно:  {vram_check['available_gb']} GB")
        if vram_check["estimated_gb"] is not None:
            print(f"[loader] VRAM оценочно:  {vram_check['estimated_gb']} GB")
        if vram_check["warning"]:
            print(f"[loader] ⚠️  {vram_check['warning']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(MODEL_CACHE_DIR),
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        load_in_8bit=MODEL_LOAD_CONFIG.get("load_in_8bit", False),
        device_map=MODEL_LOAD_CONFIG.get("device_map", "auto"),
        cache_dir=str(MODEL_CACHE_DIR),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if MODEL_LOAD_CONFIG.get("device_map") != "auto":
        model = model.to(device)

    if verbose:
        print(f"[loader] Устройство: {device}")
        print(f"[loader] ✅ Модель загружена")

    return tokenizer, model, device