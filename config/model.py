MODEL_NAME   = "facebook/nllb-200-3.3B"              # ~10 GB VRAM — качество
# MODEL_NAME = "facebook/nllb-200-54b"              # ~110 GB VRAM — исследования


ENGLISH_CODE = "eng_Latn"
RUSSIAN_CODE = "rus_Cyrl"
CHINESE_CODES = ["zho_Hans", "zho_Hant"]  # упрощённый / традиционный


MODEL_LOAD_CONFIG = {

    "load_in_8bit": False,    #  — нужно будет включить для 54B на ограниченной VRAM 
    "device_map": "auto",
}


GENERATE_PARAMS = {
    "num_beams":              6,      # beam search: больше = качество ↑, скорость ↓
    "repetition_penalty":     1.05,   # штраф за повторяющиеся токены
    "no_repeat_ngram_size":   3,     # запрет повтора 3-грамм
    "length_penalty":         1.0,   # > 1 поощряет длинные переводы
    "max_new_tokens":         256,
    "early_stopping":         True,
}


BATCH_SIZE      = 4    # строк за один вызов model.generate()
MAX_INPUT_LEN   = 512  # максимум токенов на входе