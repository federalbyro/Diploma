## Метрики

- **BLEU** — показывает, насколько перевод совпадает с эталоном по словам и словосочетаниям.
- **chrF** — показывает, насколько перевод похож на эталон на уровне символов и символьных сочетаний.
- **translate_sec** — время обработки файла в секундах.
- **rows_per_sec** — скорость обработки в строках в секунду.
- **vram_peak_gb** — пиковое использование видеопамяти GPU во время обработки.
- **en_lang_ok_pct** — доля строк после шага `zh → en`, которые распознаются как английский текст.
- **ru_lang_ok_pct** — доля строк финального результата, которые распознаются как русский текст.
- **avg_en_length_ratio** — среднее отношение длины английского промежуточного текста к длине исходного текста.
- **avg_ru_length_ratio** — среднее отношение длины русского результата к длине исходного текста.
- **avg_en_repetition** — средний уровень повторяемости текста на английском промежуточном шаге.
- **avg_ru_repetition** — средний уровень повторяемости текста в финальном русском результате.
- **avg_numbers_preserved** — доля чисел и числовых характеристик, которые сохранились после перевода.
- **en_no_zh_leak_pct** — доля английских промежуточных строк, в которых не осталось китайских символов.
- **ru_no_zh_leak_pct** — доля русских строк, в которых не осталось китайских символов.

---

# Модель NLLB3.3B

## Файлы

### `aviation_simplified.csv`

- **chrF:** 39.51
- **BLEU:** 27.05
- **translate_sec:** 107.42
- **rows_per_sec:** 1.86
- **vram_peak_gb:** 7.13
- **en_lang_ok_pct:** 98.0
- **ru_lang_ok_pct:** 99.0
- **avg_en_length_ratio:** 2.861
- **avg_ru_length_ratio:** 3.337
- **avg_en_repetition:** 0.971
- **avg_ru_repetition:** 0.99
- **avg_numbers_preserved:** 0.86
- **en_no_zh_leak_pct:** 98.0
- **ru_no_zh_leak_pct:** 98.5

### `nllb_middle_trad.csv`

- **chrF:** 44.17
- **BLEU:** 39.99
- **translate_sec:** 198.93
- **rows_per_sec:** 0.50
- **vram_peak_gb:** 8.83
- **en_lang_ok_pct:** 100.0
- **ru_lang_ok_pct:** 100.0
- **avg_en_length_ratio:** 3.765
- **avg_ru_length_ratio:** 4.148
- **avg_en_repetition:** 0.879
- **avg_ru_repetition:** 0.922
- **avg_numbers_preserved:** 0.994
- **en_no_zh_leak_pct:** 100.0
- **ru_no_zh_leak_pct:** 100.0

### `sanitary_goods.csv`

#### Из полного прогона

- **chrF:** 39.46
- **BLEU:** 44.66
- **translate_sec:** 349.50
- **rows_per_sec:** 0.72
- **vram_peak_gb:** 8.29
- **en_lang_ok_pct:** 100.0
- **ru_lang_ok_pct:** 100.0
- **avg_en_length_ratio:** 3.022
- **avg_ru_length_ratio:** 3.207
- **avg_en_repetition:** 0.892
- **avg_ru_repetition:** 0.953
- **avg_numbers_preserved:** 0.856
- **en_no_zh_leak_pct:** 100.0
- **ru_no_zh_leak_pct:** 100.0

#### Из более позднего отдельного прогона

- **chrF:** 38.74
- **BLEU:** 43.49
- **translate_sec:** 392.45
- **rows_per_sec:** 0.64
- **vram_peak_gb:** 9.35
- **en_lang_ok_pct:** 100.0
- **ru_lang_ok_pct:** 100.0
- **avg_en_length_ratio:** 2.99
- **avg_ru_length_ratio:** 3.101
- **avg_en_repetition:** 0.891
- **avg_ru_repetition:** 0.955
- **avg_numbers_preserved:** 0.838
- **en_no_zh_leak_pct:** 100.0
- **ru_no_zh_leak_pct:** 100.0

### `toys_simplified.csv`

- **chrF:** 38.29
- **BLEU:** 27.10
- **translate_sec:** 106.21
- **rows_per_sec:** 1.88
- **vram_peak_gb:** 7.14
- **en_lang_ok_pct:** 100.0
- **ru_lang_ok_pct:** 100.0
- **avg_en_length_ratio:** 3.233
- **avg_ru_length_ratio:** 3.599
- **avg_en_repetition:** 0.979
- **avg_ru_repetition:** 0.977
- **avg_numbers_preserved:** 0.979
- **en_no_zh_leak_pct:** 100.0
- **ru_no_zh_leak_pct:** 100.0

---

## Общая сводка

- **Всего файлов:** 4
- **Всего строк:** 750
- **Суммарное время:** 762.06 sec
- **Средняя скорость:** 0.98 строк/сек
- **Средний chrF:** 40.36
- **Средний BLEU:** 34.70
- **Средний ru_lang_ok_pct:** 99.75