<< 'MULTILINE-COMMENT'
# Быстрые тесты на окружение... Надо довести до ума при деплое
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')" # Тут всё зависит от CUDA, от GPU, везде Pytorch может показать разное

nvidia-smi # команда для проверки свободного места

MULTILINE-COMMENT
>>