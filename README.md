Voice Anti-Spoofing (ASVspoof 2019, LCNN)

Решение задачи обнаружения синтезированной речи (spoof) на основе LCNN с STFT-признаками.

Установка:
  pip install -r requirements.txt

Данные:
  Скачайте ASVspoof 2019 LA с Kaggle и разместите в data/LA/ согласно структуре:
    data/LA/ASVspoof2019_LA_train/flac/
    data/LA/ASVspoof2019_LA_dev/flac/
    data/LA/ASVspoof2019_LA_eval/flac/
    data/LA/ASVspoof2019_LA_cm_protocols/*.txt
  Проверьте пути в src/configs/datasets/asvspoof.yaml.

Обучение:
  python train.py
  Логи и чекпоинты сохраняются в saved/{run_name}/.
  Конфигурация – src/configs/baseline.yaml.

Генерация CSV для сабмита:
  Убедитесь, что в saved/ваш_run_name/ есть model_best.pth.
  Исправьте путь к модели в src/configs/generate_csv.yaml (inferencer.from_pretrained).
  Запустите: python generate_csv.py
  На выходе – your_university_email.csv.
  Переименуйте его в ваша_почта.csv (например, iiivanov.csv) и загрузите в форму.

Проверка пайплайна:
  python test_dataset.py
  Должно вывести "Pipeline OK!".

Результаты (заполните свои):
  EER (%): X.X
  Accuracy: XX.X%

Логи экспериментов – ссылка на WandB/Comet.
