# SAS_imbalanced

Решение задачи несбалансированной классификации из соревнования https://www.kaggle.com/mlg-ulb/creditcardfraud.

Инструкции по запуску:

1. Для работы понадобятся библиотеки catboost, imblearn, yaml. Можно установить командами `pip3 install imbalanced-learn`, `pip install catboost`, и `pip install PyYAML -U`
2. Склонировать репозиторий на машину. Затем установить в качестве модуля: для этого перейдите в директорию `SAS_imbalanced`, откуда запустите `pip install -e .`
3. Перейдите в директорию `SAS_imbalanced/SAS_imbalanced`. Скачайте в неё датасет creditcardfraud: https://www.kaggle.com/mlg-ulb/creditcardfraud
4. Полный процесс обучения состоит из запуска трёх скриптов: 

а) `python3 data/split.py creditcard.csv artifacts`

б) `python3 train/train.py configs/example_config.yaml artifacts artifacts res.csv`

в) `python3 train/run.py configs/example_config.yaml artifacts artifacts artifacts`,

где configs/example_config.yaml можно заменить на какой-нибудь другой конфиг-файл из папки configs.

Скрипты, по сути, делают следующее: а) разбивает данные на обучающую, валидационную и тестовую выборки; б) Обучает модель, предварительно производя предобработку данных (отбор признаков + овер/андерсемплинг), сохраняет значения метрик в res.csv; в) запускает уже обученную модель на желаемом датасете (желаемый датасет необходимо назвать `test.csv`), записывает предсказания в файл; при наличии настоящих ярлыков - считает AUC-PR.


Замечания:
Обученная модель сохраняется на диске, поэтому для её применения на каком-нибудь другом датасете достаточно запустить лишь последний скрипт, в качестве второго аргумента указав путь к папке с тестируемой выборкой.
