# SAS_imbalanced

Инструкции по запуску:

1. Для работы понадобятся библиотеки catboost, imblearn, yaml. Можно установить командами `pip3 install imbalanced-learn`, `pip install catboost`, и `pip install PyYAML -U`
2. Склонировать репозиторий на машину. Затем установить в качестве модуля: для этого перейдите в директорию `SAS_imbalanced`, откуда запустите `pip install -e .`
3. Перейдите в директорию `SAS_imbalanced/SAS_imbalanced`. Скачайте в неё датасет creditcardfraud: https://www.kaggle.com/mlg-ulb/creditcardfraud
4. Полный процесс обучения состоит из запуска трёх скриптов: 

а) `python3 data/split.py creditcard.csv artifacts`

б) `python3 train/train.py configs/example_config.yaml artifacts artifacts res.csv`

в) `python3 train/run.py configs/example_config.yaml artifacts artifacts`,

где configs/example_config.yaml можно заменить на какой-нибудь другой конфиг-файл из папки configs.

Два замечания:
а) обученная модель сохраняется на диске, поэтому для её применения на каком-нибудь другом датасете достаточно запустить лишь последний скрипт, в качестве предпоследнего аргумента указав путь к папке с тестируемой выборкой.
б) random_seed в catboost работает странно, зафиксировать сид не удалось. При этом из-за размера датасета катбуст иногда вылетает по памяти - чтобы модель обучилась, второй скрипт, возможно, придётся перезапустить пару-тройку раз. (Брались лучшие результаты по 3 успешным запускам.) При применении модели такое не происходит, с точки зрения клиента всё предсказуемо :)
