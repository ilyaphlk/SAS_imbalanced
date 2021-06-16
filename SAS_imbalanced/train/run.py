import yaml

import os
import sys
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, average_precision_score


def run_model(config_yaml, fold_path, path_to_model):
    '''
        run a pretrained model
    '''
    with open(config_yaml) as cfg:
        params = yaml.load(cfg, Loader=yaml.FullLoader)

    clf = CatBoostClassifier()
    exp_name = params['exp_name']
    clf.load_model(os.path.join(exp_name, path_to_model))

    df_test = pd.read_csv(os.path.join(fold_path, 'test.csv'))
    y_test = df_test['Class']
    X_test = df_test.drop(columns=['Class'])

    y_pred = clf.predict(X_test)
    print("average_precision, test:", average_precision_score(y_test, y_pred))


if __name__ == "__main__":
    assert len(sys.argv) > 3, 'must specify a path to config, path folds, path to model'
    path_to_config = sys.argv[1]
    path_to_folds = sys.argv[2]
    path_to_model = sys.argv[3]

    run_model(path_to_config, path_to_folds, path_to_model)