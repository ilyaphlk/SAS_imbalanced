import yaml

import os
import sys
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, average_precision_score


def run_model(config_yaml, test_csv):
    '''
        run a pretrained model
    '''
    with open(config_yaml) as cfg:
        params = yaml.load(cfg, Loader=yaml.FullLoader)

    clf = CatBoostClassifier()
    exp_name = params['exp_name']
    clf.load_model(exp_name)

    df_test = pd.read_csv(test_csv_path)
    y_test = df_test['Class']
    X_test = df_test.drop(columns=['Class'])

    y_pred = clf.predict(X_test)
    print("average_precision, test:", average_precision_score(y_test, y_pred))


if __name__ == "__main__":
    assert len(sys.argv) > 2, 'must specify a path to config, path to test fold'
    path_to_config = sys.argv[1]
    path_to_data = sys.argv[2]