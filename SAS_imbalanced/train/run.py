import yaml

import os
import sys
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, average_precision_score


def run_model(config_yaml, fold_path, path_to_model, path_to_ans):
    '''
        run a pretrained model
    '''
    with open(config_yaml) as cfg:
        params = yaml.load(cfg, Loader=yaml.FullLoader)


    clf_class = eval(params['classifier'])
    clf_kwargs = params['classifier_kwargs']
    clf = clf_class(**clf_kwargs)
    exp_name = params['exp_name']

    mpath = os.path.join(exp_name, path_to_model)
    if clf_class == CatBoostClassifier:
        clf.load_model(mpath)
    else:
        with open(mpath, 'rb') as f:
            clf = pickle.load(f)

    df_test = pd.read_csv(os.path.join(fold_path, 'test.csv'))

    y_test = None
    X_test = df_test
    if 'Class' in df_test:
        y_test = df_test['Class']
        X_test = df_test.drop(columns=['Class'])


    y_pred = clf.predict(X_test)
    if y_test is not None:
        print("average_precision, test:", average_precision_score(y_test, y_pred))

    ans_path = os.path.join(path_to_ans, exp_name+"_ans.csv")
    df_ans = pd.DataFrame()
    df_ans['predicted'] = y_pred
    df_ans.to_csv(ans_path, index=False)



if __name__ == "__main__":
    assert len(sys.argv) > 4, 'must specify a path to config, path folds, path to model, path to ans'
    path_to_config = sys.argv[1]
    path_to_folds = sys.argv[2]
    path_to_model = sys.argv[3]
    path_to_ans = sys.argv[4]

    run_model(path_to_config, path_to_folds, path_to_model, path_to_ans)
