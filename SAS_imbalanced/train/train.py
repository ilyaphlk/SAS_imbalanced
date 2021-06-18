import yaml

from SAS_imbalanced.data.featurize import featurize
from SAS_imbalanced.data.resample import resample
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, AllKNN, TomekLinks

#from SAS_imbalanced.utils import compute_metrics
import os
import sys
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, average_precision_score
import numpy as np


def train_model(config_yaml, folds_path, out_model_path, res_csv_path):
    '''
        fit all the stuff
        featurize, resample, fit a classifier
    '''
    with open(config_yaml) as cfg:
        params = yaml.load(cfg, Loader=yaml.FullLoader)
    
    df_train = pd.read_csv(os.path.join(folds_path, 'train.csv'))
    df_dev = pd.read_csv(os.path.join(folds_path, 'dev.csv'))
    df_test = pd.read_csv(os.path.join(folds_path, 'test.csv'))

    print("featurizing... ", end='')
    df_train, df_dev, df_test = featurize(df_train, df_dev, df_test)
    print("done.")


    print("resampling... ", end='')
    resampler_class = eval(params.get('resampler', 'None'))
    resampler_kwargs = params.get('resampler_kwargs', {})
    if resampler_class is not None:
        resampler = resampler_class(**resampler_kwargs)
        df_train = resample(df_train, resampler)
    print("done.")

    clf_class = eval(params['classifier'])
    clf_kwargs = params['classifier_kwargs']
    clf = clf_class(**clf_kwargs)

    print("fitting model... ", end='')
    y_train = df_train['Class'].astype('int64')
    y_dev = df_dev['Class'].astype('int64')
    y_test = df_test['Class'].astype('int64')
    X_train = df_train.drop(columns=['Class'])
    X_dev = df_dev.drop(columns=['Class'])
    X_test = df_test.drop(columns=['Class'])
    clf.fit(X_train, y_train)
    print("done.")

    print("predicting... ")
    y_pred = clf.predict(X_train).astype('int64')
    prc_train = average_precision_score(y_train, y_pred.astype('int64'))
    print("average_precision, train:", prc_train)

    y_pred = clf.predict(X_dev).astype('int64')
    #print(y_pred.dtype)
    
    prc_dev = average_precision_score(
        np.hstack([y_dev, np.ones(1)]).astype('int64'),
        np.hstack([y_pred, np.zeros(1)]).astype('int64')
    )
    print("average_precision, validation:", prc_dev)

    y_pred = clf.predict(X_test).astype('int64')
    prc_test = average_precision_score(
        np.hstack([y_test, np.ones(1)]).astype('int64'),
        np.hstack([y_pred, np.zeros(1)]).astype('int64')
    )
    print("average_precision, test:", prc_test)

    exp_name = params['exp_name']

    res = pd.read_csv(res_csv_path) if res_csv_path is not None else pd.DataFrame()
    res[exp_name] = [prc_train, prc_dev, prc_test]
    if res_csv_path is None:
        res_csv_path = 'res.csv'
    res.to_csv(res_csv_path, index=False)

    print("saving model... ", end="")
    
    mpath = os.path.join(out_model_path, exp_name)
    if clf_class == CatBoostClassifier:
        clf.save_model(mpath)
    else:
        with open(mpath, 'wb') as f:
            pickle.dump(f)

    print("done.")


if __name__ == "__main__":
    assert len(sys.argv) > 3, 'must specify a path to config, path to folds'
    path_to_config = sys.argv[1]
    path_to_folds = sys.argv[2]

    model_save_path = "."
    if len(sys.argv) > 3:
        model_save_path = sys.argv[3]

    res_csv_path = None
    if len(sys.argv) > 4:
        res_csv_path = sys.argv[4]

    train_model(path_to_config, path_to_folds, model_save_path, res_csv_path)
