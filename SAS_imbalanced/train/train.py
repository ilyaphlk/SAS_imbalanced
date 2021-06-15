import yaml

from SAS_imbalanced.data.featurize import featurize
from SAS_imbalanced.data.resample import resample
#from SAS_imbalanced.utils import compute_metrics
import os
import sys
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, average_precision_score


def train_model(config_yaml, folds_path, out_model_path):
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


    print("resampling... ", end='') #print("done.")
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
    y_train, y_dev = df_train['Class'], df_dev['Class']
    X_train, X_dev = df_train.drop(columns=['Class']), df_dev.drop(columns=['Class'])
    clf.fit(X_train, y_train)
    print("done.")

    print("predicting... ")
    y_pred = clf.predict(X_train)
    print("average_precision, train:", average_precision_score(y_train, y_pred))

    y_pred = clf.predict(X_dev)
    print("average_precision, validation:", average_precision_score(y_dev, y_pred))

    print("saving model... ", end="")
    exp_name = params['exp_name']
    clf.save_model(os.path.join(out_model_path, exp_name))
    print("done.")


if __name__ == "__main__":
    assert len(sys.argv) > 2, 'must specify a path to config, path to folds'
    path_to_config = sys.argv[1]
    path_to_folds = sys.argv[2]

    model_save_path = "."
    if len(sys.argv) > 3:
        model_save_path = sys.argv[3]

    train_model(path_to_config, path_to_folds, model_save_path)