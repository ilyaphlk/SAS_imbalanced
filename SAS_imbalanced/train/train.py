import yaml

from SAS_imbalanced.featurize import featurize
from SAS_imbalanced.resample import resample
#from SAS_imbalanced.utils import compute_metrics
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, average_precision_score


def train_model(config_yaml,
    train_csv_path, dev_csv_path, test_csv_path,
    out_model_path):
    '''
        fit all the stuff
        featurize, resample, fit a classifier
    '''
    with open(config_yaml) as cfg:
        params = yaml.load(cfg, Loader=yaml.FullLoader)
    
    df_train = pd.read_csv(train_csv_path)
    df_dev = pd.read_csv(dev_csv_path)
    df_test = pd.read_csv(test_csv_path)

    df_train, df_dev, df_test = featurize(df_train, df_dev, df_test)

    resampler_class = eval(params.get('resampler', 'None'))
    resampler_kwargs = params.get('resampler_kwargs', {})
    if resampler_class is not None:
        resampler = resampler_class(**resampler_kwargs)
        df_train = resample(df_train, resampler)

    clf_class = eval(params['classifier'])
    clf_kwargs = params['classifier_kwargs']
    clf = clf_class(**clf_kwargs)

    y_train, y_dev = df_train['Class'], df_dev['Class']
    X_train, X_dev = df_train.drop(columns=['Class']), df_dev.drop(columns=['Class'])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    print("average_precision, train:", average_precision_score(y_train, y_pred))

    y_pred = clf.predict(X_dev)
    print("average_precision, train:", average_precision_score(y_dev, y_pred))

    exp_name = params['exp_name']
    clf.save_model(exp_name)



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