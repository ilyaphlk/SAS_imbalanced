import yaml

from SAS_imbalanced.featurize import featurize
from SAS_imbalanced.resample import resample
#from SAS_imbalanced.utils import compute_metrics
from catboost import CatBoostClassifier, save_model


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

    clf.fit(df_train.drop(columns['Class']), df_train['Class'])

    save_model(clf, out_model_path)



def run_model(model_pickle, test_csv):
    '''
        run a pretrained model
    '''
