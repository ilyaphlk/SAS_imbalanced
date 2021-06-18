from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, AllKNN, TomekLinks

def resample(df_train, resampler, out_dir=None):
    '''
        takes a dataframe with train fold and a resampler object (?)
        returns a dataframe with updated train
    '''
    X_train, y_train = df_train.drop(columns=['Class']), df_train['Class'].astype('int64')
    X_train, y_train = resampler.fit_resample(X_train, y_train)

    df_train = X_train
    df_train['Class'] = y_train

    return df_train