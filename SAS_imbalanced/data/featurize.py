from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd


def featurize(df_train, df_dev, df_test, drop_features=None):
    '''
        takes multiple folds, fits transformers on train,
        fit_transforms a train, transforms dev and test
        returns transformed dataframes
    '''

    if drop_features is not None:
        df_train = df_train.drop(columns=drop_features)
        df_dev = df_dev.drop(columns=drop_features)
        df_test = df_test.drop(columns=drop_features)


    cols_list = list(df_train.columns)
    PCA_cols = list(filter(lambda item: 'V' in item, cols_list))
    other_cols = list(set(cols_list) - set(PCA_cols))

    ct = ColumnTransformer([
        ('normalize', StandardScaler(), PCA_cols),
        ('pass', 'passthrough', other_cols)
    ])

    ct.fit(df_train)
    df_train = pd.DataFrame(ct.transform(df_train), index=df_train.index, columns=df_train.columns)
    df_dev = pd.DataFrame(ct.transform(df_dev), index=df_dev.index, columns=df_dev.columns)
    df_test = pd.DataFrame(ct.transform(df_test), index=df_test.index, columns=df_test.columns)

    return df_train, df_dev, df_test