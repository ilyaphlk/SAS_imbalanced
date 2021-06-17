from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

def transform_time(df):
    sec_in_hour = 60 * 60
    sec_in_day = sec_in_hour * 24
    df['hour'] = (df['Time'] % sec_in_day) // sec_in_hour
    df['day'] = df['Time'] // sec_in_day
    hour_2 = df['hour'] == 2
    hour_11 = df['hour'] == 11
    df['is_anomaly_hour'] = np.logical_or(hour_2.values, hour_11.values)
    df = df.drop(columns=['Time', 'hour', 'day'])

    return df


def featurize(df_train, df_dev, df_test, drop_features=None):
    '''
        takes multiple folds, fits transformers on train,
        fit_transforms a train, transforms dev and test
        returns transformed dataframes
    '''

    df_train = transform_time(df_train)
    df_dev = transform_time(df_dev)
    df_test = transform_time(df_test)

    df_train['Amount'] = np.log1p(df_train['Amount'])
    df_dev['Amount'] = np.log1p(df_dev['Amount'])
    df_test['Amount'] = np.log1p(df_test['Amount'])

    v_drop = [1, 5, 6, 8, 9, 13, 15]
    v_drop.extend(list(range(18, 28+1)))
    drop_features = list(map(lambda x: 'V'+str(x), v_drop))

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

    return df_train.astype('float'), df_dev.astype('float'), df_test.astype('float')