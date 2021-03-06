import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

def split(df, strat='chrono', out_dir=None):
    '''
      accepts a dataframe with dataset, returns train, dev and test dataframe folds
      strat: how to split the data into folds; either 'chrono' or 'stratify'
      out_dir: if specified, write csvs to this directory
    '''
    assert strat in ['chrono']

    df_train, df_dev, df_test = None, None, None
    if strat == 'chrono':
        df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
        df_dev, df_test = train_test_split(df_test, test_size=0.5, shuffle=False)

    if out_dir is not None:
        df_train.to_csv(os.path.join(out_dir, 'train.csv'))
        df_dev.to_csv(os.path.join(out_dir, 'dev.csv'))
        df_test.to_csv(os.path.join(out_dir, 'test.csv'))


    print("pos vs all ratio:")
    print(f"train: {df_train[df_train['Class'] == 1].shape[0]}/{df_train['Class'].shape[0]}")
    print(f"dev: {df_dev[df_dev['Class'] == 1].shape[0]}/{df_dev['Class'].shape[0]}")
    print(f"test: {df_test[df_test['Class'] == 1].shape[0]}/{df_test['Class'].shape[0]}")

    return df_train, df_dev, df_test


if __name__ == '__main__':
    assert len(sys.argv) > 1, 'must specify a path to dataset'
    path_to_data = sys.argv[1]

    out_dir = "."
    if len(sys.argv) > 2:
        out_dir = sys.argv[2]

    df = pd.read_csv(path_to_data)
    split(df, out_dir=out_dir)