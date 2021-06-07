from sklearn.model_selection import train_test_split

def split(df, strat='chrono', out_dir=None):
    '''
      accepts a dataframe with dataset, returns train, dev and test dataframe folds
      strat: how to split the data into folds; either 'chrono' or 'stratify'
      out_dir: if specified, write csvs to this directory
    '''
    assert strat in ['chrono']

    if strat == 'chrono':

        df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
        df_dev, df_test = train_test_split(X_test, test_size=0.5, shuffle=False)

        return df_train, df_dev, df_test