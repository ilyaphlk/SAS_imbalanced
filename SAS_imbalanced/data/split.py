def split(df, strat='chrono', out_dir=None):
    '''
      accepts a dataframe with dataset, returns train, dev and test dataframe folds
      strat: how to split the data into folds; either 'chrono' or 'stratify'
      out_dir: if specified, write csvs to this directory
    '''
    raise NotImplementedError