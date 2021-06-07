from sklearn.metrics import average_precision_score


def auc_pr(target, pred):
    '''
    Matthews correlation coefficient
    
    Parameters
    ----------
    target: np.array - true data labels
    pred: np.array - prediction probability of label "one"
    '''
    return average_precision_score(target, pred)

def mcc(target, pred):
    '''
    Matthews correlation coefficient
    
    Parameters
    ----------
    target: np.array - true data labels
    pred: np.array - prediction data labels
    '''
    true_p = target == 1
    pred_p = pred == 1
    n = len(true)
    tp = (true_p & pred_p).sum()
    tn = (~true_p & ~pred_p).sum()
    fp = (~true_p & pred_p).sum()
    fn = (true_p & ~pred_p).sum()
    down = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if (down == 0):
        return 0
    return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

def kcc(target, pred):
    '''
    Kappa Cohen coefficient
    
    Parameters
    ----------
    target: np.array - true data labels
    pred: np.array - prediction data labels
    '''
    n = len(true)
    p_0 = (true == pred).sum() / n
    p_e = (true == 1).sum() * (pred == 1).sum() / (n * n) + (true == 0).sum() * (pred == 0).sum() / (n * n)
    if p_e == 1:
        return 0
    return (p_0 - p_e) / (1 - p_e)
