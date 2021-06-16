from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def compute_metrics(y_true, y_pred_proba):#,metrics):
    '''
    Compure metrics
    
    Parameters
    ----------
    metrics: dict {metric_name: metric}
    y_true: np.array - true data labels
    y_pred_proba: np.array - prediction probability of label "one"
    '''
#     d = {}
#     y_pred = (y_pred_proba >= 0.5).astype(int)
#     for metric_name, metric in metrics.items():
#         d[metric_name] = metric(y_true, y_pred)
    
#     return d
    return auc_pr(y_true, y_pred)

def plot_runs(metric_list, xticks_range, xlabel, metric_name="AUC-PR"):#, title):
    '''
    Plot metrics for differenr situations
    
    Parameters
    ----------
    metric_list: np.array - list of values for metrics
    xticks_range: np.array - list of values/names for xlabel
    xlabel: string - xlabel name
    metric_name: string - metric name
    '''    
    metric_list0 = metric_list.T
    bar_width = 0.66
    n = len(xticks_range)
    bar0 = plt.bar(np.arange(n) - bar_width/3, metric_list0[0], bar_width/3, label="train")
    bar1 = plt.bar(np.arange(n), metric_list0[1], bar_width/3, label="val")
    bar2 = plt.bar(np.arange(n) + bar_width/3, metric_list0[2], bar_width/3, label="test")
    plt.xticks(np.arange(n), xticks_range)
    plt.xlabel(xlabel)
    plt.ylabel(metric_name)
    #plt.title(title)
    plt.legend()
    plt.show()
# '''
# Пример использования

# res = []
# for param in params:
#     temp = []
#     model.fit(x_train, param)
#     y_pred_proba = model.predict_proba(x_train)
#     temp.append(compute_metrics(y_train,y_pred_proba))
#     y_pred_proba = model.predict_proba(x_val)
#     temp.append(compute_metrics(y_val,y_pred_proba))
#     y_pred_proba = model.predict_proba(x_test)
#     temp.append(compute_metrics(y_test,y_pred_proba))
#     res.append(res)
# plot_runs(np.array(res))
    
# '''

def visualize(path):
    '''
    isualize results from .csv file
    
    Parameters
    ----------
    path: string - path to .csv file
    '''
    
    data = pd.read_csv(path, sep=";")
    ans = data.loc[:, ["train", "val", "test"]].to_numpy()
    names = data["name"]
    plot_runs(ans, names, "Названия методов")