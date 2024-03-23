import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.metrics import make_scorer, get_scorer, log_loss

def ks_stat(y, yhat):
    return ks_2samp(yhat[y==1], yhat[y!=1]).statistic

def binary_clf_metric_at_thresholds(y_true, pred_prob) -> pd.DataFrame:
    """Binary classification metrics at each threshold

    :param y_true: ground truth target array (binary)
    :param pred_prob: predicted probabilities array (float between 0-1)
    :return: DataFrame of columns [threshold, tp, fp, tn, fn, tpr, fpr, precision, recall]
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, pred_prob)
    num_p = y_true.sum()
    num_n = (1 - y_true).sum()

    t = pd.DataFrame({
        'threshold': thresholds,
        'tp': tps,
        'fp': fps,
        'tn': num_n - fps,
        'fn': num_p - tps
    })
    t['tpr'] = t['tp'] / (t['tp'] + t['fn'])
    t['fpr'] = t['fp'] / (t['fp'] + t['tn'])
    t['precision'] = t['tp'] / (t['tp'] + t['fp'])
    t['recall'] = t['tpr']
    return t