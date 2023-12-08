from plotting.utils import calc_angulardistance, calc_distance
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score


def diff(y_true, y_pred):
    # return {"x": y_pred - y_true}
    return y_pred - y_true


def rel_diff(y_true, y_pred):
    return (y_pred - y_true) / y_true


def prob2pred(y_pred, threshold=0.5):
    result = np.zeros_like(y_pred)
    result[y_pred > threshold] = 1.

    return result


def rel_resolution_fn(y_true, y_pred, agg_fn=rel_diff):
    return np.std(agg_fn(y_true, y_pred))


def resolution_fn(y_true, y_pred, agg_fn=diff):
    return np.std(agg_fn(y_true, y_pred))


def rel_bias_fn(y_true, y_pred, agg_fn=rel_diff):
    return np.mean(agg_fn(y_true, y_pred))


def bias_fn(y_true, y_pred):
    return np.mean(y_pred - y_true)


def accuracy_fn(y_true, y_pred, threshold=0.5):
    y_pred_ = prob2pred(y_pred)
    return accuracy_score(y_true, y_pred_)


def roccurve_fn(y_true, y_pred):
    return roc_curve(y_true, y_pred)[3]


def fpr_fn(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return fpr


def tpr_fn(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return tpr


def auroc_fn(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)  # switch aquired by sklearn
    except ValueError:  # for case where only one class is present in y_true
        return 0.5


def calc_correlation(x, y):
    '''Caculate correlation coefficient for large vectors'''
    samples = x.shape[0]
    centered_x = x - np.nansum(x, axis=0, keepdims=True) / samples
    centered_y = y - np.nansum(y, axis=0, keepdims=True) / samples
    try:
        cov_xy = 1. / (samples - 1) * np.dot(centered_x.T, centered_y)
        var_x = 1. / (samples - 1) * np.sum(centered_x**2, axis=0)
        var_y = 1. / (samples - 1) * np.sum(centered_y**2, axis=0)
    except ZeroDivisionError:
        cov_xy = np.nan * np.dot(centered_x.T, centered_y)
        var_x = np.nan
        var_y = np.nan
    return cov_xy / np.sqrt(var_x * var_y)


def angulardistance_fn(y_tr, y_pr):
    return 180. / np.pi * np.arccos(np.clip(np.sum(y_tr * y_pr, axis=-1) / np.linalg.norm(y_pr, axis=-1) / np.linalg.norm(y_tr, axis=-1), -1, 1))


def correlation_fn(y_true, y_pred):
    return calc_correlation(y_true, y_pred)


def percentile68_fn(y_true, y_pred, agg_fn=diff):
    return np.percentile(agg_fn(y_true, y_pred), 68)


def angular_resolution_fn(y_true, y_pred):
    return percentile68_fn(y_true, y_pred, calc_angulardistance)


def euclidean_resolution_fn(y_true, y_pred):
    return percentile68_fn(y_true, y_pred, calc_distance)
