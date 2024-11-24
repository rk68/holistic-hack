import numpy as np

def abs_average_odds_diff(
    group_1: np.ndarray,
    group_2: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> float:
    """ average odds difference (AOD) between two groups """
    group_1, group_2 = map(lambda x: x.astype(bool), (group_1, group_2))
    group_1_pred, group_1_actuals = y_pred[group_1], y_true[group_1]
    group_2_pred, group_2_actuals = y_pred[group_2], y_true[group_2]
    delta_tnr = tnr(group_1_pred, group_1_actuals) - tnr(group_2_pred, group_2_actuals)
    delta_tpr = tpr(group_1_pred, group_1_actuals) - tpr(group_2_pred, group_2_actuals)

    return 0.5 * (np.abs(delta_tnr) + np.abs(delta_tpr))

def abs_treatment_equality_deviance(
    group_1: np.ndarray,
    group_2: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> float:
    """ treatment equality deviance between two groups """
    group_1, group_2 = map(lambda x: x.astype(bool), (group_1, group_2))
    group_1_pred, group_1_actuals = y_pred[group_1], y_true[group_1]
    group_2_pred, group_2_actuals = y_pred[group_2], y_true[group_2]
    fpr_deviance = np.abs(fpr(group_1_pred, group_1_actuals) - fpr(group_2_pred, group_2_actuals))
    fnr_deviance = np.abs(fnr(group_1_pred, group_1_actuals) - fnr(group_2_pred, group_2_actuals))

    return 0.5 * (fpr_deviance + fnr_deviance)
    

def tpr(pred: np.ndarray, actual: np.ndarray) -> float:
    """ true positive rate """
    # number of true positives over total positives
    tp = np.count_nonzero((pred == 1) & (actual == 1)) # true pos
    fn = np.count_nonzero((pred == 0) & (actual == 1)) # false neg
    
    tpr = 0 if tp == 0 and fn == 0 else tp/(tp+fn)
    # print("tpr: ", tpr)
    return(tpr)

def tnr(pred: np.ndarray, actual: np.ndarray) -> float:
    """ true positive rate """
    tn = np.count_nonzero((pred == 0) & (actual == 0)) # true neg
    fp = np.count_nonzero((pred == 1) & (actual == 0)) # false pos

    tnr = 0 if tn == 0 and fp == 0 else tn/(tn+fp)
    # print("tnr :", tnr)
    return tnr

def fpr(pred: np.ndarray, actual: np.ndarray) -> float:
    """ false positive rate """
    return 1 - tnr(pred, actual)

def fnr(pred: np.ndarray, actual: np.ndarray) -> float:
    """" false negative rate """
    return 1 - tpr(pred, actual)