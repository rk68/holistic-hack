import numpy as np

def average_odds_diff(
        group_1_pred: np.ndarray,
        group_1_actuals: np.ndarray,
        group_2_pred: np.ndarray,
        group_2_actuals: np.ndarray
) -> float:
    """ average odds difference (AOD) between two groups """
    delta_tnr = tnr(group_1_pred, group_1_actuals) - tnr(group_2_pred, group_2_actuals)
    delta_tpr = tpr(group_1_pred, group_1_actuals) - tpr(group_2_pred, group_2_actuals)

    return 0.5 * (np.abs(delta_tnr) + np.abs(delta_tpr))

def treatment_equality(
        group_1_pred: np.ndarray,
        group_1_actuals: np.ndarray,
        group_2_pred: np.ndarray,
        group_2_actuals: np.ndarray
) -> float:
    """ treatment equality deviance between two groups """
    fpr_deviance = np.abs(fpr(group_1_pred, group_1_actuals) - fpr(group_2_pred, group_2_actuals))
    fnr_deviance = np.abs(fnr(group_1_pred, group_1_actuals) - fnr(group_2_pred, group_2_actuals))

    return 0.5 * (fpr_deviance + fnr_deviance)
    

def tpr(pred: np.ndarray, actual: np.ndarray) -> float:
    """ true positive rate """
    # number of true positives over total positives
    tp = np.count_nonzero((pred == 1) & (actual == 1)) # true pos
    fn = np.count_nonzero((pred == 0) & (actual == 1)) # false neg
    return tp/(tp+fn)

def tnr(pred: np.ndarray, actual: np.ndarray) -> float:
    """ true positive rate """
    tn = np.count_nonzero((pred == 0) & (actual == 0)) # true neg
    fp = np.count_nonzero((pred == 1) & (actual == 0)) # false pos
    return tn/(tn+fp)

def fpr(pred: np.ndarray, actual: np.ndarray) -> float:
    """ false positive rate """
    return 1 - tpr(pred, actual)

def fnr(pred: np.ndarray, actual: np.ndarray) -> float:
    """" false negative rate """
    return 1 - tnr(pred, actual)