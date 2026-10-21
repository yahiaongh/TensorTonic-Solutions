import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    y_left = np.array(y_left)
    y_right = np.array(y_right)
    NL, NR = len(y_left), len(y_right)
    if NL + NR == 0:
        return 0.0
    N = NL + NR 
    uniq = np.unique_counts(y_left)
    _uniq = np.unique_counts(y_right)
    gsl = 1 - np.sum((uniq.counts / NL)**2)
    gsr = 1 - np.sum((_uniq.counts / NR)**2)
    return NL / N * gsl + NR / N * gsr