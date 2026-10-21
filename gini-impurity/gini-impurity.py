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
    uniq = np.unique(y_left)
    _uniq = np.unique(y_right)
    gsl = 0.0
    for c in uniq:
        cpt = 0
        for y in y_left:
            if c == y:
                cpt += 1
        gsl += (cpt / NL)**2
    gsl = 1 - gsl
    gsr = 0.0
    for c in _uniq:
        cpt = 0
        for y in y_right:
            if c == y:
                cpt += 1
        gsr += (cpt / NR)**2
    gsr = 1 - gsr
    return NL / N * gsl + NR / N * gsr