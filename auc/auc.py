import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    M = len(fpr)
    auc = 0.0
    for i in range(M - 1):
        xi, xii, yi, yii = fpr[i], fpr[i + 1], tpr[i], tpr[i + 1]
        auc += (xii - xi) * (yi + yii) / 2
    return auc