import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    m = np.mean(y_true)
    nom = (y_true - y_pred) * (y_true - y_pred)
    sum_nom = np.sum(nom)
    denom = (y_true - m) * (y_true - m)
    sum_denom = np.sum(denom)
    print(m)
    print(nom, np.sum(nom))
    print(denom, np.sum(denom))
    if(sum_denom == 0):
        return 1.0 if all(y_true == y_pred) else 0.0
    return 1 - float(np.sum(nom) / np.sum(denom))