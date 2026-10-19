import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    y_score = np.array(y_score)
    y_true = np.array(y_true)
    ll = margin - y_true * y_score
    l = np.where(ll > 0, ll,0)
    if reduction == "mean":
        return float(np.mean(l))
    else:
        return float(np.sum(l))