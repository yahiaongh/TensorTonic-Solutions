import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    p, y = np.array(p), np.array(y)
    d = (2 * np.sum(p*y) + eps) / (np.sum(p) + np.sum(y) + eps)
    return 1 - float(d)