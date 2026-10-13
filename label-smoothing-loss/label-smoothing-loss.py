import math
import numpy as np
def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    predictions = np.array(predictions)
    K = len(predictions)
    q = np.array([1 - epsilon + epsilon / K if i == target else epsilon / K for i in range(K)])
    L = -np.sum(np.dot(q, np.log(predictions)))
    return L