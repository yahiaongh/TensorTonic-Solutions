import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    d = {}
    w = np.array(w, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)
    d["new_v"] = momentum * v + lr * grad
    d["new_w"] = w - d["new_v"]
    return d