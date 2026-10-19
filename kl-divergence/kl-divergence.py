import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p, q = np.array(p), np.array(q)
    frac = p / np.where(q > eps, q, eps)
    dkl = np.sum(p * np.log(np.where(frac>eps,frac,eps)))
    return float(dkl)