import numpy as np

def angle_between_3d(v: list, w: list) -> float:
    """
    Returns the angle as a float.
    """
    v = np.array(v)
    w = np.array(w)
    c = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
    if c < -1:
        c = -1
    if c > 1:
        c = 1
    return np.arccos(c)