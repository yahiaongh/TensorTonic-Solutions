import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    p_norm = np.linalg.norm(a) * np.linalg.norm(b)
    if p_norm == 0.0:
        return float(0.0)
    return float(np.dot(a, b).item() / p_norm)