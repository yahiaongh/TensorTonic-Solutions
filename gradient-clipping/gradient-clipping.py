import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    g = np.array(g)
    g_norm = np.linalg.norm(g)
    g_clipped = np.where(g_norm <= max_norm, g, g * max_norm / g_norm)
    return g_clipped