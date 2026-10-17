import math
import numpy as np
import scipy.special as sp
def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.array(x, dtype=float)
    ef = x
    
    return x / 2 * (1 + sp.erf(x / math.sqrt(2)))