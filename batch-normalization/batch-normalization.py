import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    if len(x.shape) == 2:
        N, D = x.shape
        mue = np.mean(x, axis=0)
        var = np.var(x,axis=0)
        _x = (x - mue) / (np.sqrt(var + eps))
        return gamma * _x + beta 
    else:
        N, C, H, W = x.shape
        gamma = gamma.reshape(1, C, 1, 1)
        beta = beta.reshape(1, C, 1, 1)
        mue = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x,axis=(0, 2, 3), keepdims=True)
        _x = (x - mue) / (np.sqrt(var + eps))
        return gamma * _x + beta 