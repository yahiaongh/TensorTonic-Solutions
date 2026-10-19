import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    m = beta1 * m + (1 - beta1) * grad 
    v = beta2 * v + (1 - beta2) * grad * grad
    _m = m / ( 1 - beta1 ** t)
    _v = v / (1 - beta2 ** t)
    param = param - lr * _m / (np.sqrt(_v) + eps)
    return (param, m, v)