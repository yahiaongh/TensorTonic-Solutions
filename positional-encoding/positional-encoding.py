import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    return np.array([[np.sin(pos / pow(base, 2 * (k // 2) / d_model)) if k % 2 == 0 else np.cos(pos / pow(base, 2 * ((k - 1) // 2) / d_model)) for k in range(d_model)] for pos in range(seq_len)])