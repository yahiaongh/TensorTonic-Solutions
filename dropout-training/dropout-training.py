import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.array(x, dtype=float)
    r = rng.random(x.shape) if rng else np.random.random(x.shape)
    dp = np.zeros(x.shape)
    x[p <= r] = x[p <= r] * 1 / (1 - p)
    x[p > r] = 0.0
    dp[p > r] = 0.0
    dp[p <= r] = 1 / (1 - p)
    return (x, dp)