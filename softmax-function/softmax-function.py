import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.array(x, dtype=float)
    if len(x.shape) == 1:
        m = np.mean(x)
        x = x - m
        sum = np.sum(np.exp(x))
        return np.exp(x ) / sum
    else:
        m = np.mean(x, axis=1)
        for i in range(len(x)):
            x[i] = x[i] - m[i]
        sum = np.sum(np.exp(x), axis=1)
        for i in range(len(x)):
            x[i] = np.exp(x[i]) / sum[i]
        return x