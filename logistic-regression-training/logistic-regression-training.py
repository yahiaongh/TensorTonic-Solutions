import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # initialize W and b randomly
    N = X.shape[0]
    W, b = np.zeros(X.shape[1]), 0.0
    # loop steps times:
    for _ in range(steps):
    #  forward pass
        z = np.dot(X, W) + b 
        ŷ = _sigmoid(z)
    #  calculate gradient
        dL_dW =  1 / N * X.T.dot(ŷ - y)
        dL_db =  1 / N * np.sum(ŷ - y)
    #  update W, b
        W = W - lr * dL_dW
        b = b - lr * dL_db
    #  goto loop 
    return (W, b)