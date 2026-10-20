import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    R = []
    if len(X_test) == 0:
        return np.empty((0, k), dtype=int)
    for x_test in X_test:
        indices = list(range(len(X_train)))
        indices.sort(key=lambda i: np.sum((np.array(x_test) - np.array(X_train[i]))**2))
        print(X_train)
        if k  <= len(X_train):
            R.append(indices[:k])
        else:
            t = indices[:k] + [-1] * (k - len(X_train))
            R.append(t)
    return np.array(R)