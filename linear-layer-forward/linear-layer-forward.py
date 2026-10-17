def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    X = np.array(X, dtype=float)
    W = np.array(W, dtype=float)
    b = np.array(b, dtype=float)
    Y = X @ W + b
    return Y.tolist()