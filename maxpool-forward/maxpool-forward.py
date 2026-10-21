import numpy as np
def maxpool_forward(X: list, pool_size: int, stride: int) -> list:
    """
    Returns the maximum value from every pooling window.
    """
    H, W = len(X), len(X[0])
    p = pool_size
    s = stride
    Hout = (H - p) // s + 1
    Wout = (W - p) // s + 1
    out = np.zeros((Hout, Wout)).tolist()
    for i in range(Hout):
        for j in range(Wout):
            o = X[i * s][j * s]
            for a in range(p):
                for b in range(p):
                    o = max(o, X[i * s + a][j * s + b])
            out[i][j] = o
    return out