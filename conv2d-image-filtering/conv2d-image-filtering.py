import numpy as np
def conv2d(image: list, kernel: list, stride: int = 1, padding: int = 0) -> list:
    """
    Returns a two-dimensional list.
    """
    image = np.array(image)
    H, W = image.shape
    X = image
    kernel = np.array(kernel)
    K = kernel
    Kh, Kw = kernel.shape
    p = padding
    s = stride
    X_pad = np.zeros((H + 2 * p, W + 2 * p))
    for i in range(p, H + p):
        for j in range(p, W + p):
            X_pad[i][j] = X[i - p][j - p]
    Hout = (H + 2 * p - Kh) // s + 1
    Wout = (W + 2 * p - Kw) // s + 1
    Y = np.zeros((Hout, Wout))
    for i in range(Hout):
        for j in range(Wout):
            for a in range(Kh):
                for b in range(Kw):
                    Y[i][j] += X_pad[i * s + a][j * s + b] * K[a][b]
    return Y.tolist()