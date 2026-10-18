import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    x = np.array(x)
    W = np.array(W)
    b = np.array(b)

    N, Cin, H, Wx = x.shape
    Cout, _, Kh, Kw = W.shape
    y = np.zeros((N, Cout, H - Kh + 1, Wx - Kw + 1))
    for n in range(N):
        for c in range(Cout):
            for i in range(H - Kh + 1):
                for j in range(Wx - Kw + 1):
                    t = 0
                    for d in range(Cin):
                        for u in range(Kh):
                            for v in range(Kw):
                                t += x[n, d, i + u, j + v] * W[c, d, u, v]
                    y[n][c][i][j] = t + b[c]
    return y