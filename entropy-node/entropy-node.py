import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    if len(y) == 0:
        return 0.0
    hs = 0.0
    C = np.max(y)
    
    for i in range(C + 1):
        pi = y.count(i) / len(y)
        if pi > 0:
            hs += -pi * np.log2(pi)
    return hs