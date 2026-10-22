import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    y = np.array(y)
    split_mask = np.array(split_mask)
    Y_L = y[split_mask == True]
    Y_R = y[split_mask == False]
    NL = len(Y_L)
    NR = len(Y_R)
    N = NL + NR 
    return H(y) - NL / N * H(Y_L) - NR / N * H(Y_R)
def H(y):
    unique, counts = np.unique_counts(y)
    pc = counts / len(y)
    log2counts = np.log2(pc)
    return - pc.dot(log2counts)
    