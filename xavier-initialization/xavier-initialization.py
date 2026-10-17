import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    L = math.sqrt(6 / (fan_in + fan_out))
    W = np.array(W)
    return (W * 2 * L - L).tolist()