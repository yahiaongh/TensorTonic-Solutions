import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    L = math.sqrt(6 / fan_in)
    return (np.array(W) * 2 * L - L).tolist()