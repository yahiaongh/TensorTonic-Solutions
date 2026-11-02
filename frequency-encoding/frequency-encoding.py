import numpy as np
def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """
    n = len(values)
    values = np.array(values)
    unique, counts = np.unique(values, return_counts=True)
    return [counts[np.where(unique==values[i])[0]] / n for i in range(n)]