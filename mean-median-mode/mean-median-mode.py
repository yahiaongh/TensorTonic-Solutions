from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    n = len(x)
    mean = sum(x) / n
    x = sorted(x)
    median = x[n // 2] if n % 2 == 1 else (x[n // 2] + x[n // 2 - 1]) / 2
    values, counts = np.unique(x, return_counts=True)
    mode = x[np.argmax(counts)]
    return {"mean": mean, "median": float(median), "mode": float(mode)}