import numpy as np
def differencing(series: list, order: int) -> list:
    """
    Returns the series after the requested differencing order.
    """
    # Write code here
    # 0 1 3 6 A F
    # t = 1 3 6 A F
    series = np.array(series)
    for _ in range(order):
        t = series
        series = np.insert(series, 0, 0)
        series = series[:-1]
        series = t - series
        series = series[1:]
    return series.tolist()