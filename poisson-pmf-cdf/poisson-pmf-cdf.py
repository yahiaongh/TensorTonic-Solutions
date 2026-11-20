import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    cdf = 0.0
    pmf = 0.0
    exp_lam = math.exp(-lam)
    p = 1
    f = 1
    for i in range(k + 1):
        if i >= 2:
            f *= i
        t = exp_lam * p / f
        cdf += t
        if i == k:
            pmf = t
        p *= lam
    return {"pmf": pmf, "cdf": cdf}