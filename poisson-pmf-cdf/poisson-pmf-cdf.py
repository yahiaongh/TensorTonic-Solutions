import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    cdf = 0
    for i in range(k+1):
        cdf += pow(math.e, -lam) * pow(lam, i) / math.factorial(i)
    pmf = pow(math.e, -lam) * pow(lam, k) / math.factorial(k)
    return {"pmf": pmf, "cdf": cdf}