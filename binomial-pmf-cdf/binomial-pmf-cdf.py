from math import factorial
def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    pmf = factorial(n) / (factorial(k) * factorial(n - k)) * pow(p, k) * pow(1 - p, n - k)
    cdf = float(0.0)
    for i in range(0, k):
        cdf = cdf + factorial(n) / (factorial(i) * factorial(n - i)) * pow(p, i) * pow(1 - p, n - i)
    return {"pmf": pmf, "cdf": cdf + pmf}