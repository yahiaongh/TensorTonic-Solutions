def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    for _ in range(steps):
        df_dx0 = 2 * a * x0 + b
        x0 = x0 - lr * df_dx0
    return x0