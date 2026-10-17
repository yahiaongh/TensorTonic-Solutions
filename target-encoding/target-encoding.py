import numpy as np
def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    r = []
    for c in categories:
        t = 0
        l = 0
        for i in range(len(categories)):
            if categories[i] == c:
                t += targets[i]
                l += 1
        r.append(t / l)
    return r