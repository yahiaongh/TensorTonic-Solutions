import numpy as np
def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    n_samples = len(y_true)
    max_y_true = np.max(y_true)
    max_y_pred = np.max(y_pred)
    n_classes = max(max_y_pred, max_y_true)
    f1_m = 0.0
    tp = 0 # 3
    fp = 0 # 
    fn = 0
    for c in range(n_classes + 1):
        for i in range(n_samples):
            if y_true[i] == c and y_pred[i] == c:
                tp += 1
            elif y_true[i] == c and y_pred[i] != c:
                fn += 1
            elif y_true[i] != c and y_pred[i] == c:
                fp += 1 
    return 2 * tp / (2 * tp + fp + fn)