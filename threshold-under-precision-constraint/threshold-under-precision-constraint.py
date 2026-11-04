import numpy as np
def getPrecision(y_true, y_pred):
    tp = 0
    fp = 0
    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
    return tp / (tp + fp)
def getRecall(y_true, y_pred):
    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
    return tp / (tp + fn)
def choose_threshold(y_true: list, scores: list, min_precision: float) -> dict:
    """
    Returns the threshold with the highest recall whose precision is at least min_precision.
    """
    y_true = np.array(y_true)
    scores = np.array(scores)
    m_precision = None
    m_recall = None
    m_t = None
    for t in scores:
        y_pred = np.where(scores >= t, 1, 0)
        precision = getPrecision(y_true, y_pred)
        recall = getRecall(y_true, y_pred)
        if precision >= min_precision:
            if m_recall is None or m_recall < recall:
                m_precision = precision
                m_recall = recall
                m_t = t
            elif m_recall == recall and m_precision < precision:
                m_precision = precision
                m_recall = recall
                m_t = t
    return {"threshold": m_t, "precision": m_precision, "recall": m_recall}