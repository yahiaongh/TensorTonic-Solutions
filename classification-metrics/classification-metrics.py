import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.union1d(y_true, y_pred)
    N = len(y_true)
    
    # Global Accuracy
    accuracy = np.sum(y_true == y_pred) / N if N > 0 else 0.0
    
    # Calculate TP, FP, FN, and Support for each unique class
    tp, fp, fn, support = {}, {}, {}, {}
    for c in classes:
        tp[c] = int(np.sum((y_true == c) & (y_pred == c)))
        fn[c] = int(np.sum((y_true == c) & (y_pred != c)))
        fp[c] = int(np.sum((y_true != c) & (y_pred == c)))
        support[c] = int(np.sum(y_true == c))

    # 1. Binary Average (Only evaluate the target positive class label)
    if average == "binary":
        tp_c = tp.get(pos_label, 0)
        fp_c = fp.get(pos_label, 0)
        fn_c = fn.get(pos_label, 0)
        
        precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        recall = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    # 2. Micro Average (Global aggregation of true/false metrics)
    elif average == "micro":
        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    # 3. Macro and Weighted Averages (Compute per-class metrics first)
    elif average in ["macro", "weighted"]:
        precisions = {}
        recalls = {}
        f1s = {}
        
        for c in classes:
            p_c = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
            r_c = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
            f_c = (2 * p_c * r_c) / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            
            precisions[c] = p_c
            recalls[c] = r_c
            f1s[c] = f_c
            
        if average == "macro":
            precision = sum(precisions.values()) / len(classes) if len(classes) > 0 else 0.0
            recall = sum(recalls.values()) / len(classes) if len(classes) > 0 else 0.0
            f1 = sum(f1s.values()) / len(classes) if len(classes) > 0 else 0.0
        else:  # weighted
            total_support = sum(support.values())
            if total_support > 0:
                precision = sum(precisions[c] * support[c] for c in classes) / total_support
                recall = sum(recalls[c] * support[c] for c in classes) / total_support
                f1 = sum(f1s[c] * support[c] for c in classes) / total_support
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
                
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6)
    }
