import numpy as np

def gini(S):
    """
    Calculates the Gini impurity of a list/array of class labels.
    """
    if len(S) == 0:
        return 0.0
    _, counts = np.unique(S, return_counts=True)
    N = len(S)
    p_k = (counts / N) ** 2
    return 1.0 - float(np.sum(p_k))

def decision_tree_split(X: list, y: list) -> list:
    """
    Returns the best feature index and threshold for splitting the data.
    """
    X = np.array(X)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    parent_gini = gini(y)
    
    best_gain = -float('inf')
    best_feature = -1
    best_threshold = -1.0
    
    for f in range(n_features):
        feature_vals = X[:, f]
        
        # np.unique automatically sorts the values
        unique_vals = np.unique(feature_vals)
        
        if len(unique_vals) <= 1:
            continue
            
        for i in range(len(unique_vals) - 1):
            threshold = (unique_vals[i] + unique_vals[i+1]) / 2.0
            
            left_mask = feature_vals <= threshold
            right_mask = ~left_mask
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            n_left = len(y_left)
            n_right = len(y_right)
            
            # Prevent empty splits
            if n_left == 0 or n_right == 0:
                continue
            
            split_gini = (n_left / n_samples) * gini(y_left) + (n_right / n_samples) * gini(y_right)
            
            gain = parent_gini - split_gini
            
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = threshold
                
    return [best_feature, float(best_threshold)]