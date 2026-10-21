import numpy as np

def silhouette_score(X: list, labels: list[int]) -> float:
    """
    Returns the mean Silhouette Score as a Python float.
    """
    X = np.array(X)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    # Pre-group points by cluster using a dictionary to avoid inhomogeneous shape errors
    clusters = {k: X[labels == k] for k in unique_labels}
    
    s = []
    for i in range(len(X)):
        x_i = X[i]
        label_i = labels[i]
        
        # 1. Compute a(i)
        same_cluster = clusters[label_i]
        # Calculate distances to all points in the same cluster
        dists_a = np.sqrt(np.sum((same_cluster - x_i) ** 2, axis=1))
        # Sum distances and divide by (n-1) because distance to itself is 0
        a_i = np.sum(dists_a) / (len(same_cluster) - 1)
        
        # 2. Compute b(i)
        b_i = float('inf')
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_cluster = clusters[other_label]
            dists_b = np.sqrt(np.sum((other_cluster - x_i) ** 2, axis=1))
            mean_dist = np.mean(dists_b)
            if mean_dist < b_i:
                b_i = mean_dist
                
        # 3. Compute s(i)
        s_i = (b_i - a_i) / max(a_i, b_i)
        s.append(s_i)
        
    return float(np.mean(s))