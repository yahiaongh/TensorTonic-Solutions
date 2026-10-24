import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    rng = np.random.default_rng(seed)
    X = np.array(X)
    y = np.array(y)
    N = X.shape[0]
    
    # rng.permutation handles integers directly, replacing np.array(list(range(N)))
    indices = rng.permutation(N) 
    
    n_batches = N // batch_size
    if N % batch_size > 0 and not drop_last:
        n_batches += 1
        
    t = 0
    for _ in range(n_batches):
        X_b = X[indices[t:t+batch_size]]
        y_b = y[indices[t:t+batch_size]]
        
        # Yield the tuple instead of appending to a list
        yield (X_b, y_b)
        
        t += batch_size