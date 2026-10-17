import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X, replacing missing values 
    with the feature's observed mean or median. Treat empty/all-NaN features as 0.0.
    """
    # 1. Convert to a base float array copy to preserve the original input
    X_arr = np.array(X, dtype=float).copy()
    
    # Identify positions containing NaN
    nan_mask = np.isnan(X_arr)
    if not np.any(nan_mask):
        return X_arr
        
    # Determine the column axis (axis=0 for 2D, axis=0 for 1D)
    axis = 0
    
    # 2. Compute the selected statistic ignoring NaNs
    if strategy == "mean":
        fill_values = np.nanmean(X_arr, axis=axis, keepdims=True)
    elif strategy == "median":
        fill_values = np.nanmedian(X_arr, axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    # 3. Handle features containing only missing values (which output NaN from nanmean/nanmedian)
    fill_values = np.where(np.isnan(fill_values), 0.0, fill_values)
    
    # 4. Broadcast the fill values grid to match the shape of X_arr
    fill_grid = np.broadcast_to(fill_values, X_arr.shape)
    
    # 5. Impute values only where NaNs exist
    X_arr[nan_mask] = fill_grid[nan_mask]
    
    return X_arr
