import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    indices = np.array(list(range(N)))
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    t = N % k 
    R = []
    val_start = 0
    for i in range(k):
        if i < t:
            val_len = N // k + 1
        else:
            val_len = N // k
        val_idx = indices[val_start:val_start + val_len]
        train_idx = np.hstack((indices[:val_start], indices[val_start + val_len:]))
        val_start += val_len
        R.append({"train_idx": train_idx, "val_idx": val_idx})
    return R