import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    n = len(seqs)
    if n == 0:
        return np.empty((0, 0), dtype=int)
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    R = []
    for seq in seqs:
        m = len(seq)
        t = seq
        if m > max_len:
            t = t[:max_len]
        else:
            Z = [pad_value] * (max_len - m)
            t = t + Z
        R.append(t)
    return np.array(R)