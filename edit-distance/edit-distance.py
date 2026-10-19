import numpy as np
def edit_distance(s1: str, s2: str) -> int:
    """
    Returns the minimum number of edits between the strings.
    """
    m = len(s1)
    n = len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for j in range(dp.shape[1]):
        dp[0][j] = j 
    for i in range(dp.shape[0]):
        dp[i][0] = i 
    for i in range(1, dp.shape[0]):
        for j in range(1, dp.shape[1]):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return int(dp[m][n])