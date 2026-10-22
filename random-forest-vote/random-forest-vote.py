import numpy as np
def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    agg = []
    for j in range(len(predictions[0])):
        votes_j = []
        for i in range(len(predictions)):
            votes_j.append(predictions[i][j])
        uniq, counts = np.unique(votes_j, return_counts=True)
        print(uniq, counts)
        agg.append(np.min(uniq[counts == counts.max()]))
    return agg