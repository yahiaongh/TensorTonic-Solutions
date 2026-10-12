def popularity_ranking(items: list, min_votes: int, global_mean: float) -> list:
    """
    Returns the weighted rating for every item.
    """
    return [items[i][0] * items[i][1] / (items[i][1] + min_votes) + global_mean * min_votes / (min_votes + items[i][1]) for i in range(len(items))]