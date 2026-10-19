def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    assigns = []
    for point in points:
        min_dist = float('inf')
        j_min = -1
        D = len(point)
        for j in range(len(centroids)):
            dist = 0
            for d in range(D):
                dist += (point[d] - centroids[j][d]) ** 2
            if dist < min_dist:
                min_dist = dist
                j_min = j
        assigns.append(j_min)
    return assigns
                