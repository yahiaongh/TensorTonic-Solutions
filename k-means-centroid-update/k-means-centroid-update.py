def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    d = {}
    for i in range(k):
        d[i] = []
    n = len(points)
    for i in range(n):
        d[assignments[i]].append(points[i])
    r = []
    for a, pts in d.items():
        Sj = len(pts)
        p = [0] * len(points[0])
        if Sj == 0:
            r.append(p)
            continue
        for j in range(Sj):
            for i in range(len(points[0])):
                p[i] += pts[j][i]
        for i in range(len(points[0])):
            p[i] /= Sj
        r.append(p)
    return r