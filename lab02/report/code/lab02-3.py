#lab02-3.py
def is_convex(points):
    p = np.asarray(points, float)
    e1 = np.roll(p, -1, 0) - p  # p[i+1] - p[i] = AB
    e2 = np.roll(e1, -1, 0)     # p[i+2] - p[i+1] = BC
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] # AB_x * BC_y - AB_y * BC_x
    nz = cross[cross != 0]
    # True, когда все обходы положительные, либо отрицательные,
    # либо точки коллинеарны, т.е. nz.size == 0
    return np.all(nz >= 0) or np.all(nz <= 0)
