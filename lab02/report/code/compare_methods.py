def is_convex_vector(points):
    p = np.asarray(points, float)
    e = np.roll(p, -1, 0) - p
    e2 = np.roll(e, -1, 0)
    cross_z = e[:, 0] * e2[:, 1] - e[:, 1] * e2[:, 0]
    nz = cross_z[cross_z != 0]
    return bool(np.all(nz >= 0) or np.all(nz <= 0))


def is_convex_loop(points):
    p = np.asarray(points, float)
    n = len(p)
    signs = []
    for i in range(n):
        a, b, c = p[i], p[(i + 1) % n], p[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross != 0:
            signs.append(cross > 0)
    return all(signs) or not any(signs)


# Сгенерируем выпуклый многоугольник (точки на окружности)
n = 100000
theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
pts = np.column_stack([np.cos(theta), np.sin(theta)])

for name, f in [("2d", is_convex_vector), ("loop", is_convex_loop)]:
    t = timeit(lambda: f(pts), number=100)
    print(f"{name:6s}: {t * 10:.3f} ms/call")
