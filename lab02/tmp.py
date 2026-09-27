import numpy as np
from timeit import timeit


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


def bench(f, pts, number=None):
    if number is None:
        # грубая автоподстройка
        t = timeit(lambda: f(pts), number=1)
        number = max(1, int(0.2 / max(t, 1e-6)))
    t = timeit(lambda: f(pts), number=number)
    return t / number * 1e6  # µs/call


for n in [10, 100, 1000, 10000, 100000]:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    print(
        f"vector={bench(is_convex_vector, pts):8.1f} µs/call  loop={bench(is_convex_loop, pts):8.1f} µs/call"
    )
