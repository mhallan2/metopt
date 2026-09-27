import numpy as np
from matplotlib import pyplot as plt


def is_convex(points):
    p = np.asarray(points, float)
    curr_edge = np.roll(p, -1, 0) - p
    next_edge = np.roll(curr_edge, -1, 0)
    cross_z = curr_edge[:, 0] * next_edge[:, 1] - curr_edge[:, 1] * next_edge[:, 0]
    nz = cross_z[cross_z != 0]
    return np.all(nz >= 0) or np.all(nz <= 0)


if __name__ == "__main__":
    # Одна точка лежит на линии между двумя другими точками, но не нарушает выпуклость
    # Повторяющиеся точки не нарушают выпуклость
    oriented_points_convex = np.array(
        [[0, 0], [1, 0], [1, 1], [0.5, 1], [0, 1], [0, 0]]
    )
    oriented_points_convex[:, 1] += 3.0
    oriented_points_convex[:, 0] -= 3.0
    oriented_points_non_convex = np.array(
        [[0, 0], [1, 0], [0.5, 0.5], [1, 1], [0, 1], [0, 0]]
    )
    star_points = np.array(
        [
            [0.000, 5.000],
            [1.176, 1.618],
            [4.755, 1.545],
            [1.902, -0.618],
            [2.939, -4.045],
            [0.000, -2.000],
            [-2.939, -4.045],
            [-1.902, -0.618],
            [-4.755, 1.545],
            [-1.176, 1.618],
            [0.000, 5.000],
        ]
    )
    # colinearity_points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    # print(is_convex(colinearity_points))  # True
    print(is_convex(oriented_points_convex))  # True
    print(is_convex(oriented_points_non_convex))  # False
    print(is_convex(star_points))  # False
    plt.plot(*oriented_points_convex.T, marker="o", color="b", label="Convex Polygon")
    plt.plot(
        *oriented_points_non_convex.T, marker="o", color="r", label="Non-Convex Polygon"
    )
    plt.plot(*star_points.T, marker="o", color="g", label="Star Shape")
    plt.legend()
    plt.title("Polygon Shapes")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.show()
