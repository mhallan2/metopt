import numpy as np
from matplotlib import pyplot as plt


def is_convex(points):
    n = len(points)

    if n < 3:
        return False

    orientation = 0

    for i in range(n):
        edge_a = points[(i + 1) % n] - points[i]
        edge_b = points[(i + 2) % n] - points[(i + 1) % n]

        current_orientation = orientation_of_cross(edge_a, edge_b)

        if current_orientation == 0:
            continue

        if orientation == 0:
            orientation = current_orientation
        elif current_orientation != orientation:
            return False

    return orientation != 0


def orientation_of_cross(vector_a, vector_b):
    x1, y1 = vector_a
    x2, y2 = vector_b
    oriented_length = x1 * y2 - y1 * x2
    return np.sign(oriented_length)

if __name__ == "__main__":
    # Одна точка лежит на линии между двумя другими точками, но не нарушает выпуклость
    # Повторяющиеся точки не нарушают выпуклость
    oriented_points_convex = np.array([[0, 0], [1, 0], [1, 1], [0.5, 1], [0, 1], [0, 0]])
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
