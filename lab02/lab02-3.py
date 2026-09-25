import numpy as np


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


oriented_points_convex = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
oriented_points_non_convex = np.array([[0, 0], [1, 0], [0.5, 0.5], [1, 1], [0, 1]])
print(is_convex(oriented_points_convex))  # True
print(is_convex(oriented_points_non_convex))  # False
