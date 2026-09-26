#lab02-3.py
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
