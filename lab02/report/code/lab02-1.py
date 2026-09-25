#... code/lab02-1.py
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

#...
points = np.random.rand((N, 2)) * 10.0 - 5.0

hull = ConvexHull(points)
hull_points = hull.points

vertex_indices = hull.vertices
hull_vertices = points[vertex_indices]

mask = np.zeros(len(points), dtype=bool)
mask[vertex_indices] = True

inner_points = points[~mask]

plt.figure(figsize=(8, 6))
plt.scatter(inner_points[:, 0], inner_points[:, 1], label="Random Points")
plt.scatter(hull_vertices[:, 0], hull_vertices[:, 1], label="Convex Hull Points")
plt.fill(points[vertex_indices][:, 0], points[vertex_indices][:, 1], alpha=0.3, label="Convex Hull")
plt.title("Convex Hull of Random Points")
#...
plt.show()
