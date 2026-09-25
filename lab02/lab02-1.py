import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull


N = 20
points = np.random.rand(N, 2) * 10.0 - 5.0

hull = ConvexHull(points)
hull_points = hull.points

vertex_indices = hull.vertices
hull_vertices = points[vertex_indices]

mask = np.zeros(len(points), dtype=bool)
mask[vertex_indices] = True

inner_points = points[~mask]

plt.figure(figsize=(8, 6))
plt.scatter(
    inner_points[:, 0],
    inner_points[:, 1],
    s=10,
    marker="o",
    label="Random Points",
    c="blue",
    alpha=0.5,
)
plt.scatter(
    hull_vertices[:, 0],
    hull_vertices[:, 1],
    s=10,
    marker="o",
    label="Convex Hull Points",
    c="red",
)
plt.fill(
    points[vertex_indices][:, 0],
    points[vertex_indices][:, 1],
    color="lightgreen",
    alpha=0.4,
)
plt.title("Convex Hull of Random Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend()
plt.savefig("convex_hull_plot.png", dpi=300, bbox_inches="tight")
plt.show()
