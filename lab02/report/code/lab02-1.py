#... code/lab02-1.py
points = np.random.rand((N, 2)) * 10.0 - 5.0

hull = ConvexHull(points)
hull_points = hull.points # Все точки лежащие в выпуклой оболочке

vertex_indices = hull.vertices # Индексы вершин выпуклой оболочки
hull_vertices = points[vertex_indices]

mask = np.zeros(len(points), dtype=bool)
mask[vertex_indices] = True
inner_points = points[~mask] # Точки внутри выпуклой оболочки
#...
plt.scatter(*inner_points.T, alpha=0.5, c='g', label="Random Points")
plt.scatter(*hull_vertices.T, alpha=0.5, c='r', label="Convex Hull Points")
plt.fill(*hull_vertices.T, alpha=0.3, c='lightgreen', label="Convex Hull")
plt.title("Convex Hull of Random Points")
#...
plt.show()
