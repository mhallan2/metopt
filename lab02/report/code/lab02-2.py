#... code/lab02-2.py
def in_A(x, y):
    return x * x + y * y <= 9.0

def in_B(x, y):
    return (x - 2.0) ** 2 + (y - 1.0) ** 2 <= 6.25
# Удобное задание сетки для визуализации
Y, X = np.mgrid[y_min:y_max:step, x_min:x_max:step]

mask_A, mask_B = in_A(X, Y), in_B(X, Y)
img = mask_A + 2 * mask_B # A = 1, B = 2, A ∩ B = 3
#...
plt.figure(figsize=(8, 8))
plt.imshow(img, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap=cmap)
plt.title("Sets A, B and A ∩ B")
#...
plt.show()
