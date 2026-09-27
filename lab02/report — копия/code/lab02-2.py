#... code/lab02-2.py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


def in_A(x, y):
    return x * x + y * y <= 9.0

def in_B(x, y):
    return (x - 2.0) ** 2 + (y - 1.0) ** 2 <= 6.25

x_min, x_max = -5.0, 5.0
y_min, y_max = -5.0, 5.0
step = 0.01
Y, X = np.mgrid[y_min:y_max:step, x_min:x_max:step]

mask_A, mask_B = in_A(X, Y), in_B(X, Y)

img = np.zeros(X.shape, dtype=int)
img[mask_A & ~mask_B], img[mask_B & ~mask_A], img[mask_A & mask_B] = 1, 2, 3

#... Настройка цветов и легенды

plt.figure(figsize=(8, 8))
plt.imshow(img, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap=cmap)
plt.title("Sets A, B and A ∩ B")
plt.grid()
plt.legend(handles=legend_elements, loc="upper right")
plt.show()
