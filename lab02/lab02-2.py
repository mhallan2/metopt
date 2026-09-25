import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


def in_A(x, y):
    return x * x + y * y <= 9.0


def in_B(x, y):
    return (x - 2.0) ** 2 + (y - 2.0) ** 2 <= 6.25


x_min, x_max = -5.0, 5.0
y_min, y_max = -5.0, 5.0
step = 0.01

Y, X = np.mgrid[y_min:y_max:step, x_min:x_max:step]

mask_A = in_A(X, Y)
mask_B = in_B(X, Y)
mask_AB = mask_A & mask_B

img = np.zeros(X.shape, dtype=int)
img[mask_A & ~mask_B] = 1
img[mask_B & ~mask_A] = 2
img[mask_A & mask_B] = 3

legend_elements = [
    Patch(facecolor="blue", label="A"),
    Patch(facecolor="red", label="B"),
    Patch(facecolor="purple", label="A ∩ B"),
]
cmap = ListedColormap([(0, 0, 0, 0), "blue", "red", "purple"])

if __name__ == "__main__":
    plt.figure(figsize=(8, 8))
    plt.imshow(img, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap=cmap)
    plt.title("Sets A, B and A ∩ B")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend(handles=legend_elements, loc="upper right")
    plt.savefig("sets_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
