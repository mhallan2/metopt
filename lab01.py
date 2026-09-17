import numpy as np
import matplotlib.pyplot as plt


a, b = 0.0, 3.0
eps = 1e-3
delta = 1e-4
phi = (1.0 + np.sqrt(5.0)) / 2.0

N_TEST_POINTS = 1000
x_grid = np.linspace(a, b, N_TEST_POINTS)


def f1(x):
    return (x - 2) ** 2 + 1


def f2(x):
    return x**2 + 2 * np.sin(3 * x)


def f3(x):
    return x**2 + 5 * np.sin(2 * x) + 2 * np.cos(3 * x)


def visualize(function, x_grid, values, x_min, x_neighbors):
    plt.plot(x_grid, values, label="f(x)", color="blue")
    plt.scatter(x_min, function(x_min), color="red", label="Минимум")
    plt.scatter(
        x_neighbors, function(x_neighbors), color="green", label="Границы интервала"
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()


def passive(function):
    N = 2 * int((b - a) / eps) - 1

    step = (b - a) / (N + 1)
    x_grid_passive = np.arange(a + step, b, step)

    values = function(x_grid_passive)

    min_idx = np.argmin(values)
    x_min = x_grid_passive[min_idx]

    left = x_grid_passive[min_idx - 1]
    right = x_grid_passive[min_idx + 1]

    x_neighbors = np.array([left, right])

    print(f"Минимум (пассивный) {x_min}")

    visualize(
        function,
        x_grid_passive,
        values,
        x_min,
        x_neighbors,
    )


def dichotomy(function):
    left, right = a, b

    n_evaluations = 0
    n_history = []
    length_history = []

    while right - left > eps:
        middle = 0.5 * (left + right)

        x1 = middle - delta
        x2 = middle + delta

        f1_value = function(x1)
        f2_value = function(x2)

        n_evaluations += 2

        if f1_value < f2_value:
            right = x2
        else:
            left = x1

        n_history.append(n_evaluations)
        length_history.append(right - left)

    x_min = 0.5 * (left + right)

    print(f"Минимум (метод дихотомии) {x_min}")
    print(f"Количество вычислений функции (метод дихотомии) {n_evaluations}")

    visualize(
        function,
        x_grid,
        function(x_grid),
        x_min,
        np.array([left, right]),
    )

    return n_history, length_history


def fib(n):
    first = 0
    second = 1

    for _ in range(n):
        first, second = second, first + second

    return first


def fibonacci_numbers_until(value):
    first = 1
    second = 1
    n = 2

    while second < value:
        first, second = second, first + second
        n += 1

    return n


def fibonacci(function):
    left, right = 0.0, 1.0

    normalized_function = lambda x: function(a + (b - a) * x)

    normalized_delta = delta / (b - a)

    fib_need = (b - a) / (eps - delta)
    n = fibonacci_numbers_until(fib_need)

    x1 = left + (right - left) * fib(n - 2) / fib(n)

    x2 = left + (right - left) * fib(n - 1) / fib(n)

    f1_value = normalized_function(x1)
    f2_value = normalized_function(x2)

    n_evaluations = 2

    n_history = []
    length_history = []

    while n > 3:
        if f1_value < f2_value:
            right = x2

            x2 = x1
            f2_value = f1_value

            n -= 1

            n_history.append(n_evaluations)
            length_history.append((b - a) * (right - left))

            if n == 3:
                break

            x1 = left + (right - left) * fib(n - 2) / fib(n)

            f1_value = normalized_function(x1)
            n_evaluations += 1

        else:
            left = x1

            x1 = x2
            f1_value = f2_value

            n -= 1

            n_history.append(n_evaluations)
            length_history.append((b - a) * (right - left))

            if n == 3:
                break

            x2 = left + (right - left) * fib(n - 1) / fib(n)

            f2_value = normalized_function(x2)
            n_evaluations += 1

    x2 = x1 + normalized_delta
    f2_value = normalized_function(x2)
    n_evaluations += 1

    if f1_value < f2_value:
        right = x2
    else:
        left = x1

    n_history.append(n_evaluations)
    length_history.append((b - a) * (right - left))

    x_min = a + (b - a) * (left + right) / 2.0
    neighbors = np.array([a + (b - a) * left, a + (b - a) * right])

    print(f"Минимум (метод Фибоначчи) {x_min}")
    print(f"Количество вычислений функции (метод Фибоначчи) {n_evaluations}")

    visualize(
        function,
        x_grid,
        function(x_grid),
        x_min,
        neighbors,
    )

    return n_history, length_history


def golden_ratio(function):
    left, right = 0.0, 1.0

    normalized_function = lambda x: function(a + (b - a) * x)

    normalized_eps = eps / (b - a)

    x1 = left + (right - left) / phi**2

    x2 = left + (right - left) / phi

    f1_value = normalized_function(x1)
    f2_value = normalized_function(x2)

    n_evaluations = 2

    n_history = []
    length_history = []

    while True:
        if f1_value < f2_value:
            right = x2

            x2 = x1
            f2_value = f1_value

            n_history.append(n_evaluations)
            length_history.append((b - a) * (right - left))

            if right - left <= normalized_eps:
                break

            x1 = left + (right - left) / phi**2

            f1_value = normalized_function(x1)
            n_evaluations += 1

        else:
            left = x1

            x1 = x2
            f1_value = f2_value

            n_history.append(n_evaluations)
            length_history.append((b - a) * (right - left))

            if right - left <= normalized_eps:
                break

            x2 = left + (right - left) / phi

            f2_value = normalized_function(x2)
            n_evaluations += 1

    x_min = a + (b - a) * (left + right) / 2.0
    neighbors = np.array([a + (b - a) * left, a + (b - a) * right])

    print(f"Минимум (метод золотого сечения) {x_min}")
    print(f"Количество вычислений функции (метод золотого сечения) {n_evaluations}")

    visualize(
        function,
        x_grid,
        function(x_grid),
        x_min,
        neighbors,
    )

    return n_history, length_history


def bracketing(function, interval):
    """Возвращает тройку (a, b, c), где a < b < c и f(b) < f(a), f(b) < f(c)"""
    left, right = interval
    h = 0.8

    x0 = left
    x1 = x0 + h

    f0 = function(x0)
    f1 = function(x1)

    found = False

    while True:
        x2 = x1 + h

        if x2 > right:
            break

        f2 = function(x2)

        if f1 < f0 and f1 < f2:
            x_min = x1
            left, right = x0, x2
            found = True
            visualize(
                function,
                x_grid,
                function(x_grid),
                x_min,
                np.array([left, right]),
            )
            return (left, x_min, right)

        x0, f0 = x1, f1
        x1, f1 = x2, f2
    print("Интервал локализации не найден") if not found else None


def parabola(function):
    x0, x1, x2 = bracketing(function, (a, b))

    y0 = function(x0)
    y1 = function(x1)
    y2 = function(x2)

    n_evaluations = 3
    n_history = []
    length_history = []

    while x2 - x0 > eps:
        denominator = (y1 - y2) * (x1 - x0) - (y1 - y0) * (x1 - x2)

        if denominator == 0:
            print("Невозможно построить параболу")
            break

        extremum = (
            x1
            - 0.5
            * ((x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0))
            / denominator
        )

        y_extremum = function(extremum)
        n_evaluations += 1

        # extremum находится справа от x1
        if x1 < extremum < x2:
            if y_extremum < y1:
                x0, y0 = x1, y1
                x1, y1 = extremum, y_extremum
            else:
                x2, y2 = extremum, y_extremum

        # extremum находится слева от x1
        elif x0 < extremum < x1:
            if y_extremum < y1:
                x2, y2 = x1, y1
                x1, y1 = extremum, y_extremum
            else:
                x0, y0 = extremum, y_extremum

        else:
            print("Вершина параболы вышла за интервал локализации")
            break

        n_history.append(n_evaluations)
        length_history.append(x2 - x0)

    x_min = x1

    print(f"Минимум (метод парабол) {x_min}")
    print(f"Количество вычислений функции (метод парабол) {n_evaluations}")

    visualize(
        function,
        x_grid,
        function(x_grid),
        x_min,
        np.array([x0, x2]),
    )

    return n_history, length_history


def visualize_convergence(method, function):
    n_history, length_history = method(function)
    print(f"Длина интервала локализации (метод {method.__name__}) {length_history[-1]}")
    plt.plot(
        n_history, length_history, "o-", label=f"Метод {method.__name__}", color="blue"
    )
    plt.xlabel("Количество вычислений функции")
    plt.ylabel("Длина интервала локализации")
    plt.legend()
    plt.grid()
    plt.show()


# passive(f1)
# visualize_convergence(dichotomy, f2)
# visualize_convergence(fibonacci, f1)
# visualize_convergence(golden_ratio, f2)
visualize_convergence(parabola, f2)
