import numpy as np
import matplotlib.pyplot as plt


a, b = -3.0, 3.0
eps = 1e-3
eps_f = 1e-3
max_evaluations = 1000
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Общий вид функции на всем интервале
    axes[0].plot(x_grid, values, label="f(x)")
    axes[0].scatter(x_min, function(x_min), label="Минимум", c="red")
    axes[0].scatter(
        x_neighbors, function(x_neighbors), label="Границы интервала", c="green"
    )

    axes[0].set_xlim(a, b)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("Функция на всем интервале")
    axes[0].legend()
    axes[0].grid()

    # Увеличение около минимума
    delta = (x_neighbors[1] - x_neighbors[0]) * 0.4

    axes[1].plot(x_grid, values, label="f(x)")
    axes[1].scatter(
        x_min,
        function(x_min),
        label="Минимум",
        c="red",
    )
    axes[1].scatter(
        x_neighbors, function(x_neighbors), label="Границы интервала", c="green"
    )

    axes[1].set_xlim(
        x_neighbors[0] - delta,
        x_neighbors[1] + delta,
    )

    axes[1].ticklabel_format(
        style="plain",
        axis="x",
        useOffset=False,
    )

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f(x)")
    axes[1].set_title("Окрестность минимума")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
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
    left, right = interval

    f_left = function(left)
    f_right = function(right)
    n_evaluations = 2

    while True:
        middle = (left + right) / 2.0
        f_middle = function(middle)
        n_evaluations += 1

        if f_middle < f_left and f_middle < f_right:
            return (left, middle, right), (f_left, f_middle, f_right), n_evaluations

        if f_left < f_middle:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle


def parabola_vertex(x0, x1, x2, y0, y1, y2):
    denominator = (y1 - y2) * (x1 - x0) - (y1 - y0) * (x1 - x2)

    if np.isclose(denominator, 0.0):
        return None

    return (
        x1
        - 0.5 * ((x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)) / denominator
    )


def parabola(function):
    (x0, x1, x2), (y0, y1, y2), n_evaluations = bracketing(function, (a, b))

    n_history = []
    length_history = []

    while x2 - x0 > eps:
        extremum = parabola_vertex(
            x0,
            x1,
            x2,
            y0,
            y1,
            y2,
        )

        use_parabola = extremum is not None

        if use_parabola and (not x0 < extremum < x2 or np.isclose(extremum, x1)):
            use_parabola = False

        if not use_parabola:
            if x1 - x0 > x2 - x1:
                extremum = 0.5 * (x0 + x1)
            else:
                extremum = 0.5 * (x1 + x2)

        y_extremum = function(extremum)
        n_evaluations += 1

        if extremum < x1:
            if y_extremum < y1:
                x2, y2 = x1, y1
                x1, y1 = extremum, y_extremum
            else:
                x0, y0 = extremum, y_extremum
        else:
            if y_extremum < y1:
                x0, y0 = x1, y1
                x1, y1 = extremum, y_extremum
            else:
                x2, y2 = extremum, y_extremum

        n_history.append(n_evaluations)
        length_history.append(x2 - x0)

    x_min = 0.5 * (x0 + x2)

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


def f4(x):
    return np.exp(20 * (x - 1.7)) - 20 * (x - 1.7) - 1


def update_lipschitz(
    x_points,
    y_points,
    current_M=-np.inf,
    safety=1.09,
):
    order = np.argsort(x_points)
    y_sorted = np.array(y_points)[order]

    slopes = np.abs(np.diff(y_sorted) / np.diff(x_points))

    new_estimate = safety * np.max(slopes)

    return max(current_M, new_estimate)


def broken_lines(function):
    x_points = np.array([a, b])
    y_values = function(x_points)

    n_evaluations = 2
    current_M = max(update_lipschitz(x_points, y_values), 1e-6)

    while True:
        R_values = []
        t_values = []

        for j in range(len(x_points) - 1):
            x_left = x_points[j]
            x_right = x_points[j + 1]

            y_left = y_values[j]
            y_right = y_values[j + 1]

            t = 0.5 * (x_left + x_right) - (y_right - y_left) / (2.0 * current_M)
            R = 0.5 * (y_left + y_right) - current_M * (x_right - x_left) / 2.0

            R_values.append(R)
            t_values.append(t)

        best_index = np.argmin(R_values)
        lower_bound = R_values[best_index]

        if np.min(y_values) - lower_bound < eps_f:
            print(
                f"Метод достиг требуемой точности после {n_evaluations} вычислений функции."
            )
            break

        if n_evaluations >= max_evaluations:
            print(
                f"Метод достиг максимального числа вычислений функции ({max_evaluations})."
            )
            break

        x_new = t_values[best_index]
        y_new = function(x_new)

        n_evaluations += 1

        insert_index = np.searchsorted(x_points, x_new)
        x_points = np.insert(x_points, insert_index, x_new)
        y_values = np.insert(y_values, insert_index, y_new)

        current_M = update_lipschitz(x_points, y_values, current_M)

    index_min = np.argmin(y_values)
    x_min = x_points[index_min]

    print(f"Минимум (метод ломаных) {x_min}")
    print(
        f"Длина интервала локализации (метод ломаных) {x_points[index_min + 1] - x_points[index_min - 1]}"
    )

    visualize(
        function,
        x_grid,
        function(x_grid),
        x_min,
        np.array([x_points[index_min - 1], x_points[index_min + 1]]),
    )

    return x_min, x_points, y_values


n0 = 20  # Количество точек для оценки константы Липшица


def estimate_lipschitz(function, interval, n_points=n0, safety=1.1):
    left, right = interval

    x = np.linspace(left, right, n_points)
    y = function(x)

    slopes = np.abs(np.diff(y) / np.diff(x))

    return safety * np.max(slopes)


def search(function):
    N = int(np.ceil((b - a) / (2.0 * eps)))
    step = (b - a) / N

    x_points = a + (np.arange(N) + 0.5) * step
    y_values = function(x_points)

    min_index = np.argmin(y_values)
    x_min = x_points[min_index]

    print(f"Минимум (метод перебора): {x_min}")
    print(f"Количество вычислений функции: {N}")
    print(
        "Оценка погрешности по координате:",
        (b - a) / (2.0 * N),
    )

    visualize(
        function,
        x_grid,
        function(x_grid),
        x_min,
        np.array(
            [
                x_points[min_index - 1],
                x_points[min_index + 1],
            ]
        ),
    )


search(f3)
# broken_lines(f3)
# passive(f1)
# functions = [f1, f2]
# for f in functions:
#     visualize_convergence(dichotomy, f)
#     visualize_convergence(fibonacci, f)
#     visualize_convergence(golden_ratio, f)
#     visualize_convergence(parabola, f)
