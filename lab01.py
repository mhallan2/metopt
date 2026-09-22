from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class OptimizationResult:
    """Единый результат метода, аналогичный по идее scipy.optimize.OptimizeResult."""

    x_min: float
    f_min: float
    n_evaluations: int
    neighbors: np.ndarray
    evaluation_points: np.ndarray
    n_history: list[int] = field(default_factory=list)
    length_history: list[float] = field(default_factory=list)
    lipschitz_M: float | None = None

    @property
    def interval_length(self):
        return self.neighbors[1] - self.neighbors[0]


eps = 1e-3
eps_f = 1e-3
max_evaluations = 1000
delta = 1e-4
phi = (1.0 + np.sqrt(5.0)) / 2.0

N_TEST_POINTS = 1000

# Текущий рабочий интервал. Перед запуском метода он автоматически
# устанавливается по тестовой функции из ТЗ.
a, b = -3.0, 3.0
x_grid = np.linspace(a, b, N_TEST_POINTS)


def f1(x):
    return (x - 2) ** 2 + 1


def f2(x):
    return x**2 + 2 * np.sin(3 * x)


def f3(x):
    return x**2 + 5 * np.sin(2 * x) + 2 * np.cos(3 * x)


FUNCTION_INTERVALS = {
    f1: (0.0, 5.0),
    f2: (0.8, 3.0),
    f3: (-3.0, 3.0),
}


def set_interval(function):
    """Устанавливает интервал из ТЗ для выбранной тестовой функции."""
    global a, b, x_grid

    if function in FUNCTION_INTERVALS:
        a, b = FUNCTION_INTERVALS[function]

    x_grid = np.linspace(a, b, N_TEST_POINTS)


def visualize(
    function,
    x_grid,
    values,
    x_min,
    x_neighbors,
    evaluation_points=None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if evaluation_points is not None and len(evaluation_points) > 0:
        evaluation_points = np.asarray(evaluation_points)
        evaluation_values = function(evaluation_points)
    else:
        evaluation_points = None
        evaluation_values = None

    # Общий вид функции на всем интервале
    axes[0].plot(x_grid, values, label="f(x)")

    if evaluation_points is not None:
        axes[0].scatter(
            evaluation_points,
            evaluation_values,
            label="Точки вычислений",
            s=18,
            alpha=0.7,
        )

    axes[0].axvline(
        x_min,
        linestyle="--",
        label="Найденный минимум",
    )
    axes[0].scatter(
        x_neighbors,
        function(x_neighbors),
        label="Границы интервала",
    )

    axes[0].set_xlim(a, b)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("Функция на всем интервале")
    axes[0].legend()
    axes[0].grid()

    # Увеличение около минимума
    local_width = x_neighbors[1] - x_neighbors[0]
    plot_delta = local_width * 0.4

    if np.isclose(plot_delta, 0.0):
        plot_delta = max((b - a) * 0.02, eps)

    axes[1].plot(x_grid, values, label="f(x)")

    if evaluation_points is not None:
        axes[1].scatter(
            evaluation_points,
            evaluation_values,
            label="Точки вычислений",
            s=18,
            alpha=0.7,
        )

    axes[1].axvline(
        x_min,
        linestyle="--",
        label="Найденный минимум",
    )
    axes[1].scatter(
        x_neighbors,
        function(x_neighbors),
        label="Границы интервала",
    )

    axes[1].set_xlim(
        x_neighbors[0] - plot_delta,
        x_neighbors[1] + plot_delta,
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


def passive(function, N=None, show=True):
    set_interval(function)

    if N is None:
        N = int(np.ceil((b - a) / eps)) - 1

        if N % 2 == 0:
            N += 1

    if N <= 0:
        raise ValueError("N должно быть положительным")

    if N % 2 == 1:
        step = (b - a) / (N + 1)
        x_grid_passive = a + step * np.arange(1, N + 1)

    else:
        k = N // 2
        centers = a + (b - a) / (k + 1) * np.arange(1, k + 1)
        x_grid_passive = np.empty(N)
        x_grid_passive[0::2] = centers - delta
        x_grid_passive[1::2] = centers
        x_grid_passive.sort()

    values = function(x_grid_passive)

    min_idx = np.argmin(values)
    x_min = x_grid_passive[min_idx]

    left = a if min_idx == 0 else x_grid_passive[min_idx - 1]
    right = b if min_idx == len(x_grid_passive) - 1 else x_grid_passive[min_idx + 1]

    x_neighbors = np.array([left, right])

    print(f"Минимум (пассивный) {x_min}")
    print(f"Количество вычислений функции (пассивный) {N}")

    if show:
        visualize(
            function,
            x_grid,
            function(x_grid),
            x_min,
            x_neighbors,
            x_grid_passive,
        )

    return OptimizationResult(
        x_min=x_min,
        f_min=function(x_min),
        n_evaluations=N,
        neighbors=x_neighbors,
        evaluation_points=x_grid_passive,
    )


def dichotomy(function, N=None, show=True):
    set_interval(function)

    left, right = a, b

    n_evaluations = 0
    n_history = []
    length_history = []
    evaluation_points = []

    while right - left > eps:
        if N is not None and n_evaluations + 2 > N:
            break

        middle = 0.5 * (left + right)

        x1 = middle - delta
        x2 = middle + delta

        f1_value = function(x1)
        f2_value = function(x2)

        evaluation_points.extend([x1, x2])
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

    if show:
        visualize(
            function,
            x_grid,
            function(x_grid),
            x_min,
            np.array([left, right]),
            np.array(evaluation_points),
        )

    return OptimizationResult(
        x_min=x_min,
        f_min=function(x_min),
        n_evaluations=n_evaluations,
        neighbors=np.array([left, right]),
        evaluation_points=np.array(evaluation_points),
        n_history=n_history,
        length_history=length_history,
    )


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


def fibonacci(function, N=None, show=True):
    set_interval(function)

    left, right = 0.0, 1.0

    normalized_function = lambda x: function(a + (b - a) * x)
    normalized_delta = delta / (b - a)

    if N is None:
        fib_need = (b - a) / (eps - delta)
        n = fibonacci_numbers_until(fib_need)
    else:
        if N < 3:
            raise ValueError("Для метода Фибоначчи нужно N >= 3")

        # В твоей реализации фактическое число вычислений равно n - 1.
        # Поэтому для сравнения ровно при N вычислениях используем n = N + 1.
        n = N + 1

    x1 = left + (right - left) * fib(n - 2) / fib(n)
    x2 = left + (right - left) * fib(n - 1) / fib(n)

    f1_value = normalized_function(x1)
    f2_value = normalized_function(x2)

    evaluation_points = [
        a + (b - a) * x1,
        a + (b - a) * x2,
    ]
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

            evaluation_points.append(a + (b - a) * x1)
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

            evaluation_points.append(a + (b - a) * x2)
            n_evaluations += 1

    x2 = x1 + normalized_delta
    f2_value = normalized_function(x2)

    evaluation_points.append(a + (b - a) * x2)
    n_evaluations += 1

    if f1_value < f2_value:
        right = x2
    else:
        left = x1

    n_history.append(n_evaluations)
    length_history.append((b - a) * (right - left))

    x_min = a + (b - a) * (left + right) / 2.0
    neighbors = np.array(
        [
            a + (b - a) * left,
            a + (b - a) * right,
        ]
    )

    print(f"Минимум (метод Фибоначчи) {x_min}")
    print(f"Количество вычислений функции (метод Фибоначчи) {n_evaluations}")

    if show:
        visualize(
            function,
            x_grid,
            function(x_grid),
            x_min,
            neighbors,
            np.array(evaluation_points),
        )

    return OptimizationResult(
        x_min=x_min,
        f_min=function(x_min),
        n_evaluations=n_evaluations,
        neighbors=neighbors,
        evaluation_points=np.array(evaluation_points),
        n_history=n_history,
        length_history=length_history,
    )


def golden_ratio(function, N=None, show=True):
    set_interval(function)

    left, right = 0.0, 1.0

    normalized_function = lambda x: function(a + (b - a) * x)
    normalized_eps = eps / (b - a)

    x1 = left + (right - left) / phi**2
    x2 = left + (right - left) / phi

    f1_value = normalized_function(x1)
    f2_value = normalized_function(x2)

    evaluation_points = [
        a + (b - a) * x1,
        a + (b - a) * x2,
    ]
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

            if N is None:
                if right - left <= normalized_eps:
                    break
            elif n_evaluations >= N:
                break

            x1 = left + (right - left) / phi**2

            f1_value = normalized_function(x1)
            evaluation_points.append(a + (b - a) * x1)
            n_evaluations += 1

        else:
            left = x1

            x1 = x2
            f1_value = f2_value

            n_history.append(n_evaluations)
            length_history.append((b - a) * (right - left))

            if N is None:
                if right - left <= normalized_eps:
                    break
            elif n_evaluations >= N:
                break

            x2 = left + (right - left) / phi

            f2_value = normalized_function(x2)
            evaluation_points.append(a + (b - a) * x2)
            n_evaluations += 1

    x_min = a + (b - a) * (left + right) / 2.0
    neighbors = np.array(
        [
            a + (b - a) * left,
            a + (b - a) * right,
        ]
    )

    print(f"Минимум (метод золотого сечения) {x_min}")
    print(f"Количество вычислений функции (метод золотого сечения) {n_evaluations}")

    if show:
        visualize(
            function,
            x_grid,
            function(x_grid),
            x_min,
            neighbors,
            np.array(evaluation_points),
        )

    return OptimizationResult(
        x_min=x_min,
        f_min=function(x_min),
        n_evaluations=n_evaluations,
        neighbors=neighbors,
        evaluation_points=np.array(evaluation_points),
        n_history=n_history,
        length_history=length_history,
    )


def bracketing(function, interval, N=None):
    left, right = interval

    f_left = function(left)
    f_right = function(right)
    n_evaluations = 2
    evaluation_points = [left, right]

    while True:
        if N is not None and n_evaluations >= N:
            return None, None, n_evaluations, evaluation_points, (left, right)

        middle = (left + right) / 2.0
        f_middle = function(middle)
        evaluation_points.append(middle)
        n_evaluations += 1

        if f_middle < f_left and f_middle < f_right:
            return (
                (left, middle, right),
                (f_left, f_middle, f_right),
                n_evaluations,
                evaluation_points,
                None,
            )

        # Только защита от зацикливания при граничном минимуме.
        if right - left <= eps:
            return None, None, n_evaluations, evaluation_points, (left, right)

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


def parabola(function, N=None, show=True):
    set_interval(function)

    bracket, values, n_evaluations, evaluation_points, boundary_interval = bracketing(
        function,
        (a, b),
        N=N,
    )

    n_history = []
    length_history = []

    if bracket is None:
        left, right = boundary_interval

        x_candidates = np.array(evaluation_points)
        y_candidates = function(x_candidates)

        min_index = np.argmin(y_candidates)
        x_min = x_candidates[min_index]

        neighbors = np.array([left, right])

        if show:
            visualize(
                function,
                x_grid,
                function(x_grid),
                x_min,
                neighbors,
                np.array(evaluation_points),
            )

        print(f"Минимум (метод парабол) {x_min}")
        print(f"Количество вычислений функции (метод парабол) {n_evaluations}")

        return OptimizationResult(
            x_min=x_min,
            f_min=function(x_min),
            n_evaluations=n_evaluations,
            neighbors=neighbors,
            evaluation_points=np.array(evaluation_points),
            n_history=n_history,
            length_history=length_history,
        )

    (x0, x1, x2) = bracket
    (y0, y1, y2) = values

    while x2 - x0 > eps:
        if N is not None and n_evaluations >= N:
            break

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
        evaluation_points.append(extremum)
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

    if show:
        visualize(
            function,
            x_grid,
            function(x_grid),
            x_min,
            np.array([x0, x2]),
            np.array(evaluation_points),
        )

    return OptimizationResult(
        x_min=x_min,
        f_min=function(x_min),
        n_evaluations=n_evaluations,
        neighbors=np.array([x0, x2]),
        evaluation_points=np.array(evaluation_points),
        n_history=n_history,
        length_history=length_history,
    )


def visualize_convergence(method, function):
    result = method(function, show=False)

    if not result.length_history:
        print(f"Для метода {method.__name__} нет истории локализации.")
        return

    print(
        f"Длина интервала локализации (метод {method.__name__}) "
        f"{result.length_history[-1]}"
    )

    plt.semilogy(
        result.n_history,
        result.length_history,
        "o-",
        label=f"Метод {method.__name__}",
    )
    plt.xlabel("Количество вычислений функции")
    plt.ylabel("Длина интервала локализации")
    plt.legend()
    plt.grid()
    plt.show()


def compare_convergence(function):
    """Один график сходимости последовательных методов из ТЗ."""
    methods = [
        ("Дихотомия", dichotomy),
        ("Фибоначчи", fibonacci),
        ("Золотое сечение", golden_ratio),
        ("Параболы", parabola),
    ]

    plt.figure(figsize=(8, 5))

    for name, method in methods:
        result = method(function, show=False)

        if result.length_history:
            plt.semilogy(
                result.n_history,
                result.length_history,
                "o-",
                label=name,
            )

    plt.xlabel("Количество вычислений функции")
    plt.ylabel("Длина интервала локализации")
    plt.title("Сравнение скорости сходимости")
    plt.legend()
    plt.grid()
    plt.show()


# def f4(x):
#     return np.exp(20 * (x - 1.7)) - 20 * (x - 1.7) - 1


# FUNCTION_INTERVALS[f4] = (0.8, 3.0)


def update_lipschitz(
    x_points,
    y_points,
    current_M=-np.inf,
    safety=1.09,
):
    order = np.argsort(x_points)
    x_sorted = np.array(x_points)[order]
    y_sorted = np.array(y_points)[order]

    slopes = np.abs(np.diff(y_sorted) / np.diff(x_sorted))

    new_estimate = safety * np.max(slopes)

    return max(current_M, new_estimate)


def broken_lines(function, N=None, show=True):
    set_interval(function)

    x_points = np.array([a, b])
    y_values = function(x_points)

    n_evaluations = 2
    # M оценивается только по уже вычисленным методом точкам
    # и адаптивно увеличивается по ходу поиска.
    current_M = max(
        update_lipschitz(x_points, y_values),
        1e-6,
    )

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

        if N is None:
            if np.min(y_values) - lower_bound < eps_f:
                print(
                    "Метод достиг требуемой точности после "
                    f"{n_evaluations} вычислений функции."
                )
                break

            if n_evaluations >= max_evaluations:
                print(
                    "Метод достиг максимального числа вычислений функции "
                    f"({max_evaluations})."
                )
                break

        elif n_evaluations >= N:
            break

        x_new = t_values[best_index]
        y_new = function(x_new)

        n_evaluations += 1

        insert_index = np.searchsorted(x_points, x_new)
        x_points = np.insert(x_points, insert_index, x_new)
        y_values = np.insert(y_values, insert_index, y_new)

        current_M = update_lipschitz(
            x_points,
            y_values,
            current_M,
        )

    index_min = np.argmin(y_values)
    x_min = x_points[index_min]

    left_neighbor = a if index_min == 0 else x_points[index_min - 1]
    right_neighbor = b if index_min == len(x_points) - 1 else x_points[index_min + 1]
    neighbors = np.array([left_neighbor, right_neighbor])

    print(f"Минимум (метод ломаных) {x_min}")
    print(f"Количество вычислений функции (метод ломаных) {n_evaluations}")
    print(f"Текущая оценка M: {current_M}")

    if show:
        visualize(
            function,
            x_grid,
            function(x_grid),
            x_min,
            neighbors,
            x_points,
        )

    return OptimizationResult(
        x_min=x_min,
        f_min=y_values[index_min],
        n_evaluations=n_evaluations,
        neighbors=neighbors,
        evaluation_points=x_points,
        lipschitz_M=current_M,
    )


# def estimate_lipschitz(function, interval, n_points=20, safety=1.1):
#     left, right = interval

#     x = np.linspace(left, right, n_points)
#     y = function(x)

#     slopes = np.abs(np.diff(y) / np.diff(x))

#     return safety * np.max(slopes)


def search(function, N=None, show=True):
    set_interval(function)

    # Если N не задано, выбираем его из требуемой погрешности
    # по координате x_min.
    if N is None:
        N = int(np.ceil((b - a) / (2.0 * eps)))

    step = (b - a) / N

    x_points = a + (np.arange(N) + 0.5) * step
    y_values = function(x_points)

    min_index = np.argmin(y_values)
    x_min = x_points[min_index]

    left_neighbor = max(a, x_min - 0.5 * step)
    right_neighbor = min(b, x_min + 0.5 * step)
    neighbors = np.array([left_neighbor, right_neighbor])

    print(f"Минимум (метод перебора): {x_min}")
    print(f"Количество вычислений функции: {N}")
    print(
        "Оценка погрешности по координате:",
        (b - a) / (2.0 * N),
    )

    if show:
        visualize(
            function,
            x_grid,
            function(x_grid),
            x_min,
            neighbors,
            x_points,
        )

    return OptimizationResult(
        x_min=x_min,
        f_min=y_values[min_index],
        n_evaluations=N,
        neighbors=neighbors,
        evaluation_points=x_points,
    )


def reference_minimum(function, n_points=2000):
    """Плотная сетка только для проверки фактической ошибки в сравнении."""
    set_interval(function)

    x = np.linspace(a, b, n_points)
    y = function(x)

    index = np.argmin(y)

    return x[index], y[index]


def compare_local_methods_equal_evaluations(function, N=20):
    """Сравнение локальных методов при одинаковом числе вычислений."""

    if N % 2 != 0:
        raise ValueError("Для дихотомии выберите четное N")

    x_reference, _ = reference_minimum(function)

    methods = [
        ("Пассивный", passive),
        ("Дихотомия", dichotomy),
        ("Фибоначчи", fibonacci),
        ("Золотое сечение", golden_ratio),
        ("Параболы", parabola),
    ]

    results = []

    for name, method in methods:
        result = method(function, N=N, show=False)

        results.append(
            (
                name,
                result.n_evaluations,
                result.x_min,
                abs(result.x_min - x_reference),
                result.interval_length,
            )
        )

    print(f"\nСравнение локальных методов при N = {N}")
    print(
        f"{'Метод':<20}"
        f"{'Вычислений':>14}"
        f"{'x_min':>16}"
        f"{'|x - x*|':>16}"
        f"{'Длина интервала':>20}"
    )

    for name, n_eval, x_min, x_error, interval_length in results:
        print(
            f"{name:<20}"
            f"{n_eval:>14d}"
            f"{x_min:>16.9f}"
            f"{x_error:>16.6e}"
            f"{interval_length:>20.6e}"
        )

    names = [row[0] for row in results]
    interval_lengths = [row[4] for row in results]

    plt.figure(figsize=(9, 5))
    plt.bar(names, interval_lengths)
    plt.yscale("log")
    plt.ylabel("Длина интервала локализации")
    plt.title(f"Одинаковое число вычислений: N = {N}")
    plt.xticks(rotation=20)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def compare_global_methods_equal_evaluations(function, N=325):
    """Сравнение перебора и метода ломаных при одном бюджете N."""
    x_reference, f_reference = reference_minimum(function)

    search_result = search(
        function,
        N=N,
        show=False,
    )
    broken_result = broken_lines(
        function,
        N=N,
        show=False,
    )

    results = [
        (
            "Перебор",
            search_result,
        ),
        (
            "Ломаные",
            broken_result,
        ),
    ]

    print(f"\nСравнение глобальных методов при N = {N}")
    print(
        f"{'Метод':<15}{'Вычислений':>14}{'x_min':>16}{'|x - x*|':>16}{'|f - f*|':>16}"
    )

    for name, result in results:
        print(
            f"{name:<15}"
            f"{result.n_evaluations:>14d}"
            f"{result.x_min:>16.9f}"
            f"{abs(result.x_min - x_reference):>16.6e}"
            f"{abs(result.f_min - f_reference):>16.6e}"
        )


# Примеры запуска:


broken_lines(f3)
passive(f1)
golden_ratio(f2)

visualize_convergence(dichotomy, f1)
compare_convergence(f1)

compare_local_methods_equal_evaluations(f1, N=20)
compare_global_methods_equal_evaluations(f3, N=325)
