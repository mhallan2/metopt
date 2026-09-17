import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


a, b = -3.0, 3.0
eps = 1e-3
delta = 1e-4
phi = (1.0 + np.sqrt(5.0)) / 2.0

N_TEST_POINTS = 1000
bracket_step = 0.8
N_COMPARE = 10  # N_min = 5, N_max = 20

x_grid = np.linspace(a, b, N_TEST_POINTS)


def f1(x):
    """Унимодальная."""
    return (x - 2) ** 2 + 1


def f2(x):
    """Унимодальная, но не выпуклая."""
    return x**2 + 2 * np.sin(3 * x)


def f3(x):
    """Многоэкстремальная."""
    return x**2 + 5 * np.sin(2 * x) + 2 * np.cos(3 * x)


def visualize(function, x_grid, values, x_min, x_neighbors):
    """Строит график функции с отмеченными минимумом и интервалом локализации."""
    x_neighbors = np.asarray(x_neighbors)
    plt.plot(x_grid, values, label="f(x)", color="blue")
    plt.scatter(x_min, function(x_min), color="red", label="Минимум")
    plt.scatter(
        x_neighbors,
        function(x_neighbors),
        color="green",
        label="Границы интервала",
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()


def passive(function):
    """Пассивный метод."""

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
    """Метод дихотомии."""

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
    """Итеративная реализация поиска чисел Фибоначчи."""

    first = 0
    second = 1

    for _ in range(n):
        first, second = second, first + second

    return first


def fibonacci_numbers_until(value):
    """Количество предшествующих чисел Фибоначчи."""

    first = 1
    second = 1
    n = 2

    while second < value:
        first, second = second, first + second
        n += 1

    return n


def fibonacci(function):
    """Метод Фибоначчи."""

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

    while n > 3:  # Для n = 3 координаты точек совпадут (F1/F3 = F2/F3)
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
    """Метод золотого сечения."""

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
    print(
        f"Количество вычислений функции (метод золотого сечения) {n_evaluations}"
    )

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
    x0 = left
    x1 = x0 + bracket_step

    f0 = function(x0)
    f1 = function(x1)
    n_evaluations = 2

    while True:
        x2 = x1 + bracket_step

        if x2 > right:
            break

        f2 = function(x2)
        n_evaluations += 1

        # Разрешаем равенство с одной стороны: это важно, например, для f1,
        # когда две симметричные точки могут иметь одинаковые значения функции.
        if f1 <= f0 and f1 <= f2 and (f1 < f0 or f1 < f2):
            print(
                f"Количество вычислений функции (поиск тройки точек): {n_evaluations}"
            )

            x_min = x1
            neighbors = np.array([x0, x2])

            visualize(
                function,
                x_grid,
                function(x_grid),
                x_min,
                neighbors,
            )

            # Возвращаем также число вычислений, потраченных на поиск тройки.
            # Для честного сравнения метода парабол эти вычисления входят в общий бюджет.
            return (x0, x_min, x2), (f0, f1, f2), n_evaluations

        x0, f0 = x1, f1
        x1, f1 = x2, f2

    raise ValueError("Не нашлась подходящая тройка точек.")


def parabola(function):
    """Метод парабол."""

    (x0, x1, x2), (y0, y1, y2), n_evaluations = bracketing(function, (a, b))

    # После поиска тройки уже есть первый интервал локализации.
    # В историю записываем реальное суммарное число вычислений функции.
    n_history = [n_evaluations]
    length_history = [x2 - x0]

    while x2 - x0 > eps:
        denominator = (y1 - y2) * (x1 - x0) - (y1 - y0) * (x1 - x2)

        if np.isclose(denominator, 0):
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

        if abs(extremum - x1) < eps:
            n_history.append(n_evaluations)
            length_history.append(x2 - x0)
            x1 = extremum
            y1 = y_extremum
            print("Вершина параболы совпала с текущим приближением")
            break

        if x1 < extremum < x2:
            if y_extremum < y1:
                x0, y0 = x1, y1
                x1, y1 = extremum, y_extremum
            else:
                x2, y2 = extremum, y_extremum

        elif x0 < extremum < x1:
            if y_extremum < y1:
                x2, y2 = x1, y1
                x1, y1 = extremum, y_extremum
            else:
                x0, y0 = extremum, y_extremum

        else:
            raise ValueError("Вершина параболы вышла за интервал локализации.")

        n_history.append(n_evaluations)
        length_history.append(x2 - x0)

    x_min = x1
    neighbours = [x0, x2]

    print(f"Минимум (метод парабол) {x_min}")
    # print(f"Соседние точки: ({x0:.3f}, {x2:.3f})")
    print(f"Количество вычислений функции (метод парабол) {n_evaluations}")

    visualize(
        function,
        x_grid,
        function(x_grid),
        x_min,
        neighbours,
    )

    return n_history, length_history


def brute_force(function, n_intervals):
    """Метод перебора для глобальной минимизации."""

    points = np.linspace(a, b, n_intervals + 1)
    values = function(points)

    min_idx = int(np.argmin(values))
    x_min = points[min_idx]
    f_min = values[min_idx]

    m_lipschitz = estimate_M(function)
    error_bound = m_lipschitz * (b - a) / (2 * n_intervals)

    print(f"Минимум методом перебора: x = {x_min}")
    print(f"Значение функции: f(x) = {f_min}")
    print(f"Число разбиений: N = {n_intervals}")
    print(f"Оценка погрешности по функции: {error_bound}")

    visualize(
        function,
        x_grid,
        function(x_grid),
        x_min,
        np.array([x_min]),
    )

    return x_min, f_min


def estimate_M(function, left=a, right=b, n_points=10_000):
    """Численно оценивает константу Липшица по конечным разностям."""

    points = np.linspace(left, right, n_points)
    values = function(points)
    slopes = np.abs(np.diff(values) / np.diff(points))

    return float(np.max(slopes))


def broken_lines(function):
    """Метод ломаных для глобальной минимизации.

    Возвращает найденный минимум и историю работы метода.
    """

    points = [a, b]
    values = [function(a), function(b)]

    m_lipschitz = estimate_M(function)
    print(f"M = {m_lipschitz}")

    n_evaluations = 2
    n_history = []
    x_min_history = []
    f_min_history = []
    lower_bound_history = []
    gap_history = []

    while True:
        order = np.argsort(points)
        points = list(np.asarray(points)[order])
        values = list(np.asarray(values)[order])

        candidates = []
        lower_bounds = []

        for i in range(len(points) - 1):
            x_left = points[i]
            x_right = points[i + 1]
            f_left = values[i]
            f_right = values[i + 1]

            x_candidate = (x_left + x_right) / 2 + (f_left - f_right) / (
                2 * m_lipschitz
            )
            lower_bound = f_left - m_lipschitz * (x_candidate - x_left)

            candidates.append(x_candidate)
            lower_bounds.append(lower_bound)

        candidate_idx = int(np.argmin(lower_bounds))
        x_candidate = candidates[candidate_idx]
        lower_bound = lower_bounds[candidate_idx]

        best_idx = int(np.argmin(values))
        x_min = points[best_idx]
        f_min = values[best_idx]
        gap = f_min - lower_bound

        # История состояния после текущего числа вычислений функции.
        n_history.append(n_evaluations)
        x_min_history.append(x_min)
        f_min_history.append(f_min)
        lower_bound_history.append(lower_bound)
        gap_history.append(gap)

        if gap <= eps:
            break

        f_candidate = function(x_candidate)
        n_evaluations += 1
        points.append(x_candidate)
        values.append(f_candidate)

    if best_idx == 0:
        neighbors = np.array([points[0], points[1]])
    elif best_idx == len(points) - 1:
        neighbors = np.array([points[-2], points[-1]])
    else:
        neighbors = np.array([points[best_idx - 1], points[best_idx + 1]])

    visualize(
        function,
        x_grid,
        function(x_grid),
        x_min,
        neighbors,
    )

    history = {
        "n_evaluations": np.asarray(n_history),
        "x_min": np.asarray(x_min_history),
        "f_min": np.asarray(f_min_history),
        "lower_bound": np.asarray(lower_bound_history),
        "gap": np.asarray(gap_history),
    }

    print(f"Минимум (метод ломаных): x = {x_min}")
    print(f"Значение функции: f(x) = {f_min}")
    print(f"Количество вычислений функции: {n_evaluations}")

    return x_min, f_min, history


# def analyze_broken_lines_history(history):
#     """Показывает, насколько рано стабилизируется найденный x_min."""

#     n_history = history["n_evaluations"]
#     x_history = history["x_min"]
#     final_x = x_history[-1]

#     # Первое вычисление, после которого лучший x_min больше не меняется.
#     stabilization_idx = None
#     for i in range(len(x_history)):
#         if np.allclose(x_history[i:], final_x, rtol=0.0, atol=1e-12):
#             stabilization_idx = i
#             break

#     # Первое вычисление, после которого оценка x_min уже в eps от финальной.
#     eps_idx = None
#     for i in range(len(x_history)):
#         if np.all(np.abs(x_history[i:] - final_x) <= eps):
#             eps_idx = i
#             break

#     if stabilization_idx is not None:
#         print(
#             "x_min окончательно перестал меняться после "
#             f"{n_history[stabilization_idx]} вычислений функции."
#         )

#     if eps_idx is not None:
#         print(
#             f"В пределах eps = {eps} от итогового x_min метод находится уже после "
#             f"{n_history[eps_idx]} вычислений функции."
#         )

#     print(
#         "После этого метод продолжает вычисления, чтобы уменьшить разрыв "
#         "между лучшим найденным значением и нижней липшицевой оценкой."
#     )

#     return stabilization_idx, eps_idx


def visualize_convergence(method, function):
    """Строит график длины отрезка локализации от числа вычислений функции методом."""

    n_history, length_history = method(function)
    function_name = method.__name__
    final_length = length_history[-1]
    print(
        f"Длина интервала локализации (метод {function_name}) {final_length}",
    )
    print(60 * "-")
    plt.plot(
        n_history,
        length_history,
        "o-",
        label=f"Метод {function_name}",
        color="blue",
    )
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Количество вычислений функции")
    plt.ylabel("Длина интервала локализации")
    plt.legend()
    plt.grid()
    plt.show()

    return n_history, length_history


def passive_fixed(function, n_evaluations):
    """Пассивный метод при заранее заданном числе вычислений функции."""

    step = (b - a) / (n_evaluations + 1)
    points = np.linspace(a + step, b - step, n_evaluations)
    values = function(points)

    min_idx = np.argmin(values)

    # Интервал неопределённости строим по соседним с лучшей точкой узлам.
    # На случай минимума у края используем соответствующую границу [a, b].
    if min_idx == 0:
        left = a
        right = points[1]
    elif min_idx == n_evaluations - 1:
        left = points[-2]
        right = b
    else:
        left = points[min_idx - 1]
        right = points[min_idx + 1]

    return right - left


def length_at_n(n_history, length_history, n_compare):
    """Возвращает длину интервала ровно при n_compare вычислениях."""

    if n_compare not in n_history:
        return None

    idx = n_history.index(n_compare)
    return length_history[idx]


def compare_efficiency(function, histories, n_compare=N_COMPARE):
    """Сравнивает методы при одинаковом числе вычислений функции.

    Критерий эффективности: при одинаковом N более эффективен тот метод,
    у которого меньше итоговая длина интервала локализации L_N.
    """

    print()
    print(f"Сравнение эффективности методов при N = {n_compare}")
    print(60 * "-")

    results = {}

    # Пассивный метод сравниваем при том же N, но L(N) для него не строим.
    passive_length = passive_fixed(function, n_compare)
    results["passive"] = passive_length
    print(f"{'passive':15s}: L = {passive_length:.6f}")

    for method_name, (n_history, length_history) in histories.items():
        length = length_at_n(n_history, length_history, n_compare)

        if length is None:
            print(
                f"{method_name:10s}: нет точки ровно при N = {n_compare} "
                "(метод завершился раньше или имеет другой шаг по N)"
            )
            continue

        results[method_name] = length
        print(f"{method_name:10s}: L = {length:.6f}")

    # Если у всех методов есть значение при одном и том же N,
    # можно непосредственно указать метод с минимальной длиной интервала.
    if len(results) == len(histories) + 1:
        best_method = min(results, key=results.get)
        print(60 * "-")
        print(
            f"При N = {n_compare} наименьшая длина интервала у метода "
            f"{best_method}: L = {results[best_method]:.6f}"
        )

        # Дополнительная наглядная диаграмма сравнения при одинаковом N.
        plt.bar(list(results.keys()), list(results.values()))
        plt.xlabel("Метод")
        plt.ylabel("Длина интервала локализации")
        plt.title(f"Сравнение эффективности при N = {n_compare}")
        plt.grid(axis="y")
        plt.show()
    else:
        print(60 * "-")
        print(
            "Строгое сравнение всех методов при этом N невозможно: "
            "не у каждого метода есть состояние ровно после такого числа вычислений."
        )


if __name__ == "__main__":
    x_min, f_min, history = broken_lines(f3)
    # analyze_broken_lines_history(history)
    brute_force(f3, history["n_evaluations"][-1])
