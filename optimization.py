import numpy as np
import matplotlib.pyplot as plt


# --- Общая функция для целевой функции и градиента ---
# noinspection PyShadowingNames
def objective_function(x, C11, C22, C12, C1, C2):
    return C11 * x[0] ** 2 + C22 * x[1] ** 2 + C12 * x[0] * x[1] + C1 * x[0] + C2 * x[1]


# noinspection PyShadowingNames
def gradient(x, C11, C22, C12, C1, C2):
    grad_x1 = 2 * C11 * x[0] + C12 * x[1] + C1
    grad_x2 = 2 * C22 * x[1] + C12 * x[0] + C2
    return np.array([grad_x1, grad_x2])


# --- Функция для построения графика ---
# noinspection PyShadowingNames
def plot_trajectory(history, objective_func, method_name, C11, C22, C12, C1, C2):
    x_vals = np.linspace(min(h[0] for h in history) - 1, max(h[0] for h in history) + 1, 100)
    y_vals = np.linspace(min(h[1] for h in history) - 1, max(h[1] for h in history) + 1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = objective_func(np.array([X, Y]), C11, C22, C12, C1, C2)

    plt.figure(figsize=(10, 8))
    CS = plt.contour(X, Y, Z, 20)
    plt.clabel(CS, inline=True, fontsize=8)

    trajectory_x = [h[0] for h in history]
    trajectory_y = [h[1] for h in history]
    plt.plot(trajectory_x, trajectory_y, 'r-o', label='Траектория поиска')

    # Рисуем линии уровня вблизи точек траектории
    for point in history:
        level_value = objective_func(np.array(point[:2]), C11, C22, C12, C1, C2)
        plt.contour(X, Y, Z, levels=[level_value], colors='black', linewidths=0.5)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Метод: {method_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Вывод траектории в виде таблицы
    print(f"\nТраектория поиска ({method_name}):")
    print("-----------------------------------")
    print("  X(1)      X(2)      f(X)")
    print("-----------------------------------")
    for point in history:
        print(f"{point[0]:.6f}  {point[1]:.6f}  {point[2]:.6f}")
    print("-----------------------------------")


# --- 2. Метод Релаксации ---
# noinspection PyShadowingNames
def relaxation_method(initial_point, C11, C22, C12, C1, C2, tolerance=1e-6, max_iterations=100):
    x = np.array(initial_point, dtype=float)
    history = []
    iteration = 0
    while iteration < max_iterations:
        x_old = x.copy()

        # Оптимизация по x1 (при фиксированном x2) - для максимизации, производная = 0
        x[0] = - (C12 * x[1] + C1) / (2 * C11) if C11 != 0 else x[0]

        # Оптимизация по x2 (при фиксированном x1) - для максимизации, производная = 0
        x[1] = - (C12 * x[0] + C2) / (2 * C22) if C22 != 0 else x[1]

        history.append((x_old[0], x_old[1], objective_function(x_old, C11, C22, C12, C1, C2)))

        if np.linalg.norm(x - x_old) < tolerance:
            break
        iteration += 1
    return x, history


# noinspection PyShadowingNames
def steepest_ascent_with_linesearch(initial_point, C11, C22, C12, C1, C2, tolerance=1e-6, max_iterations=100):
    x = np.array(initial_point)
    history = []
    iteration = 0
    alpha = 1.0  # Начальное значение шага
    rho = 0.1  # Коэффициент уменьшения шага (0 < rho < 1)
    c = 0.1  # Параметр условия достаточного возрастания (0 < c < 1)

    while iteration < max_iterations:
        grad = gradient(x, C11, C22, C12, C1, C2)
        grad_norm_sq = np.dot(grad, grad)
        f_current = objective_function(x, C11, C22, C12, C1, C2)

        alpha_k = alpha
        while objective_function(x + alpha_k * grad, C11, C22, C12, C1, C2) < f_current + c * alpha_k * grad_norm_sq:
            alpha_k *= rho

        x_next = x + alpha_k * grad
        history.append((x[0], x[1], f_current))

        if np.linalg.norm(x_next - x) < tolerance:
            break

        x = x_next
        iteration += 1
    return x, history


# --- 3. Метод Наискорейшего Подъёма ---
# noinspection PyShadowingNames
def steepest_ascent(initial_point, learning_rate, C11, C22, C12, C1, C2, tolerance=1e-5, max_iterations=10):
    x = np.array(initial_point)
    history = []
    iteration = 0
    while iteration < max_iterations:
        grad = gradient(x, C11, C22, C12, C1, C2)
        x_old = x.copy()
        x = x + learning_rate * grad
        history.append((x_old[0], x_old[1], objective_function(x_old, C11, C22, C12, C1, C2)))
        if np.linalg.norm(x - x_old) < tolerance:
            break
        iteration += 1
    return x, history


# --- 4. Метод Ньютона ---
# noinspection PyShadowingNames
def hessian(C11, C22, C12):
    return np.array([[2 * C11, C12],
                     [C12, 2 * C22]])


# noinspection PyShadowingNames
def newton_method(initial_point, C11, C22, C12, C1, C2, tolerance=1e-6, max_iterations=100):
    x = np.array(initial_point, dtype=float)
    history = []
    iteration = 0
    H = hessian(C11, C22, C12)
    H_inv = np.linalg.inv(H)

    while iteration < max_iterations:
        grad = gradient(x, C11, C22, C12, C1, C2)
        x_old = x.copy()
        x = x - np.dot(H_inv, grad)
        history.append((x_old[0], x_old[1], objective_function(x_old, C11, C22, C12, C1, C2)))
        if np.linalg.norm(x - x_old) < tolerance:
            break
        iteration += 1
    return x, history


# --- 5. Метод Сопряжённых Градиентов ---
# noinspection PyShadowingNames
def conjugate_gradient_method(initial_point, C11, C22, C12, C1, C2, tolerance=1e-6, max_iterations=100):
    x = np.array(initial_point, dtype=float)
    g = gradient(x, C11, C22, C12, C1, C2)
    d = g
    history = []
    iteration = 0
    H = hessian(C11, C22, C12)

    while iteration < max_iterations and np.linalg.norm(g) > tolerance:
        alpha = np.dot(g, g) / np.dot(d, np.dot(H, d))
        x_next = x + alpha * d
        g_next = gradient(x_next, C11, C22, C12, C1, C2)
        beta = np.dot(g_next, g_next) / np.dot(g, g)
        d_next = g_next + beta * d

        history.append((x[0], x[1], objective_function(x, C11, C22, C12, C1, C2)))

        x = x_next
        g = g_next
        d = d_next
        iteration += 1
    return x, history


# --- 6. Метод Бройдена ---
# noinspection PyShadowingNames
def broyden_method(initial_point, C11, C22, C12, C1, C2, tolerance=1e-6, max_iterations=100):
    x = np.array(initial_point, dtype=float)
    grad = gradient(x, C11, C22, C12, C1, C2)
    B = np.eye(len(initial_point))
    history = []
    iteration = 0

    while iteration < max_iterations and np.linalg.norm(grad) > tolerance:
        p = -B @ grad
        x_next = x + p
        grad_next = gradient(x_next, C11, C22, C12, C1, C2)
        s = x_next - x
        y = grad_next - grad

        Bs = B @ s
        if np.dot(s, Bs) != 0:
            B = B + np.outer(s - Bs, s @ B) / (s @ Bs)

        history.append((x[0], x[1], objective_function(x, C11, C22, C12, C1, C2)))

        x = x_next
        grad = grad_next
        iteration += 1
    return x, history


# --- 7. Метод Дэвидена-Флетчера-Пауэлла (DFP) ---
# noinspection PyShadowingNames
def dfp_method(initial_point, C11, C22, C12, C1, C2, tolerance=1e-6, max_iterations=100):
    x = np.array(initial_point, dtype=float)
    grad = gradient(x, C11, C22, C12, C1, C2)
    H = np.eye(len(initial_point))
    history = []
    iteration = 0

    while iteration < max_iterations and np.linalg.norm(grad) > tolerance:
        p = -H @ grad
        alpha = 1.0  # Можно добавить процедуру одномерной оптимизации
        x_next = x + alpha * p
        grad_next = gradient(x_next, C11, C22, C12, C1, C2)
        s = x_next - x
        y = grad_next - grad

        s_y = np.dot(s, y)
        Hy = np.dot(H, y)
        yHy = np.dot(y, Hy)

        if s_y > 1e-8 and yHy > 1e-8:  # Проверка на положительную определенность
            H = H + np.outer(s, s) / s_y - np.outer(Hy, Hy) / yHy

        history.append((x[0], x[1], objective_function(x, C11, C22, C12, C1, C2)))

        x = x_next
        grad = grad_next
        iteration += 1
    return x, history


# --- Задание коэффициентов ---
C11 = -31
C22 = -34
C12 = 4
C1 = 286
C2 = 388

# --- Выбор начальных точек для демонстрации особенностей методов ---
initial_point_relaxation = [0.0, 0.0]
initial_point_steepest = [-50.0, 5.0]
initial_point_newton = [1.0, 1.0]
initial_point_conjugate = [1.0, -1.0]
initial_point_broyden = [1.0, 1.0]
initial_point_dfp = [1.0, 1.0]

# --- Решение задачи различными методами и визуализация ---
# 2. Метод Релаксации
optimal_point_relaxation, history_relaxation = relaxation_method(initial_point_relaxation, C11, C22, C12, C1, C2)
plot_trajectory(history_relaxation, objective_function, "Метод Релаксации", C11, C22, C12, C1, C2)

# 3. Метод Наискорейшего Подъёма
learning_rate_steepest = 0.1  # Подберите шаг обучения
optimal_point_steepest, history_steepest = steepest_ascent_with_linesearch(initial_point_steepest,
                                                                           learning_rate_steepest, C11, C22,
                                                                           C12, C1, C2)
plot_trajectory(history_steepest, objective_function, "Метод Наискорейшего Подъёма", C11, C22, C12, C1, C2)

# 4. Метод Ньютона
optimal_point_newton, history_newton = newton_method(initial_point_newton, C11, C22, C12, C1, C2)
plot_trajectory(history_newton, objective_function, "Метод Ньютона", C11, C22, C12, C1, C2)

# 5. Метод Сопряжённых Градиентов
optimal_point_conjugate, history_conjugate = conjugate_gradient_method(initial_point_conjugate, C11, C22, C12, C1, C2)
plot_trajectory(history_conjugate, objective_function, "Метод Сопряжённых Градиентов", C11, C22, C12, C1, C2)

# 6. Метод Бройдена
optimal_point_broyden, history_broyden = broyden_method(initial_point_broyden, C11, C22, C12, C1, C2)
plot_trajectory(history_broyden, objective_function, "Метод Бройдена", C11, C22, C12, C1, C2)

# 7. Метод Дэвидена-Флетчера-Пауэлла
optimal_point_dfp, history_dfp = dfp_method(initial_point_dfp, C11, C22, C12, C1, C2)
plot_trajectory(history_dfp, objective_function, "Метод Дэвидена-Флетчера-Пауэлла", C11, C22, C12, C1, C2)
