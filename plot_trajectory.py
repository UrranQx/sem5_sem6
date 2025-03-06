import matplotlib.pyplot as plt
import numpy as np


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
