import matplotlib.pyplot as plt
import numpy as np


# --- Функция для построения графика ---
# noinspection PyShadowingNames
def plot_trajectory(history, func, method_name):
    x_vals = np.linspace(min(h[0] for h in history) - 1, max(h[0] for h in history) + 1, 100)
    y_vals = np.linspace(min(h[1] for h in history) - 1, max(h[1] for h in history) + 1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func(np.array([X, Y]))

    plt.figure(figsize=(10, 8))
    CS = plt.contour(X, Y, Z, 10, cmap='autumn', linewidths=2.5)
    plt.clabel(CS, inline=True, fontsize=12)

    trajectory_x = [h[0] for h in history]
    trajectory_y = [h[1] for h in history]
    plt.plot(trajectory_x, trajectory_y, 'r--', marker=".", label='Траектория поиска')

    # Рисуем линии уровня вблизи точек траектории
    for point in history:
        level_value = func(np.array(point[:2]))
        plt.contour(X, Y, Z, levels=[level_value], colors='black', linewidths=0.5)

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(f'Метод: {method_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(method_name)

    # Вывод траектории в виде таблицы
    print(f"\nТраектория поиска ({method_name}):")
    print("-" * 44)
    print(f'{"X_1":^14}\t{"X_2":^14}\t{"F_X":^14}\t')
    print("-" * 44)
    for point in history:
        print(f"{point[0].item():>10.3f}\t\t{point[1].item():>10.3f}\t\t{func(point).item():>10.3f}")
    print("-" * 44)

    print(f'fig saved as {method_name}.png')
    # plt.show(block=True)
    plt.close()
