import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np


# --- Функция для построения графика ---
# noinspection PyShadowingNames
def plot_trajectory(history, func, method_name, print_table=True):
    PLOTS_DIR = 'plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)

    METHOD_DIR = os.path.join(PLOTS_DIR, method_name)
    os.makedirs(METHOD_DIR, exist_ok=True)

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
    plt.savefig(f'{os.path.join(METHOD_DIR, method_name)}_2D.svg', format='svg')
    plt.savefig(f'{os.path.join(METHOD_DIR, method_name)}_2D')
    if print_table:
        # Вывод траектории в виде таблицы
        print(f"\nТраектория поиска ({method_name}):")
        print("-" * 54)
        print(f'i\t{"X_1":^14}\t{"X_2":^14}\t{"F_X":^14}\t')
        print("-" * 54)
        for i, point in enumerate(history):
            print(f"{i}\t{point[0].item():>10.3f}\t\t{point[1].item():>10.3f}\t\t{func(point).item():>10.3f}")
        print("-" * 54)

        print(f'fig saved as {os.path.join(METHOD_DIR, method_name)}_2D')
    # plt.show(block=True)
    plt.close()


# --- Функция для построения 3D графика ---q
def plot_trajectory_3d(history, func, method_name, print_table=True):
    PLOTS_DIR = 'plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)

    METHOD_DIR = os.path.join(PLOTS_DIR, method_name)
    os.makedirs(METHOD_DIR, exist_ok=True)

    mn_x = min(min([h[0] for h in history]) - 1)
    mn_y = min(min([h[1] for h in history]) - 1)
    mx_x = max(max(h[0] for h in history) + 1)
    mx_y = max(max(h[1] for h in history) + 1)
    x_vals = np.linspace(min(mn_x, mn_y), max(mx_x, mx_y), 100)
    y_vals = np.linspace(min(mn_x, mn_y), max(mx_x, mx_y), 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func(np.array([X, Y]))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Рисуем поверхность
    ax.plot_surface(X, Y, Z, cmap='autumn', alpha=0.5)

    # Рисуем траекторию
    trajectory_x = [h[0] for h in history]
    trajectory_y = [h[1] for h in history]
    trajectory_z = [func(point) for point in history]
    ax.plot(trajectory_x, trajectory_y, trajectory_z, 'r--', marker=7, label='Траектория поиска')

    # Рисуем линии уровня на поверхности
    levels = np.linspace(np.min(Z), np.max(Z), 10)  # Определяем уровни для контуров
    ax.contour3D(X, Y, Z, levels, cmap='autumn', linewidths=2)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$F(X)$')
    ax.set_title(f'Метод: {method_name}')
    ax.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{os.path.join(METHOD_DIR, method_name)}_3D.svg", format='svg')
    plt.savefig(f"{os.path.join(METHOD_DIR, method_name)}_3D")

    if print_table:
        # Вывод траектории в виде таблицы
        print(f"\nТраектория поиска ({method_name}):")
        print("-" * 54)
        print(f'i\t{"X_1":^14}\t{"X_2":^14}\t{"F_X":^14}\t')
        print("-" * 54)
        for i, point in enumerate(history):
            print(f"{i}\t{point[0].item():>10.3f}\t\t{point[1].item():>10.3f}\t\t{func(point).item():>10.3f}")
        print("-" * 54)

    print(f'fig saved as {os.path.join(METHOD_DIR, method_name)}_3D')
    plt.close()
