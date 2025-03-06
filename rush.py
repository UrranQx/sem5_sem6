import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Коэффициенты
C11 = -31
C22 = -34
C12 = 4
C1 = 286
C2 = 388


# Целевая функция
def objective(x):
    x1, x2 = x
    return C11 * x1 ** 2 + C22 * x2 ** 2 + C12 * x1 * x2 + C1 * x1 + C2 * x2


# Градиент (первые производные)
def gradient(x):
    x1, x2 = x
    dfdx1 = 2 * C11 * x1 + C12 * x2 + C1
    dfdx2 = 2 * C22 * x2 + C12 * x1 + C2
    return np.array([dfdx1, dfdx2])


def gesse():
    d2fdx1_2 = 2 * C11
    d2fdx2_2 = 2 * C22
    d2fdx1dx2 = C12
    return np.array([[d2fdx1_2, d2fdx1dx2], [d2fdx1dx2, d2fdx2_2]])


# noinspection PyShadowingNames


if __name__ == "__main__":
    starting_point = [0, 0]  # Начальная точка
