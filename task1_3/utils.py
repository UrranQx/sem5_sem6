import numpy as np


# --- Общая функция для целевой функции и градиента ---
# noinspection PyShadowingNames
def objective_function(X, C11, C22, C12, C1, C2):
    return C11 * X[0] ** 2 + C22 * X[1] ** 2 + C12 * X[0] * X[1] + C1 * X[0] + C2 * X[1]


# noinspection PyShadowingNames
def gradient(X, C11, C22, C12, C1, C2):
    grad_x1 = 2 * C11 * X[0] + C12 * X[1] + C1
    grad_x2 = 2 * C22 * X[1] + C12 * X[0] + C2
    return np.array([grad_x1, grad_x2]).reshape(1, 2).T


# noinspection PyShadowingNames
def hessian(C11, C22, C12):
    return np.array([[2 * C11, C12],
                     [C12, 2 * C22]])
