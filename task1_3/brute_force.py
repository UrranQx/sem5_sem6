from utils import objective_function
from utils import gradient as utils_gradient
import numpy as np

# --- Задание коэффициентов ---
C11 = -31
C22 = -34
C12 = 4
C1 = 286
C2 = 388


def func(Vector):
    return objective_function(
        X=Vector,
        C11=C11,
        C22=C22,
        C12=C12,
        C1=C1,
        C2=C2
    )


def gradient(Vector):
    return utils_gradient(
        X=Vector,
        C11=C11,
        C22=C22,
        C12=C12,
        C1=C1,
        C2=C2
    )


# Попробуем перебором найти максимум
def brute_force(borders=100):
    maximum = -10 * 10
    for i in range(-borders, borders):
        for j in range(-borders, borders):
            X = np.array([i, j]).reshape(1, 2).T

            # print(X)
            if func(X) >= maximum:
                maximum = func(X)
                print('grad = ')
                print(gradient(X))

                print(f'new min found with func(X) = {func(X)},\nX = '
                      f'\n{X}')


if __name__ == "__main__":
    brute_force()
