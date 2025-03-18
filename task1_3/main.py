from utils import objective_function
from utils import gradient as utils_gradient
from utils import hessian as utils_hess
from make_graphs import plot_trajectory
import numpy as np
import numpy.linalg as la

# --- Задание коэффициентов ---
C11 = -31
C22 = -34
C12 = 4
C1 = 286
C2 = 388

reference_answer = np.array([[5], [6]])  # Аналитический ответ


# X - вектор из двух значений (x1,x2).T
# Пусть например он изначально будет равен (0, 1).T
def get_random_X(low, high):
    x1 = np.random.randint(low, high)
    x2 = np.random.randint(low, high)
    X = np.array([x1, x2], dtype="float64").reshape(1, 2).T
    return X


#  Необходимое условие максимума grad_f (X*) = 0
#  Достаточное условие максимума hess_f (X*) - отрицательно определенна
# Наша функция выглядит следующим образом -
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


hessian = utils_hess(
    C11=C11,
    C22=C22,
    C12=C12
)


# Шаг вычисляется следующим образом
# Выбор длины шага
# t_i = t = const
# Алгоритм регулировки длины шага
# Использование методов одномерного поиска
# Оптимизация длины шага:
# t_i = - (grad_f_(X_i).T  @ K_i)/( K_i.T @ H(X_i) @ K_i)

# Правила остановки
# Норма градиента стала меньше d1
# Норма разности X стала меньше d2
# Модуль разности нового значения функции со старым становится меньше Eps

def statement_grad(grad_X_i, delta):
    return la.norm(grad_X_i) <= delta


def statement_args(X_new, X_old, delta):
    return la.norm(X_new - X_old) <= delta


def statement_values(f_new, f_old, delta):
    return abs(f_new - f_old) <= delta


def K_relax(grads, i):
    grad = grads[-1]
    K = np.zeros(shape=grad.shape)
    indx = (i + 1) % len(grad)
    K[indx] = grad[indx]
    return K


def K_newton(grads, hessian):
    inv_hesse = la.inv(hessian)
    K = - inv_hesse @ grads[-1]
    return K


def K_GA(grads):
    # Quickest gradient ascent
    # Наискорейший подъем
    K = grads[-1]
    return K


def K_conjugate_grads(grads, i):
    if i == 0:
        return grads[-1]
    grad_new = grads[-1]
    grad_old = grads[-2]
    relation_sqr = (la.norm(grad_new) / la.norm(grad_old)) ** 2
    K = grad_new + relation_sqr * grad_old
    return K


def K_Broyden(grads, etas):
    return -etas[-1] @ grads[-1]


def K_DFP(grads, etas):
    return -etas[-1] @ grads[-1]


optimization_methods = [
    'K_relax',
    'K_newton',
    'K_GA',
    'K_conjugate_grads',
    'K_Broyden',
    'K_DFP'
]


def absolute_optimization(
        optimization_method: str = 'K_DFP',
        stop_rule: int = 0,
        X_borders: int = 1000,
        stop_value: float = 1e-2,
        max_steps: int = 1000,
        epsilon: float = 1e-100,
):
    """

    :param optimization_method: метод оптимизации (str)
    :param stop_rule: правило остановки, 0-2 (int)
    :param X_borders: предел (-, +) генерации вектора X
    :param stop_value: значение с которым сравнивается выражение остановки
    :param max_steps: максимальное количество шагов
    :param epsilon: значение, ниже которого мы делить не будем
    :return:
    """

    # В ячейке -1 самое новое значение, -2 -> предыдущее.
    # Изначально они могут совпадать или отсутствовать, для инициализации

    random_X = get_random_X(-X_borders, X_borders)
    d = stop_value
    X = [random_X]
    f = [func(X[-1])]
    grads = [gradient(X[-1])]
    etas_broyden = [-np.eye(2)]
    etas_DFP = [-np.eye(2)]

    counter = 0
    for i in range(max_steps):
        grad = grads[-1]
        counter = i

        # Выбор функции оптимизации
        if optimization_method == 'K_relax':
            K = K_relax(grads, i)
        elif optimization_method == 'K_newton':
            K = K_newton(grads, hessian)
        elif optimization_method == 'K_GA':
            K = K_GA(grads)
        elif optimization_method == 'K_conjugate_grads':
            K = K_conjugate_grads(grads, i)
        elif optimization_method == 'K_Broyden':
            K = K_Broyden(grads, etas_broyden)
        elif optimization_method == 'K_DFP':
            K = K_DFP(grads, etas_DFP)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}\n"
                             f"methods: {['K_relax', 'K_newton', 'K_GA', 'K_conjugate_grads', 'K_Broyden', 'K_DFP']}")

        denom = K.T @ hessian @ K

        if np.all(np.abs(grad) <= epsilon):
            # print(f'grad = \n{grad}')
            break
        if abs(denom) <= epsilon:
            # print(f'denom = {denom}\t\t\t divided_num = {(grad.T @ K)}')
            # # # # denom += epsilon * np.sign(denom)
            break

        t = - (grad.T @ K) / denom

        current_best = X[-1] + t * K

        # if np.all(np.abs(gradient(current_best)) <= epsilon):
        #     print(grads)
        #     print(X)
        #     break
        # Обновление переменных для следующего цикла
        X.append(current_best)
        f.append(func(X[-1]))
        grads.append(gradient(X[-1]))

        delta_X = X[-1] - X[-2]
        delta_grads = grads[-1] - grads[-2]

        Z = delta_X - etas_broyden[-1] @ delta_grads
        A = (delta_X @ delta_X.T) / (delta_X.T @ delta_grads)
        B = (etas_DFP[-1] @ delta_grads @ delta_grads.T @ etas_DFP[-1].T) / (delta_grads.T @ etas_DFP[-1] @ delta_grads)

        delta_eta_broyden = (Z @ Z.T) / (Z.T @ delta_grads)
        delta_eta_DFP = A - B

        new_eta_broyden = etas_broyden[-1] + delta_eta_broyden
        new_eta_DFP = etas_DFP[-1] + delta_eta_DFP

        etas_broyden.append(new_eta_broyden)
        etas_DFP.append(new_eta_DFP)
        # Мы можем выбрать одно из условий остановки
        stop_checks = [
            lambda: statement_values(f[-1], f[-2], d),
            lambda: statement_args(X[-1], X[-2], d),
            lambda: statement_grad(grads[-1], d)
        ]
        # if any(check() for check in stop_checks):
        if stop_checks[stop_rule]():
            break

    X_ans = X[:counter + 1]
    iterations = counter  # -1 т.к.

    return X_ans, iterations


# Зададим границы генерации значений вектора X
borders = 2 ** 31 - 1

tolerance = 0.001
num_experiments = 1000


def benchmark_methods(
        reference_answer,
        optimization_methods,
        stop_value=1e-4,
        tol=1e-3,
        num_experiments=100

):
    for method in optimization_methods:
        total_ans = []
        total_iterations = []
        for _ in range(num_experiments):
            answers, iterations = absolute_optimization(method, stop_value=stop_value, epsilon=1e-26)
            total_ans.append(answers[-1])
            total_iterations.append(iterations)

        print(f'{'=' * 26} {method} {'=' * 26}')
        total_ans = np.asarray(total_ans)
        total_iterations = np.asarray(total_iterations)
        print(f'TOTAL_ITERATIONS:\n mean = {total_iterations.mean()}\n std = {total_iterations.std()}')

        # Проверка всех ответов на близость к эталонному
        correct_count = np.sum(np.all(np.abs(total_ans - reference_answer) <= tol, axis=1))
        # axis=1 стоит потому что нам надо чтобы и x1 и x2 соответствовали условию

        print(f'Количество правильных ответов: {correct_count}')
        print(f'total_ans mean =\n{total_ans.mean(axis=0)}\n')
        # print(f'{'=' * (54 + len(method))}')


methods_boarders = {
    'K_relax': 10,
    'K_newton': 100,
    'K_GA': 10,
    'K_conjugate_grads': 10,
    'K_Broyden': 100,
    'K_DFP': 10
}

if __name__ == "__main__":
    benchmark_methods(
        reference_answer=reference_answer,
        optimization_methods=optimization_methods,
        stop_value=0.0001,
        tol=tolerance,
        num_experiments=num_experiments
    )

    for method in optimization_methods:
        X_boarders = methods_boarders[method]
        history_X, iterations = absolute_optimization(method,
                                                      stop_value=0.001,
                                                      X_borders=X_boarders
                                                      )
        plot_trajectory(history_X, func, method)
