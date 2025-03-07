import matplotlib.pyplot as plt

import numpy as np
from numpy import linalg as la

# Разберем вариант 46
# Все данные содержаться в папке Variants.
DIRECTORY = 'Variants/'
MAIN_NAME = 'testLab1Var'
END = '.csv'
VARIANT_NUM = '46'  # '46'

data = np.genfromtxt(DIRECTORY + MAIN_NAME + VARIANT_NUM + END, delimiter=',')

time = data[:, 0]  # Время - это первый столбец
current = data[:, 1]  # Сила тока - второй столбец
voltage = data[:, 2]  # Напряжение - третий
time = time[:, np.newaxis]
current = current[:, np.newaxis]
voltage = voltage[:, np.newaxis]

# Запишите значение 11-го элемента массива напряжений и временного массива.
k = 11  # k>0
print(f'{k}-й элемент voltage: {voltage[k - 1]}\n'
      f'{k}-й элемент time: {time[k - 1]}')

# 11-й элемент voltage: [1.]
# 11-й элемент time: [0.01]

fig, (voltage_plot, current_plot) = plt.subplots(2, 1, sharex=True)

# voltage_plot.plot(time, voltage)
# current_plot.plot(time, current)
# У нас очень много данных, поэтому такое отображение о многом нам не расскажет

# Воспользуемся индексированием с помощью булевых масок в numpy, чтобы отфильтровать требуемый период
T_PERIOD = 0.1  # Задано в условии эксперимента

voltage_plot.plot(time[time < 2 * T_PERIOD], voltage[time < 2 * T_PERIOD])
current_plot.plot(time[time < 2 * T_PERIOD], current[time < 2 * T_PERIOD])

# Добавим клеточную разметку
voltage_plot.grid()
current_plot.grid()

# Добавим подписи к оси x и y соответственно
voltage_plot.set_xlabel('time, s')
voltage_plot.set_ylabel('voltage, V')

current_plot.set_xlabel('time, s')
current_plot.set_ylabel('current, A')

# Уберите комментарий, чтобы он показывал график при каждом запуске программы
# plt.show()

# Сохраним полученный рисунок
fig.savefig('_Recieved data(part)')
plt.close()

# Матрица X согласно теории будет выглядеть следующим образом
X = np.concatenate([voltage[0:-2], current[0:-2]], axis=1)
Y = current[1:-1]

K = la.inv(X.T @ X) @ X.T @ Y
# Это же выражение в методичке написано как K = np.dot(np.dot(la.inv(np.dot(X.T, X)), X.T), Y)

Td = 0.001  # Задано в условии эксперимента

R = 1 / K[0] * (1 - K[1])  # То же самое что и (1 - K[1]) / K[0]
T = -Td / np.log(K[1])
L = T * R

print(f'Расчетные значения сопротивления и индуктивности:\n'
      f'R = {R}\n'
      f'L = {L}')
# Расчетные значения сопротивления и индуктивности:
# R = [2.57400698]
# L = [0.27002006]

current_est = X @ K

fig, ax = plt.subplots(1, 1)
plt.plot(time[time < T_PERIOD], current[time < T_PERIOD], label='real data')
plt.plot(time[time < T_PERIOD], current_est[time[0:-2] < T_PERIOD], label='estimated data')
ax.grid()
ax.set_xlabel('time, s')
ax.set_ylabel('current, A')
ax.legend()
fig.savefig('_Compared data(part)')

# Уберите комментарий, чтобы он показывал график при каждом запуске программы
# plt.show()
plt.close()

R_est = []
L_est = []
n = 1000  # Количество периодов в тестовом сигнале, вычисляется так: print(time[-1] / T_PERIOD) # 1000

for i in range(0, n - 1):  # По умолчанию шаг и так равен 1
    ind = (time >= T_PERIOD * i) & (time <= T_PERIOD * (i + 1))  # Маска для индексирования i-го промежутка
    new_current = current[ind]
    new_voltage = voltage[ind]
    new_current = new_current[:, np.newaxis]
    new_voltage = new_voltage[:, np.newaxis]
    X_ = np.concatenate([new_voltage[1:-1], new_current[0:-2]], axis=1)
    Y_ = new_current[1:-1]  # TODO: В методичке почему-то используется обычный массив current / Поправка, в англ. методе
    K_ = la.inv(X_.T @ X_) @ X_.T @ Y_

    if K_[1] > 0:  # Натуральный лог K[1] имеет смысл только в этом случае
        R_ = 1 / K_[0] * (1 - K_[1])
        T_ = -Td / np.log(K_[1])
        R_est.append(R_)
        L_est.append(T_ * R_)

R_est = np.array(R_est)
L_est = np.array(L_est)

print(f'Mean value of R:\t\t\t {np.mean(R_est)} Ohm')
print(f'Standard deviation of R:\t {np.std(R_est)}')

print(f'Mean value of L:\t\t\t {np.mean(L_est)} Hn')
print(f'Standard deviation of L:\t {np.std(L_est)}')

# Mean value of R:			 1.9460125109477884 Ohm
# Standard deviation of R:	 0.16134122591606292
# Mean value of L:			 0.2767766603689984 Hn
# Standard deviation of L:	 0.002333117109094485
